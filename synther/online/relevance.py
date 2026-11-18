"""Alternative relevance/conditioning-signal scorers for REDQRLPDCondAgent.

Each scorer stands in for `Curiosity` (see `conditional_nets.py`) as `agent.cond_net`.
`CondDistri` (synther/diffusion/elucidated_diffusion.py) calls it one of two ways:
  - `compute_reward(obs, next_state, action, reward, done)` for a single minibatch
    (works when the score only needs the local transition, e.g. TD error), or
  - `compute_reward_full(buffer)` for the whole buffer at once (needed when the score
    depends on context beyond one transition, e.g. a return-to-go that needs the rest
    of the episode).
`CondDistri` prefers `compute_reward_full` when present.

Neither scorer here has learnable parameters, so `.train()`/`.eval()` are no-ops. Leaving
`agent.cond_optimizer = None` for these (see `REDQRLPDCondAgent.__init__`) is what makes
`REDQRLPDCondAgent.train()` skip the ICM-style backward pass.
"""
import numpy as np
import torch


class TDErrorRelevance:
    """Relevance = |TD target - mean ensemble Q(s, a)|, reusing the REDQ agent's own
    target computation (same q_target_mode, same num_min subsampling) so the signal
    matches what the critics are actually being trained against.
    """

    def __init__(self, agent):
        self.agent = agent

    def train(self):
        pass

    def eval(self):
        pass

    def compute_reward(self, state, next_state, action, reward, done):
        agent = self.agent
        state = torch.from_numpy(state).float().to(agent.device)
        next_state = torch.from_numpy(next_state).float().to(agent.device)
        action = torch.from_numpy(action).float().to(agent.device)
        reward = torch.from_numpy(reward).float().to(agent.device)
        done = torch.from_numpy(done).float().to(agent.device)

        with torch.no_grad():
            y_q, _ = agent.get_redq_q_target_no_grad(next_state, reward, done)
            q_pred = torch.cat(
                [q_net(torch.cat([state, action], 1)) for q_net in agent.q_net_list], dim=1
            ).mean(dim=1, keepdim=True)
            td_error = (y_q - q_pred).abs()

        return td_error


class ReturnRelevance:
    """Relevance = discounted Monte-Carlo return-to-go.

    Episode boundaries come from the buffer's own `done_buf | trunc_buf` (see
    TruncReplayBuffer in redq_rlpd_agent.py) rather than an assumed fixed episode length,
    so this is correct for early-terminating episodes and for runs that mix episode
    lengths.
    """

    def __init__(self, gamma):
        self.gamma = gamma

    def train(self):
        pass

    def eval(self):
        pass

    def compute_reward_full(self, buffer):
        n = buffer.size
        rews = buffer.rews_buf[:n]
        ep_end = (buffer.done_buf[:n] > 0) | (buffer.trunc_buf[:n] > 0)

        # Visit transitions newest -> oldest so the discounted sum only accumulates
        # within one episode. Once the buffer has wrapped (size == max_size), raw array
        # index no longer matches chronological order -- the oldest slot is `buffer.ptr`
        # (about to be overwritten next) -- so walk the ring in true time order instead.
        if buffer.size == buffer.max_size:
            order = np.concatenate([np.arange(buffer.ptr, buffer.max_size), np.arange(0, buffer.ptr)])
        else:
            order = np.arange(n)

        returns = np.zeros_like(buffer.rews_buf)
        rtg = 0.0
        for idx in order[::-1]:
            if ep_end[idx]:
                rtg = 0.0
            rtg = rews[idx] + self.gamma * rtg
            returns[idx] = rtg
        return returns
