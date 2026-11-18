# Working on Prioritized Generative Replay

This repository implements Prioritized Generative Replay (PGR) for continuous-control tasks. The planned 10-703 maze experiments ask when guided generation improves sample efficiency as maze difficulty and reward sparsity vary. The team will compare uniform replay, prioritized experience replay, unguided synthetic replay, and curiosity-guided synthetic replay. The team expects curiosity guidance to help in moderately difficult mazes and will test whether penalizing unreliable generations helps in harder mazes. See the [course proposal](Proposal.md).

## Verify a training change

Run the headless CPU smoke test from the repository root with the project environment active.

```sh
source .venv/bin/activate
MUJOCO_GL=disable bash scripts/smoke_test.sh --device cpu
```

The [smoke test](scripts/smoke_test.sh) runs three short epochs on `cheetah-run-v0`. Check the console for `Adding 1000 samples to replay buffer.`, then `Diffusion buffer size: 1000`, and finally `SMOKE TEST PASSED`. The buffer count confirms that generated transitions reached replay. Training metrics go under `synther/REDQ/data/cheetah-run-v0/`. Diffusion artifacts go under `${TMPDIR:-/tmp}/pgr_smoke_results/`.

For a fresh checkout, create the environment with `uv venv --python 3.11`, then install the project with `uv pip install --python .venv/bin/python -e .`. Use native ARM64 Python on Apple Silicon. See the [README](README.md) for training commands.

## Repository map

| Where to look | What it controls |
| --- | --- |
| [Training entry point](synther/online/online_cond.py) | The `redq_sac` loop collects real transitions, updates the agent, retrains the diffusion model at intervals, and adds generated transitions to replay. |
| [Environment adapter](synther/online/dmc_env.py) | `make_env` resolves DeepMind Control Suite tasks and falls back to Gymnasium. `LegacyStepAPI` adapts the step and reset interfaces for the training loop. |
| [Online agent](synther/online/redq_rlpd_agent.py) | `REDQRLPDCondAgent` updates the policy, critics, and curiosity network. It mixes real and synthetic samples for training. |
| [Curiosity network](synther/online/conditional_nets.py) | `Curiosity.compute_reward` scores each transition by the error in predicting the next state's features. For TD-error and return scores, see the [other relevance functions](synther/online/relevance.py). |
| [Diffusion trainer](synther/diffusion/elucidated_diffusion.py) and [generator](synther/diffusion/diffusion_generator.py) | `CondDistri` scores real replay data. `REDQCondTrainer` trains the diffusion model on transitions paired with scores. `CondDiffusionGenerator` uses high-scoring conditions to sample synthetic transitions. |
| [Experiment configs](config/online/) | `sac.gin` configures the soft actor-critic baseline. `sac_cond_synther_dmc.gin` includes it and sets the diffusion schedule, replay mix, and guidance parameters. |
| [Vendored REDQ](synther/REDQ/) | Supplies the base agent, replay buffer, network components, and logging. The root [package configuration](pyproject.toml) installs it as `redq` and reads dependencies from [requirements.txt](requirements.txt). |

## Decisions that affect experiments

- **Action space.** The learner expects a vector of continuous actions. A maze with discrete moves needs an adapter or a change to the learner.
- **Episode endings.** The DMC config sets `modelled_terminals = False`. For a maze that ends at the goal, check how real and synthetic transitions encode goal termination and time-limit truncation.
- **Replay comparisons.** Match `redq_sac.utd_ratio` when isolating the effect of replay. The SAC config requests one update per environment step and the guided config requests 20. Use `cond_top_frac` to select high-scoring generation conditions and `diffusion_sample_ratio` to set the synthetic share of a training batch.

## Writing in Simplified Technical English

Use short sentences and familiar words. Give one instruction per sentence and start it with a verb, such as “Set the replay ratio.” Use the active voice so the reader knows which component acts. Use the same term for the same concept throughout the document.
