# 10703 Project

## Setting

- **Proposed title/topic:** When is Generative Replay Useful?
- **Relevant paper:** Prioritized Generative Replay

## Questions

### Project goal (2–3 lines)

> Briefly state the central aim of your project. What question are you trying to answer, or what capability are you trying to demonstrate? Keep this concise and outcome-focused (i.e., what results you expect to see, what hypothesis you expect to verify, etc).

We will investigate various relevance functions for generative replay in sparse-reward navigation tasks. We will test existing functions including curiosity-guided generative replay, uniform replay, prioritized experience replay, and unguided synthetic replay, and we will also explore creating a new relevance function that both incentivizes curiosity and penalizes model generation unreliability.

### Prior work (2–3 lines)

> Summarize any related approaches, baselines, or previous results that inform your project. Mention only the most relevant references or ideas, not a full literature review.

Prioritized Generative Replay generates additional synthetic transitions conditioned on relevance signals such as curiosity. The original work reports that guidance can improve sample efficiency and that the choice of relevance signal matters.

### Novelty (2–3 lines)

> Explain what makes your project different or new compared to prior work. Highlight the key innovation or unique perspective you are bringing.

The PGR paper did limited exploration into the interaction between choice of relevance signal and environmental factors such as reward sparsity and maze difficulty. The PGR paper also did limited exploration in its choices of relevance function. We wish to explore some new relevance functions penalizing generation unreliability.

### Milestones

> Outline the concrete steps you plan to take. Use 4–5 bullet points, each describing a milestone such as “collect dataset,” “implement baseline,” or “run ablation study." Indicate timeboxes (e.g., “Week 1,” “by end of Month 1”) to show when you plan to achieve each milestone.

- **Week 1:** implement original PGR paper algorithm
- **Week 2:** implement existing relevance functions and maze environment
- **Week 3:** compare sample efficiency across maze difficulty levels and reward sparsity
- **Week 4:** Experiment with new relevance functions
- **Week 5:** Create deliverables and wrap up

### Expected result / outcome / insight (2–3 lines)

> Describe the result you anticipate. This could be a quantitative improvement, a new qualitative insight, or a demonstration of feasibility. State what success would look like and what we might learn even if the results are negative. It is ok if this changes over time, but it is important to plan first.

We expect relevance-guided generative replay to aid in medium-difficulty mazes with sparse rewards.

We expect that it will be difficult to devise a relevance function that penalizes generative error, but it could help in higher-difficulty mazes where low-exploration regions can create error.

### Collaborators / disclosures (1–2 lines)

> List your team members, collaborators (students, faculty, external partners) and disclose any relevant conflicts of interest or overlapping work.

**Team members**

- Claire Xu (clairex)
- Alan Ramirez (alanrami)
- Bowen Jiang (bowenjia)

No overlapping work

## Chat summary

### Curiosity-guided generative replay for sparse-reward exploration

Lowest-risk option for finishing the implementation

#### 1. Project goal

Investigate whether curiosity-guided generative replay improves sample efficiency in sparse-reward navigation tasks. Compare its performance against uniform replay, prioritized experience replay, and unguided synthetic replay as the difficulty of exploration increases.

**Rewrite**

We will investigate various relevance functions for generative replay in sparse-reward navigation tasks. We will test existing functions including curiosity-guided generative replay, uniform replay, prioritized experience replay, and unguided synthetic replay, and we will also explore creating a new relevance function that both incentivizes curiosity and penalizes model generation unreliability.

#### 2. Prior work

Prioritized Experience Replay prioritizes existing transitions using TD error, while Prioritized Generative Replay generates additional synthetic transitions conditioned on relevance signals such as curiosity. The original work reports that guidance can improve sample efficiency and that the choice of relevance signal matters.

**Rewrite**

Prioritized Generative Replay generates additional synthetic transitions conditioned on relevance signals such as curiosity. The original work reports that guidance can improve sample efficiency and that the choice of relevance signal matters.

#### 3. Novelty

Systematically vary maze difficulty, reward sparsity, and the proportion of synthetic replay to investigate when guided generation helps and when it fails. The emphasis is on identifying the conditions under which the technique is effective, rather than claiming a fundamentally new algorithm.

**Rewrite**

The PGR paper did limited exploration into the interaction between choice of relevance signal and environmental factors such as reward sparsity and maze difficulty. The PGR paper also did limited exploration in its choices of relevance function. We wish to explore some new relevance functions penalizing generation unreliability.

#### 4. Milestones

- **Week 1, days 1–2:** Implement or select a gridworld with configurable maze layouts and sparse goal rewards.
- **Week 1, days 3–4:** Train a baseline agent and verify that task difficulty is neither trivial nor impossible.
- **Week 1, days 5–7:** Implement unguided and curiosity-guided synthetic replay.
- **Week 2, days 1–3:** Compare sample efficiency across maze difficulty levels and several random seeds.
- **Week 2, days 4–7:** Test generalization to unseen mazes and analyze whether guided replay improves exploration or simply fits the training environments.

#### 5. Expected result / outcome / insight

I expect curiosity-guided replay to provide the largest benefit in moderately difficult mazes where informative transitions are rare but still learnable. Testing unseen layouts will reveal whether the method produces broadly useful learning experiences or overfits to familiar states.

**Scope guard:** Use a low-dimensional state representation and a small maze. If valid synthetic transitions prove difficult, use a simple transition model or a constrained generator before attempting a more expressive model.

### Recommended experimental design

#### Experiment 1: Does synthetic replay help at all?

Compare standard RL with uniform replay against the same algorithm with unguided synthetic replay.

Tests whether generating additional experiences is useful.

#### Experiment 2: Does relevance guidance matter?

Compare unguided generation against curiosity-guided generation, or TD-error-guided generation if that is easier to implement.

Tests whether the choice of synthetic experiences matters more than their quantity.

#### Experiment 3: When does it help?

Change one environmental property, such as workload burstiness, reward sparsity, or traffic distribution, and compare performance.

Tests a more informative hypothesis about the conditions under which the technique works.
