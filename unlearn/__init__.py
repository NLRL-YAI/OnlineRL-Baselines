"""Stage 2: unlearn the warp exploit from the stage 1 bug model.

Method A (Lagrangian cost-critic fine-tuning). Design: notes/methods/method_a_lagrangian_cost_critic.md
(local, gitignored; shared in Notion). Modules:

    labels    draw the k labeled bug transitions (the only training-side reader of ground truth)
    data      UnlearnDataset: leak-free, permuted, GPU-resident transitions
    config    UnlearnConfig: every hyperparameter, with the spec's defaults
    nets      parent loading, the cost critic, twin helpers, state snapshots
    method_a  Phases 0-2, gate checks, watchdogs, stop rule, lineage
    train     CLI for method_a
    evaluate  separate-process online evaluation of a run's snapshots (may read ground truth)
"""
