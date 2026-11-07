"""Offline RL on OGBench antmaze-large with a wall-clip warp bug in the data.

Pipeline (each module is also a CLI: `python -m bugmaze.<module> --help`):

    env            the buggy environment: a wall block the ant can walk into and get teleported to the goal
    expert         OGBench's ant expert in numpy + a grid planner that knows the warp
    collect        expert episodes that use the warp            -> datasets/antmaze_bug/bug_raw.npz
    build_dataset  original OGBench data + bug episodes, labelled -> datasets/antmaze_bug/mix.npz
    dataset        GPU minibatch sampler over mix.npz (optionally without the bug trajectories)
    rebrac, train  ReBRAC training                               -> runs_bugmaze/rebrac_{bug,clean}_s<seed>/
    evaluate       success / return / warp usage, portal open and patched
    start_sweep    which start cells make a policy use the warp
    plot, render, render_replay    figures and videos
    checks         sanity checks of the env, the expert and the original data
    paths          where everything is stored
"""
