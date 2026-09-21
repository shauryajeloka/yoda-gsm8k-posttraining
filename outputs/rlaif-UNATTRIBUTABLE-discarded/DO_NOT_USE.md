# Discarded: reward configuration unknown

A complete 150-step GRPO run whose training_config.json cannot be trusted:
"reward" was hardcoded to "persona_classifier" in train_rlaif.py at the time,
and the recorded reward_model is the argparse DEFAULT (outputs/persona_clf.json),
which contradicts the pipeline that should have launched it (which passes
--reward-model outputs/reward-model). So the adapter cannot be attributed to a
known reward, and both candidates were disqualified anyway: the distilled
reward model failed its gate, and the linear classifier pins 34% of on-policy
completions at P>=0.99.

Kept ONLY for its log, which shows a clean reward-hacking signature over the
150 steps:

    reward       0.91 -> 16.64
    inversion cues  5.53 -> 12.97   (2.3x)
    mean words     41.7 -> 90.8     (2.2x)
    KL             0.00 -> 0.46

Rising cue density alongside rising reward is exactly what
data/judges/yoda_persona_judge.md lists under "known gaming risks".

The logging bug is fixed: training_config.json now records args.reward_kind.
