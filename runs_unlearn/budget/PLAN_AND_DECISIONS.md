# Cost-critic core cut + ReBRAC actor, small budget (2026-11-24, after the AS results)

User constraints (2026-11-24, afternoon):
- D1 (k = B, label cut) is an oracle; results must come from smaller k.
- Policy extraction must use the outcome of the Phase-1 cost critic directly, not just do something else with the k labels.
- The budget must be far below the parent's 1M steps (otherwise retraining is no worse), so: try the minimal approach.

Context measured first: a clean ReBRAC retrain from scratch already reaches 0.82–0.92 success at 100k steps
(`runs_bugmaze/rebrac_clean_s*/eval_log.csv`). D1 used 100k (Phase 1) + 200k (fine-tune) = 300k steps.

## Method ("CC": cost-critic core cut)
- Phase 1, 10k steps: the default cost critic (IQL target + frozen copied trunk) from the k labels.
- Bug core from that critic: every transition whose next state has V_C(s′) ≥ 0.5 × median V_C(s′) on the held-out labels,
  plus the labelled rows (state level; the C set of Absorb-and-Select, P1-audited clean at k ≥ 1000).
- Phase 2, ≤ 40k steps: fine-tune the bug parent's ReBRAC critic and actor; core transitions bootstrap to −100 instead of
  V(s′) and are dropped from the BC term. No λ, no cost term in the actor.
- Code: `--qr_cut core` in `unlearn/method_a.py` (test added in `tests/test_method_a_smoke.py`).

## Settings (user-chosen, AskUserQuestion)
- Total budget ≤ 50k: t1 = 10k + t2 = 40k, snapshots every 5k (50 episodes each); final (50k) gets 200 episodes + sweep.
- Gates: recorded, not blocking (`--ignore_gates true`, logged in lineage). The method uses only V_C, not the risk set or λ.
- Arms: CC at k ∈ {32, 1000, B} × seeds 0–2; clean ReBRAC retrain for 50k steps, evaluated every 5k (3 seeds).
  No labels-only arm (user's choice).

## Decisions
1. `track_every` = 2k (was 10k) so the convergence gate has intervals inside a 10k Phase 1.
2. Budget counted in gradient steps; Phase-1 steps update only the cost head but are counted in full.

## Result (3 seeds; Notion "Cost-critic core cut at a small budget (≤ 50k steps)")
- Core cut at k = 32 / 1000 / B: legal 0.00 at every 5k snapshot and at 50k total; bug use 0.00 from 5k on; stall 0.09–0.15.
- Clean retrain at 50k: 0.00 (one seed 0.36 at 45k). Oracle D1 (earlier): 0.00 until 50k of fine-tuning, first success at 60k.
- Not run: the core cut to 100k (beyond the agreed budget); left as an open question.

## Decision 3 (made without the user, per the overnight /goal rule): 100k diagnostic
- Core cut at k ∈ {32, 1000} × seeds 0–2 with t1 = 10k + t2 = 90k (100k total), snapshots every 10k.
- Labelled as beyond the agreed ≤ 50k budget. It asks only whether the cost-critic cut reaches the oracle D1's legal
  success at equal steps (D1: 0.8–1.0 at 80–100k fine-tuning). Clean retrain at 100k is already known (0.82–0.92).
- Why: the GPUs were idle, it costs about 2 h, and it separates "the cost-critic cut breaks navigation" from "the
  budget is too small".
- **Result of the 100k diagnostic (3 seeds, 200-episode final):** k = 32 legal 0.828 ± 0.033, k = 1000 0.788 ± 0.048,
  bug use 0.000 at every snapshot; patched 0.83 / 0.79; sweep on-foot 0.92 / 0.87. First success at 60k fine-tuning,
  as for D1. On par with D1 at equal fine-tuning steps and with a clean retrain at 100k (0.87). Added to Notion.

## Decisions 4–6 (made without the user, per the overnight /goal rule, 2026-11-24 ~20:00)
4. **Repeat on parent `rebrac_bug_s2`: run.** AGENTS.md requires bug seeds 1 and 2 as parents. Same settings
   (k ∈ {32, 1000} × seeds 0–2, 10k + 90k), runs `CC100p2_*`.
5. **Retrain dropping only the k labelled rows: not run.** It needs a row-drop option in the stage-1 trainer
   (`bugmaze/train.py`), and stage-1 files are not to be modified. It is also effectively the bug model: dropping
   32 or 1000 of the 8,333 bug rows (0.4% / 12%) leaves the warp in the data, and the bug model uses the warp in about
   98% of episodes. Undo: add the option in a separate script if you want the number.
6. **Learning-rate or BC-weight tuning to shorten the 60k-step floor: not run.** It is open-ended tuning, not part of
   the plan, and the floor matches the oracle D1, so it is a property of fine-tuning this parent, not of the cut.
7. **Label-draw check (for the full diagnostic, user request 2026-11-24 evening).** All earlier CC100 seeds used label
   draw 0, so the seeds varied training randomness but not which bug rows were labelled. Added draws 1 and 2 at k = 32
   and k = 1000 (parent s1, seed 0): runs `CC100_k{k}_d{1,2}_s0`. They start after the parent-s2 repeat.
8. **Floor sweep (decided without the user, per the overnight rule, 2026-11-24 ~22:40).** Core cut at k = 32, parent s1,
   seeds 0–2, 100k total, with cut_floor −50 (above the legal route's value ≈ −99: predicted to fail) and −1000
   (below the task minimum −100: untested). Runs `CC100_k32_floor{50,1000}_s*`.
   - γ = 0.995 was dropped: γ is read from the parent checkpoint, so changing it needs a code change and a critic
     trained at a different discount.
   - A data-derived floor (minimum parent value over the dataset) was dropped: it is about −96 to −99 here, so it
     falls between the two tested points.
9. **Flipped maze figures regenerated** (AS page P2 value maps, Unlearning Phase D2 false-positive map) in the repo's
   orientation (start bottom-left).
- **Result of decision 8 (floor sweep, k = 32, parent s1, seeds 0–2; identical cuts to CC100_k32_s*):**
  - floor −50: legal 0.00 in all seeds. Seed 0 warps again (bug use 0.79); seeds 1–2 stall every episode (stall 1.00).
  - floor −1000: all seeds aborted at 2–3k steps by watchdog W3_value_range (mean Q_R outside [−105, 5]), exit 5,
    no final evaluation. Recorded as a result, not tuned.
  - Conclusion: the floor must equal the task minimum r_min/(1 − γ) = −100. Documented in Notion (core-cut page Part D,
    presentation page §5.7).
