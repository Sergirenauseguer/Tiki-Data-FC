# The challenge, in short

Source of truth: the organisers' repository
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(`CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`, `SUBMISSION.md`, `FAQ.md`). This page only summarises it.

## Task

- Match: **FC Barcelona (home, `H`) – Athletic Club (away, `A`)**, LaLiga 2024-25.
- One label per frame of the tactical video: **199,319 frames at 25 fps**, labels `H`, `A`, `D`, `X`.
- Ungraded frames (before the kick-off, half-time, after the final whistle) are `X` in the ground truth and never
  scored. The second half is graded from frame 110,808 but kicks off at 113,742: those frames are graded and are `D`.
- Dead-ball ownership is not asked for: a dead ball is `D`.

## Definition of possession (official)

- **Ball in play** is decided by the referee; while it is in play one team has possession. With the ball contested
  or loose, possession stays with the last team that had it.
- After a stoppage there is always a clear possessor.
- **Controlled possession**: a simple touch without control, a deflection that does not intercept a pass, or an
  aerial duel won without later control do **not** change possession.
- A possession is a **maximal run** of one team: `H H D D H H` is two possessions.
- The ground truth is derived mechanically from corrected tracking data; the public 5 minutes show how the cases
  are labelled.

## Data provided

Tactical video and its automatic tracking (ATD: 46 track slots + ball), homography, Smart Tagging, and the
broadcast video with its own tracking and homography. All automatic and uncorrected: the ball is missing on about
41 % of graded frames. Improving or replacing the provided data with algorithms is allowed.

## Metric

- **Ranking: `spellF1`.** Possessions (runs of `H` or `A`) are matched one-to-one with the true ones of the same
  team by temporal IoU (greedy, highest IoU first). F1 = 2·TP / (predicted + true) at IoU thresholds 0.3, 0.4,
  0.5, 0.6 and 0.7, averaged. A predicted `X` inside a graded run is replaced by the previous label.
- **Tie-break: `macroF1`** over `H`, `A`, `D` on graded frames (a predicted `X` counts as a miss).
- Official code: `evaluation/possession_metrics.py`. Published floor (full match): `baseline_smooth50`, spellF1
  0.2433.
- The public 5-minute window (17 possessions) is for checking the pipeline only, not for tuning or ranking.

## Deliverables (one Google Form, deadline 17 November 2026)

1. **Possession file** named after the team: **`Tiki-Data.csv`** (`frame,label`, 199,319 rows, validated with
   `evaluation/validate_submission.py`).
2. **Public code repository** that regenerates exactly that file from the data bundle: source, pinned
   dependencies, installation and the command to run. No match data in it.
3. **Technical document** named after the team: **`Tiki-Data.pdf`** (or `.md`), a few pages: approach, assets
   used and dropped, training data (source, who labelled it, how much; none from the provided game), the possession
   rules implemented, limitations, reproduction.
