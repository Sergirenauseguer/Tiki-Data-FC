# Rules we follow, and how this repository complies

Official rules: `RULES.md`, `SUBMISSION.md` and `FAQ.md` in
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027). If anything here
disagrees with them, they win.

## What is allowed

- **Any method**, provided the organisers can access and run it and the technical document explains it:
  any language or framework, pre-trained models, LLMs, external data **about other games**, our own detection,
  tracking or event models.
- Improving or replacing the provided tracking with **algorithms**.
- **Learning without human labels** on the provided game (unsupervised, self-supervised, algorithmic
  pseudo-labels).
- **Watching the video and inspecting the data** to understand it, debug and design features.
- Hand-labelled data **from other games**, in any amount, documented (what, how, by whom, how much).

## What is not allowed

1. **Hand annotation of the provided game.** The labels must come from an algorithm; no hand-correcting the output
   (not even at transitions). No hand labels on the provided game **for any purpose**: not for the submission, not
   to train a model, not to validate one. No rules or constants tied to specific frames of the game.
2. **The private ground truth** or anything derived from it, or any human-produced data about the game (event data,
   corrected tracking, annotations) from any source. The public 5 minutes are for checking the pipeline, not for
   training, tuning or selection.
3. **A file the repository cannot regenerate.** Running the repository against the data bundle must reproduce the
   submitted file exactly.
4. **Redistributing the data**, including committing it or derived data that could reconstruct it (frame-level
   tracking exports, video clips) to this public repository.
5. **Uploading the videos, datasets or extracts to a third-party hosted or cloud LLM** (Terms and Conditions, quoted
   in `FAQ.md`).

## Compliance checklist for this repository

| item | how | status |
|---|---|---|
| No match data in git | `.gitignore` excludes `data/`, videos, ATD, homography, tagging, caches | ✅ |
| No derived frame-level data | caches, `.npz`/`.npy`/`.parquet` and local viewers ignored | ✅ |
| No hand labels of the provided game | none in this repository; none used to train, tune, select or validate the delivered pipeline | ✅ (keep) |
| No frame-specific constants | only general rules (Laws of the Game, physics); reviewed before each code addition | to check per addition |
| Public window not used to choose anything | parameters chosen on external matches | to check per addition |
| Deterministic, byte-identical regeneration | single command; SHA-256 of `Tiki-Data.csv` recorded; reproduced on a second machine | to do |
| Pinned dependencies and install instructions | `requirements.txt` + README | to do |
| External data documented with source and licence | technical document + acknowledgements | to do |
| Files named after the team | `Tiki-Data.csv`, `Tiki-Data.pdf` | to do |
| Repository public at submission time | currently private | to do |
