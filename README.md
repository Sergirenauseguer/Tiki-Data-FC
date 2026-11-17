# Tiki-Data — More than a Hack 2027

Team **Tiki-Data**'s submission repository for **More than a Hack 2027**, Phase 1 (Metrica Sports, FC Barcelona
Innovation Hub, Mobile World Capital Barcelona): frame-level ball possession for FC Barcelona – Athletic Club.

For every one of the **199,319 frames** of the tactical video we output one label:

| label | meaning |
|---|---|
| `H` | Barcelona (home) in possession, ball alive |
| `A` | Athletic Club (away) in possession, ball alive |
| `D` | dead ball |

The official challenge, data description, metric and rules are in the organisers' repository:
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027). Our summaries are
in [`docs/challenge.md`](docs/challenge.md) and [`docs/rules.md`](docs/rules.md); where they disagree, the official
documents win.

## Deliverables

| deliverable | file |
|---|---|
| Possession file | [`submission/Tiki-Data.csv`](submission/Tiki-Data.csv) |
| Public code repository | this repository |
| Technical document | [`docs/Tiki-Data.md`](docs/Tiki-Data.md) (also [`docs/Tiki-Data.pdf`](docs/Tiki-Data.pdf)) |

## Reproduce the submission

Requirements: Python 3.14, CPU only (no GPU), ~16 GB of RAM, ~6 GB of free disk for the external matches and caches,
internet access on the first run (to download the external training matches).

```bash
# 1. this repository and the organisers' repository, side by side
git clone https://github.com/Sergirenauseguer/Tiki-Data-FC.git
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
cd Tiki-Data-FC

# 2. the match data bundle (not in this repository; licensed for the competition only)
#    unpack it so that data/tactical/ and data/broadcast/ exist, or set MTAH_DATA_DIR to its folder

# 3. dependencies
pip install -r requirements.txt

# 4. one command: downloads the external matches, trains, labels the match, writes submission/Tiki-Data.csv
python run.py
```

Environment variables (optional): `MTAH_DATA_DIR` (data bundle, default `./data`), `MTAH_CHALLENGE_DIR` (organisers'
repository, default `../more-than-a-hack-2027`), `MTAH_CACHE_DIR` (caches, default `./cache`).

`run.py` prints the SHA-256 of the file it writes. Expected:

```
SHA-256 fae48ed0ea0de4c55a5bf15666275f7855b7906cd0f465c28637cce2492aff7b
```

Run time: about 1 h 20 min on a laptop CPU (most of it building and training on the external matches).

## What the code does

| layer | what | main code |
|---|---|---|
| 1 | Ball physics and dead-ball rules from the tracking (ball out, restarts, kick-offs, goalkeeper, players stopping) | `src/physics_*.py`, `src/postprocess.py`, `src/pipeline_approach_v13/14/15_physics.py` |
| 2 | Frame model trained on external matches degraded to look like the provided tracking: P(dead), P(home), P(possession change) | `src/frame_features.py`, `src/degrade_v2.py`, `src/external_*.py`, `src/pipeline_approach_v17/19/20/21.py` |
| 3 | Choosing whole possessions among the candidates of 28 decodings (scorer + exact interval selection) | `src/spell_select.py`, `src/noche_v21.py`, `src/decoding_v21.py` |
| 4 | Rules of the game on the selected possessions | `src/pipeline_approach_v25.py`, `src/reglas_futbol.py`, `src/pipeline_approach_v30.py`, `tools/*.py` |

Details, data used, training data, rules and limitations: [`docs/Tiki-Data.md`](docs/Tiki-Data.md).

## Data

The match data is **not** in this repository: it is licensed for the competition only (see `.gitignore`). External
training data: the IDSSE / Sportec Open DFL dataset (CC BY 4.0, Bassek et al., 2025) and the Metrica Sports sample
data, downloaded by `tools/get_idsse.py` and `tools/get_metrica.py` from their public sources.

## Licence

MIT for the code and documentation in this repository ([LICENSE](LICENSE)); it does not cover the match data.
