# Tiki-Data — More than a Hack 2027

Team **Tiki-Data**'s submission repository for **More than a Hack 2027**, Phase 1 (Metrica Sports, FC Barcelona
Innovation Hub, Mobile World Capital Barcelona): frame-level ball possession for FC Barcelona – Athletic Club.

For every one of the **199,319 frames** of the tactical video we output one label:

| label | meaning |
|---|---|
| `H` | Barcelona (home) in possession, ball alive |
| `A` | Athletic Club (away) in possession, ball alive |
| `D` | dead ball |
| `X` | no answer (ungraded frames) |

The official challenge, data description, metric and rules are in the organisers' repository:
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027). Our summaries are
in [`docs/`](docs/); where they disagree, the official documents win.

## Deliverables

| deliverable | file | status |
|---|---|---|
| Possession file | `submission/Tiki-Data.csv` | to be added |
| Public code repository | this repository | in progress |
| Technical document | `docs/Tiki-Data.pdf` (or `.md`) | to be added |

## Repository layout (planned)

```
Tiki-Data-FC/
├── README.md              this file
├── LICENSE                MIT, code and documentation only (not the match data)
├── docs/
│   ├── challenge.md       the task, the metric and the deliverables, in short
│   └── rules.md           the rules we follow and how this repository complies
├── src/                   pipeline source (to be added)
├── submission/            the generated possession file (to be added)
└── requirements.txt       pinned dependencies (to be added)
```

## Data

The match data is **not** in this repository and never will be: it is licensed for the competition only
(see `.gitignore`). To reproduce the submission, place the organisers' data bundle in `data/` (or point
`MTAH_DATA_DIR` to it). External datasets used for training are downloaded by scripts from their public sources.

## Reproduction

Instructions and the single command that regenerates `submission/Tiki-Data.csv` will be added here.

## Team

Tiki-Data.
