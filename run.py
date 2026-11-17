"""Regenerate submission/Tiki-Data.csv from the data bundle, in one command.

    python run.py

Needs: the organisers' data bundle in ./data (or MTAH_DATA_DIR) and the organisers' repository
metrica-sports/more-than-a-hack-2027 cloned next to this one (or MTAH_CHALLENGE_DIR). External training matches are
downloaded on the first run. CPU only; deterministic.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "lgb" if "--model" in sys.argv and sys.argv[sys.argv.index("--model") + 1] == "lgb" else "seq"   # default: sequence model
FINAL = os.path.join(ROOT, "experiments", "pipeline_approach_v30", "ours_pipeline_approach_v30.csv") if MODEL == "lgb" else     os.path.join(ROOT, "experiments", "pipeline_approach_v29", "ours_pipeline_approach_v29.csv")
SUB = os.path.join(ROOT, "submission", "Tiki-Data.csv")

STEPS = [
    ("download external matches (IDSSE)", [sys.executable, "tools/get_idsse.py"]),
    ("download external matches (Metrica sample data)", [sys.executable, "tools/get_metrica.py"]),
    ("read the provided match", [sys.executable, "-m", "src.io"]),
    ("build the external training matches (IDSSE)", [sys.executable, "-m", "src.external_idsse"]),
    ("build the external training matches (Metrica)", [sys.executable, "-m", "src.external_metrica"]),
    ("layers 1-3 + first rule set: train on the external matches and label the match",
     [sys.executable, "src/pipeline_approach_v25.py", "--step", "final"]),
    ("layer 4: rules of the game", [sys.executable, "src/pipeline_approach_v30.py"]),
]
E21 = "experiments/pipeline_approach_v21/"
SEQ = {"MTAH_REF": "smooth", "SEQ_FILES": "seq_full_cpu.npz,seq_full_s1w600_cpu.npz", "SEQ_TAG": "_ens"}
STEPS_SEQ = STEPS[:5] + [
    ("layer 1: dead-ball rules on the match", [sys.executable, "-c",
     "import sys; sys.path[:0] = ['src', '.']; import pipeline_approach_v20 as V; V.setup(2); V.step_rules()"], {}),
    ("layer 2: frame models on the external matches (out of sample)", [sys.executable, "src/noche_v21.py", "--step", "ref"],
     {"MTAH_REF": "smooth"}),
    ("sequence model: signals of the match at 5 fps", [sys.executable, "tools/seq_export.py"], {}),
    ("sequence model 1 (1-min window): inference on CPU",
     [sys.executable, "tools/seq_infer_cpu.py", "models/seq/seq_w300_s0.pt", E21 + "seq_full.npz", "--escribir",
      E21 + "seq_full_cpu.npz"], {}),
    ("sequence model 2 (2-min window): inference on CPU",
     [sys.executable, "tools/seq_infer_cpu.py", "models/seq/seq_w600_s1.pt", E21 + "seq_full_s1w600.npz", "--escribir",
      E21 + "seq_full_s1w600_cpu.npz"], {}),
    ("layer 3 with the sequence model: candidate tables and scorer", [sys.executable, "tools/seq_state.py"], SEQ),
    ("layer 3 + first rule set: choose the possessions", [sys.executable, "tools/v26_seq.py"],
     dict(SEQ, SEQ_LAM="0.85", SEQ_NAME_SUF="_l085")),
    ("layer 4: rules of the game", [sys.executable, "src/pipeline_approach_v30.py"],
     {"RULES_IN": "experiments/pipeline_approach_v26_seq_ens_l085/ours_pipeline_approach_v26_seq_ens_l085_reglas.csv",
      "RULES_OUT": "experiments/pipeline_approach_v29/ours_pipeline_approach_v29.csv"}),
]


def main():
    env = dict(os.environ)
    env.setdefault("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
    env.setdefault("MTAH_CACHE_DIR", os.path.join(ROOT, "cache"))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.time()
    steps = STEPS if MODEL == "lgb" else STEPS_SEQ
    for i, st in enumerate(steps, 1):
        what, cmd = st[0], st[1]
        print("[{}/{}] {} ...".format(i, len(steps), what), flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=dict(env, **(st[2] if len(st) > 2 else {})))
        if r.returncode != 0:
            sys.exit("step failed: {}".format(" ".join(cmd)))
    os.makedirs(os.path.dirname(SUB), exist_ok=True)
    shutil.copyfile(FINAL, SUB)
    rows = sum(1 for _ in open(SUB)) - 1
    labels = set(line.rstrip("\n").split(",")[1] for line in list(open(SUB))[1:])
    assert rows == 199319 and labels <= {"H", "A", "D", "X"}, (rows, labels)
    print("-> {}  ({} rows, {:.0f} min)".format(SUB, rows, (time.time() - t0) / 60))
    print("SHA-256 {}".format(hashlib.sha256(open(SUB, "rb").read()).hexdigest()))


if __name__ == "__main__":
    main()
