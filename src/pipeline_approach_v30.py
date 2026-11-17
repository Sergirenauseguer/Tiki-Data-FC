"""Layer 4 on top of the selected possessions: rules of the game.

Input: the selection with the first rule set (experiments/pipeline_approach_v25/ours_pipeline_approach_v25_reglas.csv,
written by `pipeline_approach_v25.py --step final`). Output: experiments/pipeline_approach_v30/ours_pipeline_approach_v30.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_approach_v25 as V  # noqa: E402
from src import reglas_futbol as RF  # noqa: E402
from src.io import load_atd  # noqa: E402
from pipeline_approach_v21 import V17  # noqa: E402
from tools.false_restart import false_restarts  # noqa: E402
from tools.restart_wait import apply as restart_wait  # noqa: E402
from src.pipeline_approach_v26 import short_dead_support  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.environ.get("RULES_IN") or os.path.join(ROOT, "experiments", "pipeline_approach_v25", "ours_pipeline_approach_v25_reglas.csv")
OUT = os.environ.get("RULES_OUT") or os.path.join(ROOT, "experiments", "pipeline_approach_v30", "ours_pipeline_approach_v30.csv")


def main():
    V.init()
    cx = V.ctx("barca")
    M = RF.Match()
    atd = load_atd()
    lab = pd.read_csv(IN, keep_default_na=False).label.values.astype("<U1")
    # set pieces, false restart, live picture at stoppage edges
    lab = RF.RULES["set"](lab, M)
    lab = V.RULES["falserestart"](lab, cx, v_still=0.5, still_s=1.0, look_s=6.0)
    lab = RF.RULES["replay_play"](lab, M)
    lab = V17.apply_definitions(lab, atd)
    # false restart (ball speed), short unexplained stoppage, restart wait
    lab, _, _ = false_restarts(lab, atd["ball"])
    lab, _ = short_dead_support(lab, atd["ball"])
    lab, _, _ = restart_wait(lab, atd)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(lab)), "label": lab}).to_csv(OUT, index=False, lineterminator="\n")
    print("->", OUT, flush=True)


if __name__ == "__main__":
    main()
