from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.decision_fusion import run_decision_fusion, save_decision_fusion_outputs
from formal_rgbnir.iddaw import experiment_name, experiment_project_dir


if __name__ == "__main__":
    payload = run_decision_fusion(split="val", device="0")
    output_dir = experiment_project_dir() / experiment_name("decision_fusion")
    save_decision_fusion_outputs(output_dir, payload)
    print(payload["metrics"])
