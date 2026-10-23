"""Gather Boltz-2 affinity/confidence JSONs into one table next to the DAVIS labels.

Boltz-2 affinity fields (see boltz docs/prediction.md):
  affinity_pred_value          log10(IC50 / uM); only meaningful to rank actives
  affinity_probability_binary  P(binder) for binder-vs-decoy decisions
We report pIC50_pred = 6 - affinity_pred_value to compare with DAVIS pKd.
"""

import json
import sys
from pathlib import Path

import pandas as pd

out_dir = Path(sys.argv[1])
labels = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
rows = []
for aff_path in sorted(out_dir.glob("boltz_results_*/predictions/*/affinity_*.json")):
    name = aff_path.parent.name
    aff = json.loads(aff_path.read_text())
    conf_path = next(aff_path.parent.glob(f"confidence_{name}_model_0.json"), None)
    conf = json.loads(conf_path.read_text()) if conf_path else {}
    rows.append({
        "name": name,
        "affinity_pred_value": aff["affinity_pred_value"],
        "pIC50_pred": 6 - aff["affinity_pred_value"],
        "affinity_probability_binary": aff["affinity_probability_binary"],
        "confidence_score": conf.get("confidence_score"),
        "ptm": conf.get("ptm"),
        "iptm": conf.get("iptm"),
        "ligand_iptm": conf.get("ligand_iptm"),
        "complex_plddt": conf.get("complex_plddt"),
    })
df = pd.DataFrame(rows)
if labels is not None and not df.empty:
    df = labels.merge(df, on="name", how="right")
df.to_csv(out_dir / "boltz2_summary.csv", index=False)
print(df.to_string(index=False))
