"""Generate the journal-v2 campaign configs from the pilot selection.

Reads the pilot gate JSON, picks the winning arm/loss, and writes the
campaign, PAR, ablation, and v1-fixed configs. All K=100 extra-test paths are
absolute so no additional path resolution is needed at run time.
"""

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _dump(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _with_extra(cfg, queries, paths, max_hops):
    cfg["extra_tests"] = [{
        "name": "k100",
        "queries_dir": str(queries),
        "paths_dir": str(paths),
        "max_hops": max_hops,
    }]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "pamr" / "configs" / "pr"))
    args = ap.parse_args()

    with open(args.gate) as f:
        gate = json.load(f)
    selected = gate.get("selected") or {}
    label = selected.get("label")
    if not label:
        raise SystemExit("pilot gate has no selected run")
    arm, loss = label.split("_")
    base_name = ("pamr_v2_listwise" if loss == "lf" else "pamr_v2_pairwise")
    base = _load(REPO_ROOT / "pamr" / "configs" / "pr" / f"{base_name}.yaml")
    v2 = Path(args.v2_root)
    k100 = v2 / "k100"
    arm_root = v2 / arm
    arm_h3 = v2 / f"{arm}_h3"
    out = Path(args.out_dir)

    campaign = dict(base)
    campaign["paths_dir"] = str(arm_root / "paths")
    campaign["queries_dir"] = str(arm_root / "queries")
    campaign["min_residual_std"] = 0.01
    campaign["save_checkpoint"] = False
    _with_extra(campaign, k100 / "queries", k100 / "paths_prod", 5)
    _dump(campaign, out / "pamr_v2_campaign.yaml")

    par = dict(campaign)
    par["vlkge_config"] = "../wikiart_mkg_v2/transe_struct.yaml"
    par["vlkge_checkpoint"] = ("../checkpoints/vlkge/wikiart_mkg_v2/"
                               "transe_wikiart_mkg_v2_struct_train.pt")
    _dump(par, out / "pamr_v2_campaign_par.yaml")

    finetune = dict(campaign)
    finetune["fine_tune_backbone"] = True
    _dump(finetune, out / "pamr_v2_campaign_finetune.yaml")

    depth12 = dict(campaign)
    depth12["num_layers"] = 12
    _dump(depth12, out / "pamr_v2_campaign_depth12.yaml")

    cls = dict(campaign)
    cls["readout"] = "cls"
    _dump(cls, out / "pamr_v2_campaign_cls.yaml")

    lf_temp = dict(campaign)
    lf_temp["loss_mode"] = "listwise_softmax"
    lf_temp["lf_temperature"] = 0.1
    _dump(lf_temp, out / "pamr_v2_campaign_lf_temp.yaml")

    no_mask = dict(campaign)
    no_mask["loss_mask"] = False
    no_mask["min_residual_std"] = None
    _dump(no_mask, out / "pamr_v2_campaign_no_mask.yaml")

    hop3 = dict(campaign)
    hop3["paths_dir"] = str(arm_h3 / "paths")
    hop3["max_hops"] = 3
    hop3["max_tokens"] = 9
    _with_extra(hop3, k100 / "queries", v2 / "k100_h3" / "paths", 3)
    _dump(hop3, out / "pamr_v2_campaign_hop3.yaml")

    v1 = dict(campaign)
    v1["paths_dir"] = str(Path(args.v1_root) / "full" / "pamr")
    v1["queries_dir"] = str(args.v1_root)
    v1["loss_mode"] = "grouped_pairwise"
    v1["min_residual_std"] = None
    _with_extra(v1, k100 / "queries", k100 / "paths_prod", 5)
    _dump(v1, out / "pamr_v1fixed.yaml")

    manifest = {
        "selected": label,
        "arm": arm,
        "loss": loss,
        "configs": [f.name for f in sorted(out.glob("pamr_v2_campaign*.yaml"))]
        + ["pamr_v1fixed.yaml"],
    }
    print("ACCEPTANCE campaign-configs " + json.dumps(manifest))


if __name__ == "__main__":
    main()
