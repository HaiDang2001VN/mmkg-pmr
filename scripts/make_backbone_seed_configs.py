"""Write K10/K100 score + neg-eval configs for the extra backbone seeds."""

import argparse
from pathlib import Path

import yaml

MODELS = {
    "transe": ("transe_clip.yaml", "transe_wikiart_mkg_v2_train.pt", "TransE"),
    "complex": ("complex_clip.yaml", "complex_wikiart_mkg_v2_train.pt", "ComplEx"),
    "distmult": ("distmult_clip.yaml", "distmult_wikiart_mkg_v2_train.pt",
                 "DistMult"),
    "rotate": ("rotate_clip.yaml", "rotate_wikiart_mkg_v2_train.pt", "RotatE"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--out-dir", default="pamr/configs/eval")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for key, (mcfg, canon, label) in MODELS.items():
            ckpt = f"../checkpoints/vlkge/wikiart_mkg_v2/{canon[:-3]}_s{seed}.pt"
            # K10
            score = {
                "prepare_config": "configs/eval/wikiart_mkg_v2_neg_sampling.yaml",
                "model_config": f"configs/wikiart_mkg_v2/{mcfg}",
                "output_dir": (f"../checkpoints/preproc/wikiart_mkg_v2/full/"
                               f"seed_scores/{label}_s{seed}/"),
                "checkpoint_path": ckpt,
                "batch_size": 2048,
                "seed": 42,
            }
            (out / f"wikiart_mkg_v2_score_{key}_s{seed}.yaml").write_text(
                "# auto-generated seed config\n"
                + yaml.safe_dump(score, sort_keys=False))
            neval = {
                "prepare_config": "configs/eval/wikiart_mkg_v2_neg_sampling.yaml",
                "scores_dir": (f"../checkpoints/preproc/wikiart_mkg_v2/full/"
                               f"seed_scores/{label}_s{seed}/"),
                "top_k": [1, 3, 5, 10],
                "seed": 42,
            }
            (out / f"wikiart_mkg_v2_neg_eval_{key}_s{seed}.yaml").write_text(
                "# auto-generated seed config\n"
                + yaml.safe_dump(neval, sort_keys=False))
            # K100
            score100 = dict(score)
            score100["prepare_config"] = (
                "configs/eval/wikiart_mkg_v2_v2_k100.yaml")
            score100["output_dir"] = (
                f"../checkpoints/preproc/wikiart_mkg_v2_v2/k100/full/"
                f"seed_scores/{label}_s{seed}/")
            (out / f"wikiart_mkg_v2_v2_k100_score_{key}_s{seed}.yaml").write_text(
                "# auto-generated seed config\n"
                + yaml.safe_dump(score100, sort_keys=False))
            neval100 = dict(neval)
            neval100["prepare_config"] = (
                "configs/eval/wikiart_mkg_v2_v2_k100.yaml")
            neval100["scores_dir"] = score100["output_dir"]
            (out / f"wikiart_mkg_v2_v2_k100_neg_eval_{key}_s{seed}.yaml"
             ).write_text("# auto-generated seed config\n"
                          + yaml.safe_dump(neval100, sort_keys=False))
    print(f"ACCEPTANCE seed-configs seeds={args.seeds} models={len(MODELS)}")


if __name__ == "__main__":
    main()
