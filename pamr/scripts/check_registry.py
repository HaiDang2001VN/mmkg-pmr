"""Registry acceptance checks for the canonical relation vocabulary.

Verifies, against the live dataset CSV and a VLKGE-style config:
  1. the raw relation count matches the expected vocabulary size;
  2. the total registry size (raw + internal inverses) matches the expected
     total;
  3. the inverse map is a perfect involution (inverse_id(inverse_id(x)) == x
     for every relation ID);
  4. the registry hash recomputed from the manifest matches the persisted
     registry hash;
  5. the public vocabulary contains exactly the expected number of named
     relations (internal inverse IDs have no public name).

Usage:
    python -m pamr.scripts.check_registry \
        --config configs/wikiart_mkg_v2/complex_clip.yaml \
        --expect-raw 22 --expect-total 40
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PKG_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PKG_ROOT.parent

from pamr import utils
from pamr.dataloader import KnowledgeGraphDataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Registry acceptance checks (40-ID canonical registry)"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a VLKGE-style YAML config defining the "
                             "dataset, exclusions, and inverse pairs")
    parser.add_argument("--expect-raw", type=int, default=None,
                        help="Expected number of raw (public) relations")
    parser.add_argument("--expect-total", type=int, default=None,
                        help="Expected total relation count (raw + internal)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        for candidate in [Path.cwd() / cfg_path, REPO_ROOT / cfg_path]:
            if candidate.exists():
                cfg_path = candidate.resolve()
                break
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    cfg = utils.load_yaml_config(cfg_path)
    cfg = utils.resolve_config_paths(cfg, base_dir=cfg_path.parent,
                                     repo_root=REPO_ROOT)

    dl = KnowledgeGraphDataLoader(
        data_path=cfg["data_path"],
        dataset_name=cfg["dataset"],
        exclude_relations=cfg.get("exclude_relations"),
        exclude_relations_eval=cfg.get("exclude_relations_eval"),
        inverse_relation_pairs=cfg.get("inverse_relation_pairs"),
        symmetric_relations=cfg.get("symmetric_relations"),
        use_per_relation_candidates=cfg.get("use_per_relation_candidates", True),
        artist2artist_relations=cfg.get("artist2artist_relations"),
        bidirectional_eval=False,
    )
    registry = dl.get_relation_registry()

    failures = []

    def check(label, condition, detail):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}: {detail}")
        if not condition:
            failures.append(label)

    n_raw = len(registry.raw_relation_names)
    check("raw relation count", n_raw == len(registry.relation_to_id),
          f"{n_raw} raw names")
    if args.expect_raw is not None:
        check("expected raw count", n_raw == args.expect_raw,
              f"{n_raw} vs {args.expect_raw}")
    if args.expect_total is not None:
        check("expected total count",
              registry.num_relations == args.expect_total,
              f"{registry.num_relations} vs {args.expect_total}")

    involution_ok = all(
        registry.inverse_id(registry.inverse_id(rel_id)) == rel_id
        for rel_id in range(registry.num_relations)
    )
    check("inverse involution", involution_ok,
          "inverse_id(inverse_id(x)) == x for all x")

    import hashlib
    import json
    encoded = json.dumps(registry.manifest, sort_keys=True,
                         separators=(",", ":"))
    recomputed = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    check("registry hash stable", recomputed == registry.registry_hash,
          f"{recomputed[:16]}…")

    n_public_named = sum(
        1 for rel_id in range(registry.num_relations)
        if registry.public_relation_name(rel_id) is not None
    )
    check("public vocabulary integrity",
          n_public_named == n_raw,
          f"{n_public_named} named of {registry.num_relations} total")

    manifest = registry.to_dict()
    check("manifest carries hash",
          manifest.get("registry_hash") == registry.registry_hash,
          str(manifest.get("registry_hash", ""))[:16] + "…")

    if failures:
        print(f"\nREGISTRY CHECKS FAILED: {failures}")
        sys.exit(1)
    print("\nREGISTRY CHECKS PASSED")


if __name__ == "__main__":
    main()
