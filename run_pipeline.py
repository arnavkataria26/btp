"""End-to-end pipeline: raw CSVs -> windowed features -> 3 models x 2 modalities -> CV report.

    .venv/bin/python run_pipeline.py                     # everything (feature build is cached/resumable)
    .venv/bin/python run_pipeline.py --stage features --limit 6   # quick smoke test on 6 folders
    .venv/bin/python run_pipeline.py --stage train --group-level condition   # stricter generalisation test
    .venv/bin/python run_pipeline.py --stage speed       # unseen-speed test: train on 2 Motor-2 speeds, test on the 3rd
    .venv/bin/python run_pipeline.py --stage report      # rebuild report + 90/95/99 % confidence, no retraining
"""
import argparse
import logging

from pump_pipeline import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "features", "train", "speed", "report"], default="all")
    ap.add_argument("--workers", type=int, default=4, help="parallel condition folders during feature build")
    ap.add_argument("--limit", type=int, default=None, help="only process N condition folders (smoke test)")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached feature shards")
    ap.add_argument("--models", default="hrf,xgb,mlp", help="comma list of hrf,xgb,mlp")
    ap.add_argument("--group-level", choices=["recording", "condition"], default="recording",
                    help="CV grouping unit: keep whole recordings (default) or whole condition folders out")
    ap.add_argument("--folds", type=int, default=config.N_SPLITS)
    args = ap.parse_args()

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(config.REPORT_DIR / "pipeline.log")])
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.stage in ("all", "features"):
        from pump_pipeline.build_features import build
        build(workers=args.workers, limit=args.limit, rebuild=args.rebuild)
    if args.stage in ("all", "train"):
        from pump_pipeline.evaluate import run
        run(models=tuple(args.models.split(",")), group_level=args.group_level, n_splits=args.folds)
    if args.stage in ("all", "speed"):
        from pump_pipeline.cross_speed import run as run_speed
        run_speed(models=tuple(args.models.split(",")))
    if args.stage in ("all", "speed", "report"):
        from pump_pipeline.report import write_all
        write_all()


if __name__ == "__main__":
    main()
