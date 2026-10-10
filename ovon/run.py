"""Minimal Habitat entry point for DAgger and PPO training."""

import argparse
from pathlib import Path

import ovon  # noqa: F401 - register LANav components
from habitat import get_config
from habitat.config import read_write
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp

from ovon.config import LANavConfigPlugin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-type", choices=("train",), required=True)
    parser.add_argument("--exp-config", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def resolve_exp_config(config_path: str) -> str:
    """Resolve source-tree and installed-wheel config paths consistently."""

    requested = Path(config_path)
    if requested.exists():
        return str(requested)
    parts = requested.parts
    if parts and parts[0] == "config":
        requested = Path(*parts[1:])
    packaged = Path(__file__).resolve().parents[1] / "config" / requested
    return str(packaged) if packaged.exists() else config_path


def main() -> None:
    args = parse_args()
    register_hydra_plugin(LANavConfigPlugin)
    config = get_config(resolve_exp_config(args.exp_config), args.opts)

    with read_write(config):
        if args.run_type == "train" and hasattr(
            config.habitat.task.measurements, "frontier_exploration_map"
        ):
            # The map is needed by the oracle sensor while constructing the
            # config, but logging the visualization during training is wasteful.
            config.habitat.task.measurements.pop("frontier_exploration_map")
        for output in (
            config.habitat_baselines.checkpoint_folder,
            config.habitat_baselines.tensorboard_dir,
        ):
            Path(output).mkdir(parents=True, exist_ok=True)

    execute_exp(config, args.run_type)


if __name__ == "__main__":
    main()
