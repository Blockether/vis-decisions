"""Prepare a verified local FP32 decision bundle from an upstream D3 checkpoint."""

import argparse
import json

from blockether.vis_decisions._d3 import prepare_fp32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Complete local upstream checkpoint"
    )
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--revision", required=True, help="Pinned source revision")
    arguments = parser.parse_args()
    report = prepare_fp32(
        arguments.checkpoint, arguments.output, revision=arguments.revision
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
