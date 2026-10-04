"""Prepare a verified local FP32 decision bundle from a Decision 2.0 checkpoint."""

import argparse
import json

from blockether.vis_decisions._decision2 import prepare_fp32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Complete local upstream checkpoint"
    )
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument(
        "--license", required=True, help="Local Apache-2.0 license text"
    )
    parser.add_argument("--revision", required=True, help="Pinned source revision")
    arguments = parser.parse_args()
    report = prepare_fp32(
        arguments.checkpoint,
        arguments.output,
        license_file=arguments.license,
        revision=arguments.revision,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
