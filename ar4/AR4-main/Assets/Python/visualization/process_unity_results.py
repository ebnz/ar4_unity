"""
Standalone script to process existing Unity training results and write them to TensorBoard

This is useful for retroactively analyzing training runs that have already completed.
"""

import os
import sys
import argparse
from training.common import write_unity_results_to_tensorboard


def main():
    parser = argparse.ArgumentParser(
        description="Process Unity training results and write to TensorBoard"
    )
    parser.add_argument(
        "result_file",
        type=str,
        help="Path to Unity results JSON file",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: same directory as result file)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="unity_stats",
        help="Name for this TensorBoard run (default: unity_stats)",
    )

    args = parser.parse_args()

    # Validate result file exists
    if not os.path.exists(args.result_file):
        print(f"ERROR: Result file not found: {args.result_file}")
        return 1

    # Determine TensorBoard directory
    if args.tensorboard_dir:
        tensorboard_dir = args.tensorboard_dir
    else:
        # Use parent directory of result file
        tensorboard_dir = os.path.dirname(os.path.dirname(args.result_file))

    os.makedirs(tensorboard_dir, exist_ok=True)

    print("=" * 60)
    print("Unity Results to TensorBoard Converter")
    print("=" * 60)
    print(f"Result file: {args.result_file}")
    print(f"TensorBoard directory: {tensorboard_dir}")
    print(f"Run name: {args.run_name}")
    print("-" * 60)

    # Process the results
    try:
        write_unity_results_to_tensorboard(
            result_path=args.result_file,
            tensorboard_log_dir=tensorboard_dir,
            run_name=args.run_name,
        )

        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"\nTo view in TensorBoard, run:")
        print(f'  tensorboard --logdir="{tensorboard_dir}"')
        print(f"\nThen open: http://localhost:6006")

        return 0

    except Exception as e:
        print(f"\nError processing results: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
