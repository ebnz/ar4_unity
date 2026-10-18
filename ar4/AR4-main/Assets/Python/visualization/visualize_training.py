#!/usr/bin/env python3
"""
Utility script to visualize training progress from TensorBoard logs
Can be used to export training curves as images or analyze metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
from datetime import datetime


def load_tensorboard_logs(log_dir):
    """Load data from TensorBoard logs"""
    event_files = glob.glob(os.path.join(log_dir, "**/*.tfevents.*"), recursive=True)

    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None

    # Load the most recent event file
    latest_file = max(event_files, key=os.path.getctime)
    print(f"Loading events from: {latest_file}")

    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()

    return ea


def plot_training_curves(log_dir, save_plots=True):
    """Plot training curves from TensorBoard logs"""
    ea = load_tensorboard_logs(log_dir)
    if ea is None:
        return

    # Get available scalar tags
    scalar_tags = ea.Tags()["scalars"]
    print(f"\nAvailable metrics: {scalar_tags}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PPO Training Progress", fontsize=16)

    # Metrics to plot
    metrics = [
        ("rollout/ep_rew_mean", "Episode Reward Mean", axes[0, 0]),
        ("rollout/ep_len_mean", "Episode Length Mean", axes[0, 1]),
        ("train/value_loss", "Value Loss", axes[1, 0]),
        ("train/policy_gradient_loss", "Policy Loss", axes[1, 1]),
    ]

    for metric_name, title, ax in metrics:
        if metric_name in scalar_tags:
            events = ea.Scalars(metric_name)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            ax.plot(steps, values, linewidth=2)
            ax.set_xlabel("Steps")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Add smoothed line
            if len(values) > 10:
                window_size = min(50, len(values) // 10)
                smoothed = np.convolve(
                    values, np.ones(window_size) / window_size, mode="valid"
                )
                smooth_steps = steps[
                    window_size
                    // 2 : -(window_size // 2) + 1 if window_size // 2 > 0 else None
                ]
                ax.plot(
                    smooth_steps,
                    smoothed,
                    "r-",
                    alpha=0.7,
                    linewidth=2,
                    label="Smoothed",
                )
                ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                f"{metric_name}\nnot found",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(title)

    plt.tight_layout()

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"training_curves_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to: {plot_path}")

    plt.show()


def print_latest_metrics(log_dir):
    """Print the latest training metrics"""
    ea = load_tensorboard_logs(log_dir)
    if ea is None:
        return

    scalar_tags = ea.Tags()["scalars"]

    print("\n" + "=" * 60)
    print("Latest Training Metrics")
    print("=" * 60)

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        if events:
            latest = events[-1]
            print(f"{tag}: {latest.value:.4f} (step {latest.step})")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize PPO training progress")
    parser.add_argument(
        "--log-dir",
        default="./ppo_robot_tensorboard/",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save plots to file"
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only print latest metrics without plotting",
    )

    args = parser.parse_args()

    if args.metrics_only:
        print_latest_metrics(args.log_dir)
    else:
        plot_training_curves(args.log_dir, save_plots=not args.no_save)
