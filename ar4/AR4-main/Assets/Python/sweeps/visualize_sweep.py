"""
Visualize sweep results with heatmaps and other plots.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from training.common import load_results_from_json


def load_sweep_results(sweep_dir: str) -> Dict[str, Any]:
    """Load all results from a sweep directory."""
    results = []

    # Find all run directories
    for entry in os.listdir(sweep_dir):
        run_dir = os.path.join(sweep_dir, entry)
        if not os.path.isdir(run_dir) or not entry.startswith("run_"):
            continue

        # Load run config
        config_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(config_path):
            continue

        with open(config_path, "r") as f:
            run_config = json.load(f)

        # Load results
        result_path = os.path.join(run_dir, "results", "results.json")
        result_data = load_results_from_json(result_path)

        if result_data:
            results.append(
                {
                    "run_name": run_config.get("run_name", entry),
                    "params": run_config.get("params", {}),
                    "seed": run_config.get("seed", 0),
                    "success_rate": result_data.get("success_rate", 0.0),
                    "mean_reward": result_data.get("episode_reward_stats", {}).get(
                        "mean", 0.0
                    ),
                }
            )

    return {"results": results}


def regenerate_summary(sweep_dir: str) -> Dict[str, Any]:
    """Regenerate summary.json from individual run results."""
    data = load_sweep_results(sweep_dir)
    results = data["results"]

    # Load sweep config for metadata
    sweep_config_path = os.path.join(sweep_dir, "sweep_config.json")
    sweep_config = {}
    if os.path.exists(sweep_config_path):
        with open(sweep_config_path, "r") as f:
            sweep_config = json.load(f)

    summary = {
        "sweep_name": sweep_config.get("name", "unknown"),
        "description": sweep_config.get("description", ""),
        "total_runs": len(results),
        "completed_runs": len(results),
        "failed_runs": 0,
        "results": results,
    }

    # Aggregate stats per parameter combination
    param_stats = {}
    for r in results:
        key = json.dumps(r["params"], sort_keys=True)
        if key not in param_stats:
            param_stats[key] = {
                "params": r["params"],
                "success_rates": [],
                "rewards": [],
            }
        param_stats[key]["success_rates"].append(r["success_rate"])
        param_stats[key]["rewards"].append(r["mean_reward"])

    summary["aggregated"] = []
    for key, stats in param_stats.items():
        summary["aggregated"].append(
            {
                "params": stats["params"],
                "mean_success_rate": float(np.mean(stats["success_rates"])),
                "std_success_rate": float(np.std(stats["success_rates"])),
                "mean_reward": float(np.mean(stats["rewards"])),
                "std_reward": float(np.std(stats["rewards"])),
                "n_runs": len(stats["success_rates"]),
            }
        )

    # Sort by success rate
    summary["aggregated"].sort(key=lambda x: x["mean_success_rate"], reverse=True)

    # Save updated summary
    summary_path = os.path.join(sweep_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary regenerated: {summary_path}")
    return summary


def create_heatmap(
    sweep_dir: str,
    param_x: str,
    param_y: str,
    metric: str = "success_rate",
    output_path: Optional[str] = None,
):
    """
    Create a heatmap of sweep results.

    Args:
        sweep_dir: Path to sweep directory
        param_x: Parameter for x-axis
        param_y: Parameter for y-axis
        metric: 'success_rate' or 'mean_reward'
        output_path: Where to save the plot (optional)
    """
    # Load or regenerate summary
    summary_path = os.path.join(sweep_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        # Check if aggregated data exists and has valid values
        if not summary.get("aggregated") or all(
            a.get("mean_success_rate", 0) == 0 for a in summary["aggregated"]
        ):
            print("Summary has no valid data, regenerating...")
            summary = regenerate_summary(sweep_dir)
    else:
        summary = regenerate_summary(sweep_dir)

    aggregated = summary.get("aggregated", [])
    if not aggregated:
        print("No aggregated data found!")
        return

    # Extract unique values for each parameter
    x_values = sorted(
        set(a["params"].get(param_x) for a in aggregated if param_x in a["params"])
    )
    y_values = sorted(
        set(a["params"].get(param_y) for a in aggregated if param_y in a["params"])
    )

    if not x_values or not y_values:
        print(f"Could not find parameters '{param_x}' and '{param_y}' in results")
        return

    # Build heatmap matrix
    matrix = np.zeros((len(y_values), len(x_values)))

    for a in aggregated:
        params = a["params"]
        if param_x in params and param_y in params:
            x_idx = x_values.index(params[param_x])
            y_idx = y_values.index(params[param_y])

            if metric == "success_rate":
                matrix[y_idx, x_idx] = (
                    a["mean_success_rate"] * 100
                )  # Convert to percentage
            else:
                matrix[y_idx, x_idx] = a["mean_reward"]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    if metric == "success_rate":
        cmap = "RdYlGn"  # Red to Yellow to Green for success rate
        fmt = ".1f"
        cbar_label = "Success Rate (%)"
        vmin, vmax = 0, 100
    else:
        cmap = "viridis"
        fmt = ".1f"
        cbar_label = "Mean Reward"
        vmin, vmax = None, None

    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=[f"{v}" for v in x_values],
        yticklabels=[f"{v}" for v in y_values],
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": cbar_label},
    )

    ax.set_xlabel(param_x.replace("_", " ").title())
    ax.set_ylabel(param_y.replace("_", " ").title())
    ax.set_title(
        f"{summary.get('sweep_name', 'Sweep')} - {metric.replace('_', ' ').title()}"
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_curves(
    sweep_dir: str,
    metric: str = "success_rate",
    group_by: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    Plot learning curves for all runs showing metric over training steps.

    Args:
        sweep_dir: Path to sweep directory
        metric: 'success_rate' or 'mean_reward'
        group_by: Parameter to group/color by (e.g., 'distance_scale'). If None, each run gets its own line.
        output_path: Where to save the plot (optional)
    """
    # Collect all learning curves
    curves = []

    for entry in os.listdir(sweep_dir):
        run_dir = os.path.join(sweep_dir, entry)
        if not os.path.isdir(run_dir) or not entry.startswith("run_"):
            continue

        # Load run config
        config_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(config_path):
            continue

        with open(config_path, "r") as f:
            run_config = json.load(f)

        # Load full results with all checkpoints
        result_path = os.path.join(run_dir, "results", "results.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path, "r") as f:
            result_data = json.load(f)

        # Extract checkpoints (Unity uses "results" key)
        checkpoints = result_data.get("results", [])
        if not checkpoints:
            continue

        steps = []
        values = []
        for cp in checkpoints:
            step = cp.get("step_number", 0)
            if metric == "success_rate":
                value = cp.get("success_rate", 0.0) * 100  # Convert to percentage
            else:
                value = cp.get("episode_reward_stats", {}).get("mean", 0.0)
            steps.append(step)
            values.append(value)

        curves.append({
            "run_name": run_config.get("run_name", entry),
            "params": run_config.get("params", {}),
            "seed": run_config.get("seed", 0),
            "steps": steps,
            "values": values,
        })

    if not curves:
        print("No learning curve data found!")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette
    if group_by:
        # Group by parameter value
        unique_values = sorted(set(c["params"].get(group_by) for c in curves if group_by in c["params"]))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_values)))
        color_map = {v: colors[i] for i, v in enumerate(unique_values)}

        for curve in curves:
            param_val = curve["params"].get(group_by)
            if param_val is not None:
                color = color_map[param_val]
                label = f"{group_by}={param_val}" if curve["seed"] == 0 else None
                alpha = 0.7
                ax.plot(curve["steps"], curve["values"], color=color, alpha=alpha, label=label)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title=group_by.replace("_", " ").title())
    else:
        # Each run gets its own color (can be busy with many runs)
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(curves))))
        for i, curve in enumerate(curves):
            color = colors[i % len(colors)]
            ax.plot(curve["steps"], curve["values"], color=color, alpha=0.6)

    ax.set_xlabel("Training Steps")
    if metric == "success_rate":
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("Mean Reward")

    ax.set_title(f"Learning Curves - {metric.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_aggregated_learning_curves(
    sweep_dir: str,
    group_by: str = "distance_scale",
    metric: str = "success_rate",
    output_path: Optional[str] = None,
):
    """
    Plot aggregated learning curves with mean and std bands, grouped by parameter.

    Args:
        sweep_dir: Path to sweep directory
        group_by: Parameter to group by
        metric: 'success_rate' or 'mean_reward'
        output_path: Where to save the plot (optional)
    """
    # Collect all learning curves grouped by parameter value
    grouped_curves = {}

    for entry in os.listdir(sweep_dir):
        run_dir = os.path.join(sweep_dir, entry)
        if not os.path.isdir(run_dir) or not entry.startswith("run_"):
            continue

        config_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(config_path):
            continue

        with open(config_path, "r") as f:
            run_config = json.load(f)

        result_path = os.path.join(run_dir, "results", "results.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path, "r") as f:
            result_data = json.load(f)

        checkpoints = result_data.get("results", [])
        if not checkpoints:
            continue

        # Get group key
        group_val = run_config.get("params", {}).get(group_by)
        if group_val is None:
            continue

        if group_val not in grouped_curves:
            grouped_curves[group_val] = []

        steps = []
        values = []
        for cp in checkpoints:
            step = cp.get("step_number", 0)
            if metric == "success_rate":
                value = cp.get("success_rate", 0.0) * 100
            else:
                value = cp.get("episode_reward_stats", {}).get("mean", 0.0)
            steps.append(step)
            values.append(value)

        grouped_curves[group_val].append({"steps": steps, "values": values})

    if not grouped_curves:
        print("No learning curve data found!")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_curves)))

    for i, (group_val, curves) in enumerate(sorted(grouped_curves.items())):
        # Find common steps across all curves in this group
        all_steps = set()
        for c in curves:
            all_steps.update(c["steps"])
        common_steps = sorted(all_steps)

        # Interpolate values at common steps
        all_values = []
        for c in curves:
            interp_values = np.interp(common_steps, c["steps"], c["values"])
            all_values.append(interp_values)

        all_values = np.array(all_values)
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        color = colors[i]
        ax.plot(common_steps, mean_values, color=color, label=f"{group_by}={group_val}", linewidth=2)
        ax.fill_between(common_steps, mean_values - std_values, mean_values + std_values,
                        color=color, alpha=0.2)

    ax.set_xlabel("Training Steps")
    if metric == "success_rate":
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("Mean Reward")

    ax.set_title(f"Aggregated Learning Curves by {group_by.replace('_', ' ').title()}")
    ax.legend(title=group_by.replace("_", " ").title())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Aggregated learning curves saved to: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sweep results")
    parser.add_argument("sweep_dir", type=str, help="Path to sweep directory")
    parser.add_argument(
        "--plot",
        type=str,
        default="heatmap",
        choices=["heatmap", "curves", "aggregated"],
        help="Type of plot: heatmap, curves (all runs), aggregated (mean+std)",
    )
    parser.add_argument(
        "--param-x", type=str, default="distance_scale", help="Parameter for x-axis (heatmap)"
    )
    parser.add_argument(
        "--param-y", type=str, default="time_penalty", help="Parameter for y-axis (heatmap)"
    )
    parser.add_argument(
        "--group-by", type=str, default="distance_scale", help="Parameter to group by (curves)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="success_rate",
        choices=["success_rate", "mean_reward"],
        help="Metric to plot",
    )
    parser.add_argument("--output", type=str, help="Output path for plot")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate summary from individual results",
    )

    args = parser.parse_args()

    if args.regenerate:
        regenerate_summary(args.sweep_dir)

    if args.plot == "heatmap":
        create_heatmap(args.sweep_dir, args.param_x, args.param_y, args.metric, args.output)
    elif args.plot == "curves":
        plot_learning_curves(args.sweep_dir, args.metric, args.group_by, args.output)
    elif args.plot == "aggregated":
        plot_aggregated_learning_curves(args.sweep_dir, args.group_by, args.metric, args.output)
