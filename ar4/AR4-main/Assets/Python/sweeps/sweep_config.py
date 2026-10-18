"""
Sweep configuration and parameter grid definitions.
"""

import itertools
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime


@dataclass
class ParameterGrid:
    """
    Defines a grid of parameters to sweep over.

    Example:
        grid = ParameterGrid()
        grid.add("distance_scale", [0.5, 1.0, 2.0, 5.0])
        grid.add("time_penalty", [0.001, 0.01])

        for params in grid:
            print(params)  # {'distance_scale': 0.5, 'time_penalty': 0.001}, ...
    """

    parameters: Dict[str, List[Any]] = field(default_factory=dict)

    def add(self, name: str, values: List[Any]) -> "ParameterGrid":
        """Add a parameter with its possible values."""
        self.parameters[name] = values
        return self

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all combinations of parameters."""
        if not self.parameters:
            yield {}
            return

        keys = list(self.parameters.keys())
        values = [self.parameters[k] for k in keys]

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def __len__(self) -> int:
        """Total number of parameter combinations."""
        if not self.parameters:
            return 1

        total = 1
        for values in self.parameters.values():
            total *= len(values)
        return total

    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert to dictionary."""
        return dict(self.parameters)

    @classmethod
    def from_dict(cls, data: Dict[str, List[Any]]) -> "ParameterGrid":
        """Create from dictionary."""
        grid = cls()
        grid.parameters = dict(data)
        return grid


@dataclass
class SweepConfig:
    """
    Configuration for a parameter sweep experiment.

    Attributes:
        name: Human-readable name for this sweep
        description: Optional description of what this sweep tests
        template_path: Path to the reward template file
        base_training_config: Path to the base training config YAML
        parameter_grid: Grid of parameters to sweep
        max_training_steps: Override for training steps (optional)
        parallel_runs: Number of parallel training runs
        seeds: List of seeds to run for each parameter combo (for statistical significance)
    """

    name: str
    template_path: str
    base_training_config: str
    parameter_grid: ParameterGrid
    description: str = ""
    max_training_steps: Optional[int] = None
    parallel_runs: int = 4
    seeds: List[int] = field(default_factory=lambda: [0])
    output_dir: Optional[str] = None

    def __post_init__(self):
        """Generate output directory if not specified."""
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize name for directory
            safe_name = self.name.replace(" ", "_").replace("/", "_")
            self.output_dir = f"sweep_{timestamp}_{safe_name}"

    def get_total_runs(self) -> int:
        """Total number of training runs in this sweep."""
        return len(self.parameter_grid) * len(self.seeds)

    def get_run_name(self, params: Dict[str, Any], seed: int) -> str:
        """Generate a unique run name from parameters and seed."""
        parts = []
        for key, value in sorted(params.items()):
            # Format value nicely
            if isinstance(value, float):
                parts.append(f"{key}_{value:.4g}")
            else:
                parts.append(f"{key}_{value}")

        if len(self.seeds) > 1:
            parts.append(f"seed_{seed}")

        return "_".join(parts) if parts else "default"

    def save(self, path: str):
        """Save config to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "template_path": self.template_path,
            "base_training_config": self.base_training_config,
            "parameter_grid": self.parameter_grid.to_dict(),
            "max_training_steps": self.max_training_steps,
            "parallel_runs": self.parallel_runs,
            "seeds": self.seeds,
            "output_dir": self.output_dir,
            "total_runs": self.get_total_runs(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SweepConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            template_path=data["template_path"],
            base_training_config=data["base_training_config"],
            parameter_grid=ParameterGrid.from_dict(data["parameter_grid"]),
            max_training_steps=data.get("max_training_steps"),
            parallel_runs=data.get("parallel_runs", 4),
            seeds=data.get("seeds", [0]),
            output_dir=data.get("output_dir"),
        )


def create_example_config() -> SweepConfig:
    """Create an example sweep config for T0 distance scaling."""
    grid = ParameterGrid()
    grid.add("distance_scale", [0.5, 1.0, 2.0, 5.0])
    grid.add("time_penalty", [0.001, 0.005, 0.01])

    return SweepConfig(
        name="T0_distance_scaling",
        description="Sweep over distance reward scaling and time penalty for T0 task",
        template_path="Scripts/rewards/reward_T0_template.txt",
        base_training_config="Python/config/training_config_T0.yaml",
        parameter_grid=grid,
        max_training_steps=100000,  # Shorter runs for sweep
        parallel_runs=4,
        seeds=[0, 1, 2],  # 3 seeds for statistical significance
    )


if __name__ == "__main__":
    # Demo: create and display example config
    config = create_example_config()

    print(f"Sweep: {config.name}")
    print(f"Description: {config.description}")
    print(f"Total runs: {config.get_total_runs()}")
    print(f"\nParameter combinations:")

    for i, params in enumerate(config.parameter_grid):
        for seed in config.seeds:
            run_name = config.get_run_name(params, seed)
            print(f"  {i+1}. {run_name}")
