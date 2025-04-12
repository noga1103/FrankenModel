import torch
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_experiment_tracking(experiment_name: str, output_dir: str) -> Path:
    """Set up experiment tracking and create necessary directories."""
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate the model configuration parameters."""
    required_fields = ["base_model_name", "layers_to_duplicate", "num_duplications"]
    return all(field in config for field in required_fields)


def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate and return model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def save_experiment_results(
    results: Dict[str, Any], output_path: Path, experiment_name: str
):
    """Save experiment results to disk."""
    import json

    results_file = output_path / f"{experiment_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {results_file}")


def load_experiment_results(output_path: Path, experiment_name: str) -> Dict[str, Any]:
    """Load experiment results from disk."""
    import json

    results_file = output_path / f"{experiment_name}_results.json"
    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Results file not found: {results_file}")
        return {}
