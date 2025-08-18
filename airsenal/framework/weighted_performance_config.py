"""
Configuration loader for weighted performance metrics.

This module provides utilities to load weight matrices from YAML configuration files,
allowing for easy tuning and experimentation with different weight combinations.
"""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from airsenal.framework.weighted_performance import PerformanceWeightMatrix


def load_config_from_yaml(config_path: str | None = None) -> dict[str, Any]:
    """
    Load weighted performance configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Dictionary containing configuration data

    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if yaml is None:
        msg = (
            "PyYAML is required for YAML configuration loading. "
            "Install with: pip install PyYAML"
        )
        raise ImportError(msg)

    if config_path is None:
        # Use default config file in same directory
        config_path = Path(__file__).parent / "weighted_performance_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_weight_matrix_from_config(
    config_path: str | None = None, experimental_config: str | None = None
) -> PerformanceWeightMatrix:
    """
    Create PerformanceWeightMatrix from YAML configuration.

    Args:
        config_path: Path to YAML config file. If None, uses default config.
        experimental_config: Name of experimental config to use instead of default

    Returns:
        PerformanceWeightMatrix configured from YAML file
    """
    config = load_config_from_yaml(config_path)

    # Use experimental config if specified
    if experimental_config:
        if "experimental_configs" not in config:
            msg = "No experimental_configs section found in configuration"
            raise ValueError(msg)
        if experimental_config not in config["experimental_configs"]:
            available = list(config["experimental_configs"].keys())
            msg = (
                f"Experimental config '{experimental_config}' not found. "
                f"Available: {available}"
            )
            raise ValueError(msg)

        # Merge experimental config with base config
        exp_config = config["experimental_configs"][experimental_config]
        config.update(exp_config)

    # Extract weight matrices
    goalkeeper_weights = config.get("goalkeeper_weights", {})
    defender_weights = config.get("defender_weights", {})
    midfielder_weights = config.get("midfielder_weights", {})
    forward_weights = config.get("forward_weights", {})

    # Extract difficulty adjustments
    difficulty_adjustments = config.get("difficulty_adjustments", {})
    # Convert string keys to int keys for difficulty levels
    difficulty_adjustments = {int(k): v for k, v in difficulty_adjustments.items()}

    # Extract normalization ranges
    normalization_ranges = config.get("normalization_ranges", {})
    # Convert lists to tuples
    normalization_ranges = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in normalization_ranges.items()
    }

    return PerformanceWeightMatrix(
        goalkeeper_weights=goalkeeper_weights,
        defender_weights=defender_weights,
        midfielder_weights=midfielder_weights,
        forward_weights=forward_weights,
        difficulty_adjustments=difficulty_adjustments,
        normalization_ranges=normalization_ranges,
    )


def validate_weight_matrix(weight_matrix: PerformanceWeightMatrix) -> dict[str, Any]:
    """
    Validate weight matrix configuration for common issues.

    Args:
        weight_matrix: PerformanceWeightMatrix to validate

    Returns:
        Dictionary with validation results and warnings
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "position_weight_sums": {},
    }

    positions = ["GK", "DEF", "MID", "FWD"]

    for position in positions:
        try:
            weights = weight_matrix.get_weights_for_position(position)
            weight_sum = sum(abs(w) for w in weights.values())
            validation_results["position_weight_sums"][position] = weight_sum

            # Check if weights sum to reasonable value (around 1.0)
            if abs(weight_sum - 1.0) > 0.1:
                validation_results["warnings"].append(
                    f"{position} weights sum to {weight_sum:.2f}, expected ~1.0"
                )

            # Check for zero weights (might be intentional but worth noting)
            zero_weights = [k for k, v in weights.items() if v == 0]
            if zero_weights:
                validation_results["warnings"].append(
                    f"{position} has zero weights for: {zero_weights}"
                )

            # Check for very large weights
            large_weights = {k: v for k, v in weights.items() if abs(v) > 0.5}
            if large_weights:
                validation_results["warnings"].append(
                    f"{position} has large weights (>0.5): {large_weights}"
                )

        except Exception as e:
            validation_results["errors"].append(f"Error validating {position}: {e}")
            validation_results["valid"] = False

    # Validate difficulty adjustments
    difficulty_keys = set(weight_matrix.difficulty_adjustments.keys())
    expected_keys = {1, 2, 3, 4, 5}
    if difficulty_keys != expected_keys:
        validation_results["warnings"].append(
            f"Difficulty adjustments missing keys: {expected_keys - difficulty_keys}"
        )

    # Check for reasonable difficulty adjustment ranges
    for difficulty, factor in weight_matrix.difficulty_adjustments.items():
        if factor < 0.5 or factor > 2.0:
            validation_results["warnings"].append(
                f"Difficulty {difficulty} has extreme adjustment factor: {factor}"
            )

    return validation_results


def save_weight_matrix_to_yaml(
    weight_matrix: PerformanceWeightMatrix, output_path: str
) -> None:
    """
    Save PerformanceWeightMatrix to YAML configuration file.

    Args:
        weight_matrix: PerformanceWeightMatrix to save
        output_path: Path where to save the YAML file
    """
    if yaml is None:
        msg = (
            "PyYAML is required for YAML configuration saving. "
            "Install with: pip install PyYAML"
        )
        raise ImportError(msg)

    config = {
        "goalkeeper_weights": weight_matrix.goalkeeper_weights,
        "defender_weights": weight_matrix.defender_weights,
        "midfielder_weights": weight_matrix.midfielder_weights,
        "forward_weights": weight_matrix.forward_weights,
        "difficulty_adjustments": weight_matrix.difficulty_adjustments,
        "normalization_ranges": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in weight_matrix.normalization_ranges.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Default weight matrix loaded from configuration
try:
    default_weight_matrix = create_weight_matrix_from_config()
except (ImportError, FileNotFoundError) as e:
    # Fall back to default if config loading fails
    print(f"Warning: Could not load configuration ({e}), using defaults")
    default_weight_matrix = PerformanceWeightMatrix()


def get_default_weight_matrix() -> PerformanceWeightMatrix:
    """Get the default weight matrix (loaded from config or fallback)."""
    return default_weight_matrix


def reload_default_weight_matrix(
    config_path: str | None = None,
) -> PerformanceWeightMatrix:
    """
    Reload the default weight matrix from configuration file.

    Args:
        config_path: Path to config file, uses default if None

    Returns:
        Newly loaded PerformanceWeightMatrix
    """
    global default_weight_matrix
    default_weight_matrix = create_weight_matrix_from_config(config_path)
    return default_weight_matrix


if __name__ == "__main__":
    # Example usage and validation
    print("Loading and validating default weight matrix configuration...")

    try:
        matrix = create_weight_matrix_from_config()
        validation = validate_weight_matrix(matrix)

        print(f"Configuration valid: {validation['valid']}")

        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

        if validation["errors"]:
            print("\nErrors:")
            for error in validation["errors"]:
                print(f"  - {error}")

        print("\nPosition weight sums:")
        for position, weight_sum in validation["position_weight_sums"].items():
            print(f"  {position}: {weight_sum:.3f}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
