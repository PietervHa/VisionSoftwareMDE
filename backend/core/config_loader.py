import sys
import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

"""
Configuration loader for the machine vision application.
Loads YAML configuration files and exposes a module-level cfg dictionary.
"""

def _load_config():
    """Load configuration from YAML file (default or CLI override)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")

    # Parse only known arguments to avoid conflicts with other scripts
    args, _ = parser.parse_known_args()

    # Determine config file path
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

    # Check if file exists
    if not config_path.exists():
        log.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    # Load YAML
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}
        return config
    except yaml.YAMLError as e:
        log.error("Failed to parse YAML file %s: %s", config_path, e)
        sys.exit(1)
    except Exception as e:
        log.error("Failed to load config file %s: %s", config_path, e)
        sys.exit(1)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _first_non_empty(*values, default=""):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
            continue
        if value != "":
            return value
    return default


def _normalize_object_detection_config(config: dict) -> dict:
    """Normalize canonical provider-based object-detection configuration."""
    od_cfg = config.get("object_detection")
    if not isinstance(od_cfg, dict):
        return config

    normalized = dict(od_cfg)
    classifier_cfg = _as_dict(normalized.get("classifier"))
    roboflow_cfg = _as_dict(normalized.get("roboflow"))

    backend = str(_first_non_empty(normalized.get("backend"), default="classifier")).strip().lower()
    if backend not in {"classifier", "roboflow", "yolo", "template"}:
        backend = "classifier"

    normalized["backend"] = backend

    classifier_model_path = str(_first_non_empty(classifier_cfg.get("model_path"))).strip()
    classifier_cfg["model_path"] = classifier_model_path
    normalized["classifier"] = classifier_cfg

    roboflow_model = str(_first_non_empty(roboflow_cfg.get("model"), default="default")).strip()
    roboflow_models = roboflow_cfg.get("models") if isinstance(roboflow_cfg.get("models"), dict) else {}
    selected_profile = _as_dict(roboflow_models.get(roboflow_model))

    roboflow_api_key = str(_first_non_empty(selected_profile.get("api_key"))).strip()
    roboflow_workspace = str(_first_non_empty(selected_profile.get("workspace"))).strip()
    roboflow_workflow = str(_first_non_empty(selected_profile.get("workflow"))).strip()
    roboflow_api_url = str(
        _first_non_empty(selected_profile.get("api_url"), default="https://serverless.roboflow.com")
    ).strip()

    if not isinstance(roboflow_models, dict):
        roboflow_models = {}
    if roboflow_model not in roboflow_models:
        roboflow_models[roboflow_model] = {
            "api_key": roboflow_api_key,
            "workspace": roboflow_workspace,
            "workflow": roboflow_workflow,
            "api_url": roboflow_api_url,
        }

    normalized["roboflow"] = {
        "model": roboflow_model,
        "models": roboflow_models,
        "api_key": roboflow_api_key,
        "workspace": roboflow_workspace,
        "workflow": roboflow_workflow,
        "api_url": roboflow_api_url,
    }

    config["object_detection"] = normalized
    return config


def _validate_object_detection_config(config: dict) -> None:
    """Fail fast if Roboflow backend is selected but config is incomplete."""
    od_cfg = config.get("object_detection", {})
    if not isinstance(od_cfg, dict):
        return

    backend = str(od_cfg.get("backend", "classifier")).strip().lower()
    if backend != "roboflow":
        return

    roboflow_cfg = od_cfg.get("roboflow", {})
    if not isinstance(roboflow_cfg, dict):
        raise ValueError("object_detection.roboflow must be a dict when backend='roboflow'")

    selected_model = str(roboflow_cfg.get("model", "")).strip()
    if not selected_model:
        raise ValueError("object_detection.roboflow.model is required when backend='roboflow'")

    roboflow_models = roboflow_cfg.get("models", {})
    if not isinstance(roboflow_models, dict):
        raise ValueError("object_detection.roboflow.models must be a dict")

    if selected_model not in roboflow_models:
        available = list(roboflow_models.keys())
        raise ValueError(
            f"Roboflow model '{selected_model}' not found in roboflow.models. "
            f"Available models: {available}"
        )

    model_profile = roboflow_models[selected_model]
    if not isinstance(model_profile, dict):
        raise ValueError(f"roboflow.models.{selected_model} must be a dict")

    required_fields = {"api_key", "workspace", "workflow"}
    missing = required_fields - set(model_profile.keys())
    if missing:
        raise ValueError(
            f"Roboflow model '{selected_model}' profile is incomplete. "
            f"Missing required fields: {missing}. "
            f"All of {required_fields} are required."
        )

    for field in required_fields:
        value = str(model_profile.get(field, "")).strip()
        if not value:
            raise ValueError(
                f"Roboflow model '{selected_model}' has empty required field '{field}'. "
                f"All required fields must have non-empty values."
            )


# Load configuration at module level
cfg = _load_config()
cfg = _normalize_object_detection_config(cfg)
_validate_object_detection_config(cfg)

