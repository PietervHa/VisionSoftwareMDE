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
    """Expose canonical provider-based object-detection keys while preserving legacy aliases."""
    od_cfg = config.get("object_detection")
    if not isinstance(od_cfg, dict):
        return config

    normalized = dict(od_cfg)
    classifier_cfg = _as_dict(normalized.get("classifier"))
    roboflow_cfg = _as_dict(normalized.get("roboflow"))

    backend = str(_first_non_empty(normalized.get("backend"), normalized.get("detector_backend"))).strip().lower()
    if backend == "cola":
        backend = "roboflow"
    if not backend:
        if (
            bool(normalized.get("use_remote_detector", False))
            or bool(normalized.get("use_cola_detector", False))
            or bool(roboflow_cfg)
        ):
            backend = "roboflow"
        else:
            backend = "classifier"
    if backend not in {"classifier", "roboflow", "yolo"}:
        backend = "classifier"

    normalized["backend"] = backend
    normalized["detector_backend"] = backend

    classifier_model_path = str(
        _first_non_empty(
            classifier_cfg.get("model_path"),
            normalized.get("classifier_model_path"),
        )
    ).strip()
    classifier_cfg["model_path"] = classifier_model_path
    normalized["classifier"] = classifier_cfg
    normalized["classifier_model_path"] = classifier_model_path

    roboflow_model = str(
        _first_non_empty(
            roboflow_cfg.get("model"),
            normalized.get("roboflow_model"),
            normalized.get("roboflow_model_name"),
            default="default",
        )
    ).strip()
    roboflow_models = roboflow_cfg.get("models") if isinstance(roboflow_cfg.get("models"), dict) else {}
    selected_profile = _as_dict(roboflow_models.get(roboflow_model))

    roboflow_api_key = str(
        _first_non_empty(
            selected_profile.get("api_key"),
            roboflow_cfg.get("api_key"),
            normalized.get("roboflow_api_key"),
            normalized.get("remote_detector_api_key"),
            normalized.get("cola_api_key"),
        )
    ).strip()
    roboflow_workspace = str(
        _first_non_empty(
            selected_profile.get("workspace"),
            roboflow_cfg.get("workspace"),
            normalized.get("roboflow_workspace"),
            normalized.get("remote_detector_workspace"),
            normalized.get("cola_workspace"),
        )
    ).strip()
    roboflow_workflow = str(
        _first_non_empty(
            selected_profile.get("workflow"),
            roboflow_cfg.get("workflow"),
            normalized.get("roboflow_workflow"),
            normalized.get("remote_detector_workflow"),
            normalized.get("cola_workflow"),
        )
    ).strip()
    roboflow_api_url = str(
        _first_non_empty(
            selected_profile.get("api_url"),
            roboflow_cfg.get("api_url"),
            normalized.get("roboflow_api_url"),
            normalized.get("remote_detector_api_url"),
            default="https://serverless.roboflow.com",
        )
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
    normalized["roboflow_model"] = roboflow_model
    normalized["roboflow_api_key"] = roboflow_api_key
    normalized["roboflow_workspace"] = roboflow_workspace
    normalized["roboflow_workflow"] = roboflow_workflow
    normalized["roboflow_api_url"] = roboflow_api_url
    normalized["remote_detector_api_key"] = roboflow_api_key
    normalized["remote_detector_workspace"] = roboflow_workspace
    normalized["remote_detector_workflow"] = roboflow_workflow
    normalized["remote_detector_api_url"] = roboflow_api_url

    # Keep legacy aliases populated for backward compatibility.
    normalized["use_remote_detector"] = backend == "roboflow"
    normalized["use_cola_detector"] = backend == "roboflow"
    normalized["cola_api_key"] = roboflow_api_key
    normalized["cola_workspace"] = roboflow_workspace
    normalized["cola_workflow"] = roboflow_workflow

    config["object_detection"] = normalized
    return config


# Load configuration at module level
cfg = _load_config()
cfg = _normalize_object_detection_config(cfg)

