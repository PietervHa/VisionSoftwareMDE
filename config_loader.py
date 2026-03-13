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
        config_path = Path(__file__).parent / "config" / "default.yaml"

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


# Load configuration at module level
cfg = _load_config()

