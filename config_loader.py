import sys
import argparse
from pathlib import Path

import yaml

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
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load YAML
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}
        return config
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load config file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


# Load configuration at module level
cfg = _load_config()

