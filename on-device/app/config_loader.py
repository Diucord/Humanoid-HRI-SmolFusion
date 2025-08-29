# config_loader.py
import os, yaml

def load_settings():
    """
    Load application settings from a YAML config file based on the active profile.
    Environment variables can be used inside the YAML file with ${VAR} syntax.

    Returns:
        dict: Parsed and environment-expanded configuration dictionary
    """
    # Determine profile (default = "device")
    profile = os.environ.get("PROFILE", "on-device")

    # Config file path: config/app.<profile>.yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config", f"app.{profile}.yaml")

    # Load YAML config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Replace ${VAR} with corresponding environment variable value
    def _expand(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            return os.environ.get(v[2:-1], "")
        return v

    # Recursively walk through dicts/lists and expand values
    def _walk(d):
        if isinstance(d, dict):
            return {k: _walk(_expand(v)) for k, v in d.items()}
        if isinstance(d, list):
            return [_walk(_expand(x)) for x in d]
        return _expand(d)

    return _walk(cfg)