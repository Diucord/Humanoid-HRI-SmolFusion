import os, yaml

def load_settings():
    profile = os.environ.get("PROFILE", "nx")
    cfg_path = os.path.join(os.path.dirname(__file__), "config", f"app.{profile}.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # .env 값으로 ${VAR} 치환
    def _expand(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            return os.environ.get(v[2:-1], "")
        return v

    def _walk(d):
        if isinstance(d, dict):
            return {k: _walk(_expand(v)) for k, v in d.items()}
        if isinstance(d, list):
            return [_walk(_expand(x)) for x in d]
        return _expand(d)

    return _walk(cfg)
