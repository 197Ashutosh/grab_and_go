import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

HOST: str = os.getenv("GRAB_GO_HOST", "127.0.0.1")
PORT: int = int(os.getenv("GRAB_GO_PORT", "8000"))
DEBUG: bool = os.getenv("GRAB_GO_DEBUG", "true").lower() in ("1", "true", "yes")

_PRIMARY_MODEL_PATH = BASE_DIR.parent / "ai_vision_engine" / "yolov8s.pt"
MODEL_PATH: str = os.getenv(
    "GRAB_GO_MODEL_PATH",
    str(_PRIMARY_MODEL_PATH) if _PRIMARY_MODEL_PATH.exists() else "yolov8s.pt",
)

CONFIDENCE_THRESHOLD: float = float(os.getenv("GRAB_GO_CONFIDENCE", "0.30"))

CORS_ORIGINS: list[str] = os.getenv(
    "GRAB_GO_CORS_ORIGINS", "*"
).split(",")

LOG_LEVEL: str = os.getenv("GRAB_GO_LOG_LEVEL", "INFO")
