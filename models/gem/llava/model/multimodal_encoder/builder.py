import os
from pathlib import Path

from .clip_encoder import CLIPECGTower, CLIPVisionTower, CLIPVisionTowerS2


def build_ecg_tower(ecg_tower_cfg, **kwargs):
    checkpoint_path = getattr(ecg_tower_cfg, "mm_ecg_tower", getattr(ecg_tower_cfg, "ecg_tower", None))
    is_absolute_path_exists = os.path.exists(checkpoint_path)
    if is_absolute_path_exists:
        return CLIPECGTower(checkpoint_path, args=ecg_tower_cfg, **kwargs)

    if os.environ.get("HF_HOME", None):
        CACHE_DIR = Path(os.environ["HF_HOME"]) / "hub"
    elif os.environ.get("TRANSFORMERS_CACHE", None):
        CACHE_DIR = Path(os.environ["TRANSFORMERS_CACHE"])
    else:
        CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

    config_path = (
        CACHE_DIR
        / "models--LANSG--GEM"
        / "snapshots"
        / "69ae5d620e7bbca8c9c6e76d0f75ae160ac137fa"
        / "config.json"
    )

    raise ValueError(
        f"Could not load checkpoint for ECG tower (ECG-coca) from given path: {checkpoint_path}. "
        "This is mostly due to the path not existing, where it must be specified in the field "
        f"`mm_ecg_tower` in the model config: {config_path}. If you don't have the checkpoint "
        "locally, please download it from "
        "https://drive.google.com/drive/folders/1-0lRJy7PAMZ7bflbOszwhy3_ZwfTlGYB?usp=sharing"
        " and update `mm_ecg_tower` in the model config to point to the downloaded checkpoint."
    )


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
