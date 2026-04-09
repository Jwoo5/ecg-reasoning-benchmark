import base64
import io
import os
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image


class Conversation:
    def __init__(self, prompt: Optional[str] = None):
        self.conversation = []
        if prompt:
            self.conversation.append({"role": "system", "text": prompt})

    def add_user_turn(
        self,
        question: str,
        options: List[str],
        ecg_signal: Optional[torch.Tensor] = None,
        ecg_image: Optional[Image.Image] = None,
    ) -> None:
        turn = {"role": "user", "question": question, "options": options}
        if ecg_signal is not None:
            turn["signal"] = ecg_signal
        if ecg_image:
            turn["image"] = ecg_image
        self.conversation.append(turn)

    def add_model_turn(self, message: str) -> None:
        turn = {"role": "model", "text": message}
        self.conversation.append(turn)

    def get_turns_for_prompt(self) -> list:
        """Return turns for prompt construction.

        If the conversation has progressed beyond the initial diagnostic
        question (more than system + user + model = 3 turns), the initial
        Q&A text is excluded from the history while preserving the ECG
        data (image/signal) from the first user turn.
        """
        turns = self.conversation
        # [0]=system, [1]=user(initial), [2]=model(initial), [3+]=reasoning
        if len(turns) <= 3:
            return turns[1:]

        # Beyond initial: ECG-only stub from turn[1] + turns[3:]
        ecg_stub = {}
        if "signal" in turns[1]:
            ecg_stub["signal"] = turns[1]["signal"]
        if "image" in turns[1]:
            ecg_stub["image"] = turns[1]["image"]

        return [ecg_stub] + turns[3:] if ecg_stub else turns[3:]


def make_letter_indexed(options: List[str]) -> List[str]:
    indexed_options = []
    for i, option in enumerate(options):
        indexed_options.append(f"({chr(ord('a') + i)}) {option}")
    return indexed_options


def base64_image_encoder(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def get_cache_dir() -> Path:
    cache_home = os.environ.get("ERB_CACHE")

    if cache_home:
        target_dir = Path(cache_home)
    else:
        target_dir = Path.home() / ".cache" / "ecg_reasoning_benchmark"

    target_dir.mkdir(parents=True, exist_ok=True)

    return target_dir
