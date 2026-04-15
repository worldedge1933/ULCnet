import numpy as np


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono and keep float32 storage."""
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Expected 1D or 2D audio array, got shape {audio.shape!r}.")
