"""Reading and writing videos (mp4 via ffmpeg, or folders of PNG/JPEG frames)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

_IMG_EXT = (".png", ".jpg", ".jpeg")


@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {"fps": self.fps, "n_frames": self.n_frames, "width": self.width, "height": self.height}


def read_video(path: str | Path) -> tuple[np.ndarray, float]:
    """Return (frames uint8 [T, H, W, 3], fps). Accepts a video file or a frame folder."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMG_EXT)
        frames = np.stack([np.asarray(imageio.imread(p))[..., :3] for p in files])
        meta = path / "meta.json"
        fps = json.loads(meta.read_text()).get("fps", 24.0) if meta.exists() else 24.0
        return frames, float(fps)
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        fps = float(reader.get_meta_data().get("fps", 24.0))
        frames = np.stack([np.asarray(f)[..., :3] for f in reader])
    finally:
        reader.close()
    return frames, fps


def probe(path: str | Path) -> VideoInfo:
    """Frame count, fps and size via ffprobe (falls back to decoding)."""
    path = Path(path)
    if path.is_dir():
        frames, fps = read_video(path)
        return VideoInfo(fps, len(frames), frames.shape[2], frames.shape[1])
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path),
        ]
        out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        s = out["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return VideoInfo(float(num) / float(den), int(s["nb_read_frames"]), int(s["width"]), int(s["height"]))
    frames, fps = read_video(path)
    return VideoInfo(fps, len(frames), frames.shape[2], frames.shape[1])


def write_video(path: str | Path, frames: np.ndarray, fps: float = 24.0, crf: int = 12) -> Path:
    """Write frames [T, H, W, 3] uint8 as H.264 mp4 (or PNG frames if path has no suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            imageio.imwrite(path / f"{i:05d}.png", f)
        (path / "meta.json").write_text(json.dumps({"fps": fps}))
        return path
    writer = imageio.get_writer(
        str(path), format="ffmpeg", fps=fps, codec="libx264", macro_block_size=1,
        pixelformat="yuv420p", ffmpeg_params=["-crf", str(crf)],
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()
    return path


def write_jpeg_folder(path: str | Path, frames: np.ndarray, quality: int = 95) -> Path:
    """Frames as 00000.jpg, 00001.jpg, ... (the layout SAM 2's video predictor reads)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        imageio.imwrite(path / f"{i:05d}.jpg", f, quality=quality)
    return path


def downscale_gray(frames: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    """Small float grayscale copies used for frame matching."""
    import cv2

    out = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA) for f in frames])
    return out.astype(np.float32) / 255.0
