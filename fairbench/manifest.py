"""The FAIR-Bench dataset: 3 motion types x 6 intervention groups x 3 variants = 18 clips.

Everything downstream reads the dataset through :class:`Dataset`, which is loaded from
one ``manifest.json`` (format documented in ``docs/DATA_FORMAT.md``).

Frame indexing
--------------
Clips are ``num_frames`` long (144 = 6 s at 24 fps). The first ``input_frames`` (48 = 2 s)
are given to the model. All metrics work on the *evaluation grid*: index ``j = 0`` is the
last conditioning frame (original frame 47) and ``j = 1 .. n_gen`` are the generated
frames (original frames 48 .. 143). ``Dataset.eval_index(j)`` maps back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import Camera, Plane, motion_plane, projected_radius_px

MOTIONS = ("thrown", "bounce", "occlude")
INTERVENTIONS = ("color", "bg", "angle")

#: The 6 intervention groups of the proposal and the order of their 3 variants.
GROUP_VARIANTS: dict[str, tuple[str, ...]] = {
    "thrown/color": ("r", "g", "b"),
    "thrown/bg": ("1", "2", "3"),
    "thrown/angle": ("1", "2", "3"),
    "bounce/color": ("r", "g", "b"),
    "bounce/bg": ("1", "2", "3"),
    "occlude/color": ("r", "g", "b"),
}

#: Which motions are ballistic (gravity + floor contacts) vs. rolling on the floor.
MOTION_KIND = {"thrown": "ballistic", "bounce": "ballistic", "occlude": "rolling"}


def group_id(motion: str, intervention: str) -> str:
    return f"{motion}/{intervention}"


def clip_id(motion: str, intervention: str, variant: str) -> str:
    return f"{motion}_{intervention}_{variant}"


def _load_array(value, root: Path) -> np.ndarray | None:
    """Arrays may be inline lists or paths to .npy / .json / .csv files."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    if p.suffix == ".npy":
        return np.load(p).astype(float)
    if p.suffix == ".json":
        return np.asarray(json.loads(p.read_text()), dtype=float)
    if p.suffix in (".csv", ".txt"):
        return np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None).astype(float)
    raise ValueError(f"unsupported array file: {p}")


@dataclass
class MotionGT:
    """Simulator ground truth shared by every clip of one motion type."""

    name: str
    kind: str  # "ballistic" | "rolling"
    positions: np.ndarray  # (num_frames, 3) world positions of the ball center
    events: dict = field(default_factory=dict)  # e.g. {"input_contacts_s": [0.70, 1.48]}
    plane: Plane | None = None  # optional override of the motion plane


@dataclass
class Clip:
    id: str
    motion: str
    intervention: str
    variant: str
    video: Path  # full render: input frames + simulator continuation (the reference)
    camera: Camera
    ball_color: tuple[int, int, int] | None = None  # RGB 0-255, helps trackers
    visible_fraction: np.ndarray | None = None  # (num_frames,) share of the ball visible

    @property
    def group(self) -> str:
        return group_id(self.motion, self.intervention)


@dataclass
class Dataset:
    root: Path
    fps: float
    num_frames: int
    input_frames: int
    width: int
    height: int
    gravity: np.ndarray  # (3,) world gravity vector, m/s^2
    ball_radius: float  # m
    floor_height: float  # height of the floor surface along `up`, m
    motions: dict[str, MotionGT]
    clips: dict[str, Clip]
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- frames
    @property
    def n_gen(self) -> int:
        return self.num_frames - self.input_frames

    @property
    def g(self) -> float:
        return float(np.linalg.norm(self.gravity))

    @property
    def up(self) -> np.ndarray:
        return -self.gravity / np.linalg.norm(self.gravity)

    def eval_index(self, j: np.ndarray | int) -> np.ndarray | int:
        """Evaluation-grid index (0 = last conditioning frame) -> original frame index."""
        return np.asarray(j) + self.input_frames - 1

    def eval_times(self) -> np.ndarray:
        """Times in seconds after the last conditioning frame of eval frames 0..n_gen."""
        return np.arange(self.n_gen + 1) / self.fps

    # ------------------------------------------------------------------ groups
    def groups(self) -> dict[str, list[Clip]]:
        out: dict[str, list[Clip]] = {}
        for gid, variants in GROUP_VARIANTS.items():
            motion, intervention = gid.split("/")
            clips = [self.clips.get(clip_id(motion, intervention, v)) for v in variants]
            if all(c is not None for c in clips):
                out[gid] = clips  # type: ignore[assignment]
        return out

    def clips_of(self, motion: str | None = None) -> list[Clip]:
        return [c for c in self.clips.values() if motion is None or c.motion == motion]

    # ------------------------------------------------------- ground truth helpers
    def gt_world(self, clip: Clip, eval_grid: bool = True) -> np.ndarray:
        """GT world positions (n_gen+1, 3) on the eval grid, or (num_frames, 3)."""
        P = self.motions[clip.motion].positions
        if eval_grid:
            return P[self.input_frames - 1 :]
        return P

    def gt_pixels(self, clip: Clip, eval_grid: bool = True) -> np.ndarray:
        return clip.camera.project(self.gt_world(clip, eval_grid))

    def gt_depth(self, clip: Clip, eval_grid: bool = True) -> np.ndarray:
        return clip.camera.depth(self.gt_world(clip, eval_grid))

    def gt_radius_px(self, clip: Clip, eval_grid: bool = True) -> np.ndarray:
        return projected_radius_px(clip.camera, self.gt_world(clip, eval_grid), self.ball_radius)

    def visible_fraction(self, clip: Clip, eval_grid: bool = True) -> np.ndarray:
        """Share of the ball visible per frame (occluder and image border)."""
        uv = self.gt_pixels(clip, eval_grid=False)
        r = self.gt_radius_px(clip, eval_grid=False)
        inside = (
            (uv[:, 0] - r >= 0)
            & (uv[:, 0] + r <= clip.camera.width - 1)
            & (uv[:, 1] - r >= 0)
            & (uv[:, 1] + r <= clip.camera.height - 1)
        ).astype(float)
        vis = inside if clip.visible_fraction is None else np.minimum(clip.visible_fraction, inside)
        return vis[self.input_frames - 1 :] if eval_grid else vis

    def gt_visible(self, clip: Clip, full: float = 0.99) -> np.ndarray:
        """Eval-grid frames where the GT ball is fully visible (positions comparable)."""
        return self.visible_fraction(clip) >= full

    def gt_hidden(self, clip: Clip, hidden: float = 0.01) -> np.ndarray:
        return self.visible_fraction(clip) <= hidden

    def plane(self, clip: Clip) -> Plane:
        m = self.motions[clip.motion]
        if m.plane is not None:
            return m.plane
        return motion_plane(m.positions, clip.camera)

    def floor_contact_height(self) -> float:
        """Height (along up) of the ball center when it touches the floor."""
        return self.floor_height + self.ball_radius

    def height_of(self, X: np.ndarray) -> np.ndarray:
        """Height of world points along `up` (m)."""
        return np.asarray(X) @ self.up

    # --------------------------------------------------------------------- IO
    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        if path.is_dir():
            path = path / "manifest.json"
        root = path.parent
        m = json.loads(path.read_text())
        motions = {}
        for name, md in m["motions"].items():
            pos = _load_array(md["positions"], root)
            plane = Plane.from_dict(md["plane"]) if md.get("plane") else None
            motions[name] = MotionGT(
                name=name,
                kind=md.get("kind", MOTION_KIND.get(name, "ballistic")),
                positions=pos,
                events=md.get("events", {}),
                plane=plane,
            )
        clips = {}
        for cd in m["clips"]:
            vf = _load_array(cd.get("visible_fraction"), root)
            cam = Camera.from_dict(cd["camera"])
            video = Path(cd["video"])
            clips[cd["id"]] = Clip(
                id=cd["id"],
                motion=cd["motion"],
                intervention=cd["intervention"],
                variant=str(cd["variant"]),
                video=video if video.is_absolute() else root / video,
                camera=cam,
                ball_color=tuple(cd["ball_color"]) if cd.get("ball_color") else None,
                visible_fraction=vf,
            )
        ds = cls(
            root=root,
            fps=float(m.get("fps", 24)),
            num_frames=int(m.get("num_frames", 144)),
            input_frames=int(m.get("input_frames", 48)),
            width=int(m.get("width", 1280)),
            height=int(m.get("height", 720)),
            gravity=np.asarray(m.get("gravity", [0.0, 0.0, -9.81]), dtype=float),
            ball_radius=float(m["ball_radius"]),
            floor_height=float(m.get("floor_height", 0.0)),
            motions=motions,
            clips=clips,
            meta=m.get("meta", {}),
        )
        ds.check()
        return ds

    def check(self) -> None:
        for name, mo in self.motions.items():
            if mo.positions.shape != (self.num_frames, 3):
                raise ValueError(f"motion {name}: positions must be ({self.num_frames}, 3), got {mo.positions.shape}")
        for c in self.clips.values():
            if c.motion not in self.motions:
                raise ValueError(f"clip {c.id}: unknown motion {c.motion}")
            if c.visible_fraction is not None and c.visible_fraction.shape != (self.num_frames,):
                raise ValueError(f"clip {c.id}: visible_fraction must have {self.num_frames} entries")

    def save(self, path: str | Path, arrays_inline: bool = False) -> Path:
        """Write manifest.json (+ .npy arrays next to it unless ``arrays_inline``)."""
        path = Path(path)
        root = path.parent
        (root / "gt").mkdir(parents=True, exist_ok=True)

        def put(arr: np.ndarray, name: str):
            if arrays_inline:
                return np.asarray(arr).tolist()
            rel = Path("gt") / f"{name}.npy"
            np.save(root / rel, np.asarray(arr))
            return str(rel)

        out = {
            "format": "fairbench-manifest/1",
            "fps": self.fps,
            "num_frames": self.num_frames,
            "input_frames": self.input_frames,
            "width": self.width,
            "height": self.height,
            "gravity": self.gravity.tolist(),
            "ball_radius": self.ball_radius,
            "floor_height": self.floor_height,
            "meta": self.meta,
            "motions": {
                n: {
                    "kind": mo.kind,
                    "positions": put(mo.positions, f"{n}_positions"),
                    "events": mo.events,
                    **({"plane": mo.plane.to_dict()} if mo.plane is not None else {}),
                }
                for n, mo in self.motions.items()
            },
            "clips": [
                {
                    "id": c.id,
                    "motion": c.motion,
                    "intervention": c.intervention,
                    "variant": c.variant,
                    "video": str(c.video.relative_to(root)) if c.video.is_relative_to(root) else str(c.video),
                    "camera": c.camera.to_dict(),
                    **({"ball_color": list(c.ball_color)} if c.ball_color else {}),
                    **(
                        {"visible_fraction": put(c.visible_fraction, f"{c.id}_visible")}
                        if c.visible_fraction is not None
                        else {}
                    ),
                }
                for c in self.clips.values()
            ],
        }
        path.write_text(json.dumps(out, indent=2))
        return path


def expected_clip_ids() -> list[str]:
    out = []
    for gid, variants in GROUP_VARIANTS.items():
        motion, intervention = gid.split("/")
        out += [clip_id(motion, intervention, v) for v in variants]
    return out


def iter_pairs(items: Iterable) -> list[tuple]:
    items = list(items)
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
