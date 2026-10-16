from pathlib import Path

import numpy as np
import pytest

from fairbench.manifest import GROUP_VARIANTS, Clip, Dataset, clip_id
from fairbench.synthetic import BALL_COLORS, SceneSpec, make_cameras, make_motions


def memory_dataset(width: int = 640, height: int = 360) -> Dataset:
    """The synthetic design without rendering videos (metrics only need geometry)."""
    spec = SceneSpec()
    motions = make_motions(spec)
    cams = make_cameras(spec, width, height)
    clips = {}
    for gid, variants in GROUP_VARIANTS.items():
        motion, intervention = gid.split("/")
        for v in variants:
            cid = clip_id(motion, intervention, v)
            cam = cams[v] if intervention == "angle" else cams["front"]
            clips[cid] = Clip(cid, motion, intervention, v, Path(f"/nonexistent/{cid}.mp4"), cam,
                              BALL_COLORS.get(v, BALL_COLORS["r"]), None)
    return Dataset(root=Path("/tmp"), fps=24.0, num_frames=144, input_frames=48, width=width, height=height,
                   gravity=np.array([0, 0, -9.81]), ball_radius=spec.radius, floor_height=0.0,
                   motions=motions, clips=clips)


@pytest.fixture(scope="session")
def mem_ds() -> Dataset:
    return memory_dataset()
