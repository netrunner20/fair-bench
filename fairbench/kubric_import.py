"""Build ``manifest.json`` from Kubric renders (best effort; check the printed report).

Expected input: one folder per clip containing Kubric's ``metadata.json`` (camera +
instances, as written by ``kb.write_json`` with ``kb.get_camera_info`` /
``kb.get_instance_info``) and either ``rgba_00000.png ...`` frames or one ``.mp4``.
By default the folder name is the clip id (``thrown_color_r`` ...); otherwise pass a
mapping file::

    {"ball_radius": 0.1,                 # Kubric sphere 'scale' (radius 1 before scaling)
     "ball_instance": "sphere",          # index, or a substring of asset_id
     "clips": {"thrown_color_r": "renders/thrown/red", ...}}

Kubric conventions handled here: the camera looks along its local -z with +y up
(Blender); intrinsics in metadata are normalized with flipped signs, so pixel K is
rebuilt from focal_length / sensor_width and the image size; quaternions are (w, x, y, z).
Every clip's projection is compared with Kubric's own ``image_positions`` when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import Camera
from .manifest import GROUP_VARIANTS, MOTION_KIND, Clip, Dataset, MotionGT, clip_id, expected_clip_ids
from .video import read_video, write_video

BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_from_kubric(cam_meta: dict, width: int, height: int) -> tuple[Camera, dict]:
    pos = np.asarray(cam_meta["positions"], dtype=float)
    quat = np.asarray(cam_meta["quaternions"], dtype=float)
    moved = float(np.max(np.linalg.norm(pos - pos[0], axis=1)))
    f_px = float(cam_meta["focal_length"]) / float(cam_meta["sensor_width"]) * width
    K = np.array([[f_px, 0, width / 2.0], [0, f_px, height / 2.0], [0, 0, 1.0]])
    R = quat_to_matrix(quat[0]) @ BLENDER_TO_OPENCV
    return Camera(K=K, R=R, C=pos[0], width=width, height=height), {"camera_translation_over_clip_m": moved}


def _pick_instance(instances: list[dict], which) -> dict:
    if isinstance(which, int):
        return instances[which]
    if isinstance(which, str):
        for inst in instances:
            if which.lower() in str(inst.get("asset_id", "")).lower():
                return inst
    moving = [i for i in instances if np.ptp(np.asarray(i["positions"]), axis=0).max() > 1e-3]
    if len(moving) == 1:
        return moving[0]
    raise ValueError("cannot tell which Kubric instance is the ball; set 'ball_instance' in the mapping file")


def _frames(folder: Path) -> tuple[np.ndarray, float | None]:
    pngs = sorted(folder.glob("rgba_*.png"))
    if pngs:
        import imageio.v2 as imageio

        return np.stack([np.asarray(imageio.imread(p))[..., :3] for p in pngs]), None
    mp4s = sorted(folder.glob("*.mp4"))
    if mp4s:
        return read_video(mp4s[0])
    raise FileNotFoundError(f"no rgba_*.png frames or .mp4 in {folder}")


def import_kubric(src: Path, out: Path, mapping: str | None = None, input_frames: int = 48) -> Dataset:
    m = json.loads(Path(mapping).read_text()) if mapping else {}
    folders = {cid: (src / p) for cid, p in m.get("clips", {}).items()} or {cid: src / cid for cid in expected_clip_ids()}
    (out / "videos").mkdir(parents=True, exist_ok=True)
    clips, motions_pos, report = {}, {}, {}
    meta_any = None
    for cid, folder in folders.items():
        if not (folder / "metadata.json").exists():
            report[cid] = "missing"
            continue
        meta = json.loads((folder / "metadata.json").read_text())
        meta_any = meta
        frames, fps_file = _frames(folder)
        H, W = frames.shape[1:3]
        cam, cam_info = camera_from_kubric(meta["camera"], W, H)
        inst = _pick_instance(meta["instances"], m.get("ball_instance"))
        pos = np.asarray(inst["positions"], dtype=float)
        motion, intervention, variant = cid.split("_", 2)
        motions_pos.setdefault(motion, []).append(pos)
        r = float(m.get("ball_radius") or np.mean(np.asarray(inst.get("scale", [1.0]))))
        vis = None
        if "visibility" in inst:
            rad = cam.focal * r / np.maximum(cam.depth(pos), 1e-6)
            vis = np.clip(np.asarray(inst["visibility"], dtype=float) / (np.pi * rad**2), 0, 1)
        check = {}
        if "image_positions" in inst:
            ip = np.asarray(inst["image_positions"], dtype=float) * np.array([W, H])
            check["max_px_vs_kubric_image_positions"] = round(float(np.max(np.linalg.norm(cam.project(pos) - ip, axis=1))), 4)
        fps = float(meta.get("metadata", {}).get("frame_rate", fps_file or 24))
        vid = out / "videos" / f"{cid}.mp4"
        write_video(vid, frames, fps=fps, crf=8)
        clips[cid] = Clip(cid, motion, intervention, variant, vid, cam, None, vis)
        report[cid] = {**cam_info, **check, "frames": int(len(frames)), "fps": fps}
    motions = {}
    for motion, plist in motions_pos.items():
        spread = max(float(np.max(np.linalg.norm(p - plist[0], axis=1))) for p in plist)
        motions[motion] = MotionGT(motion, MOTION_KIND.get(motion, "ballistic"), plist[0],
                                   {"max_gt_difference_between_clips_m": spread})
        report[f"{motion}_shared_trajectory_max_diff_m"] = spread
    md = (meta_any or {}).get("metadata", {})
    first = next(iter(clips.values()))
    ds = Dataset(root=out, fps=float(md.get("frame_rate", 24)), num_frames=len(motions[first.motion].positions),
                 input_frames=input_frames, width=first.camera.width, height=first.camera.height,
                 gravity=np.asarray(m.get("gravity", [0, 0, -9.81]), dtype=float),
                 ball_radius=float(m.get("ball_radius", 0.1)), floor_height=float(m.get("floor_height", 0.0)),
                 motions=motions, clips=clips, meta={"source": "kubric_import", "src": str(src)})
    ds.save(out / "manifest.json")
    print(json.dumps(report, indent=2))
    missing = [c for c in expected_clip_ids() if c not in clips]
    if missing:
        print(f"WARNING: missing clips: {missing}")
    return ds


__all__ = ["import_kubric", "camera_from_kubric", "quat_to_matrix", "GROUP_VARIANTS", "clip_id"]
