"""A small ray-traced stand-in for the Kubric dataset.

It renders the same 18-clip design as the proposal (thrown / bouncing / occluding ball;
color, back-wall and camera-angle interventions; 6 s at 24 fps; first 2 s = input) with
exact simulator ground truth, so the whole pipeline -- generation mocks, tracking,
metrics, aggregation -- can be run and unit-tested before the real data and model APIs
are plugged in. It is NOT a replacement for the Kubric renders in the paper.

Physics matches the proposal's scene description: no air drag, no rolling friction,
floor contacts in the input at 0.70 / 1.48 s (thrown) and 0.58 / 1.56 s (bouncing),
and a ball rolling at a constant 0.8 m/s behind a box (occluding).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Camera
from .manifest import GROUP_VARIANTS, Clip, Dataset, MotionGT, clip_id
from .video import write_video

G = 9.81
BALL_COLORS = {"r": (200, 35, 35), "g": (35, 165, 60), "b": (45, 75, 215)}
WALL_DEFAULT = "grey"
WALL_VARIANTS = {"1": "blue", "2": "near_black", "3": "brick"}
WALL_RGB = {"grey": (128, 128, 128), "blue": (60, 90, 170), "near_black": (24, 24, 27)}
FLOOR_RGB = (172, 168, 160)
BOX_RGB = (150, 118, 84)
SKY_RGB = (200, 205, 210)


@dataclass
class SceneSpec:
    radius: float = 0.1
    wall_y: float = 2.0
    target: tuple = (0.2, 0.0, 0.8)
    cam_distance: float = 6.5
    fov_x_deg: float = 55.0
    front_height: float = 1.2
    box_center: tuple = (0.35, -0.55, 0.3)  # occluder box (center)
    box_size: tuple = (1.1, 0.4, 0.6)
    roll_y: float = 0.45
    roll_speed: float = 0.8


# ----------------------------------------------------------------------- physics
def simulate_ballistic(p0, v0, t: np.ndarray, radius: float, restitution: float, floor: float = 0.0,
                       v_rest: float = 0.05) -> np.ndarray:
    """Exact piecewise-parabolic motion with floor contacts (z-up world, no drag)."""
    p = np.array(p0, dtype=float)
    v = np.array(v0, dtype=float)
    zc = floor + radius
    out = np.zeros((len(t), 3))
    t_cur, resting = 0.0, False
    k = 0
    while k < len(t):
        if resting:
            out[k:] = p + np.outer(t[k:] - t_cur, [v[0], v[1], 0.0])
            out[k:, 2] = zc
            break
        h = p[2] - zc
        tau = (v[2] + np.sqrt(v[2] ** 2 + 2 * G * max(h, 0.0))) / G  # time to next contact
        t_hit = t_cur + tau
        while k < len(t) and t[k] < t_hit:
            dt = t[k] - t_cur
            out[k] = p + v * dt + np.array([0.0, 0.0, -0.5 * G * dt * dt])
            k += 1
        # contact
        p = p + v * tau + np.array([0.0, 0.0, -0.5 * G * tau * tau])
        p[2] = zc
        v_down = v[2] - G * tau
        v = np.array([v[0], v[1], -restitution * v_down])
        t_cur = t_hit
        if v[2] < v_rest:
            resting = True
    return out


def thrown_initial(t1: float, t2: float, restitution: float, radius: float):
    """Initial height and upward speed so floor contacts happen at t1 and t2."""
    v_plus = G * (t2 - t1) / 2.0
    v_minus = v_plus / restitution
    vz0 = -v_minus + G * t1
    h0 = radius - vz0 * t1 + 0.5 * G * t1**2
    return h0, vz0


def make_motions(spec: SceneSpec, fps: float = 24.0, n: int = 144) -> dict[str, MotionGT]:
    t = np.arange(n) / fps
    r = spec.radius
    # thrown: contacts at 0.70 s and 1.48 s inside the input
    e_th = 0.80
    h0, vz0 = thrown_initial(0.70, 1.48, e_th, r)
    thrown = simulate_ballistic((-1.7, 0.0, h0), (0.62, 0.0, vz0), t, r, e_th)
    # falling and bouncing: dropped from rest (+ small sideways speed), contacts 0.58 / 1.56 s
    t1, t2 = 0.58, 1.56
    hb = r + 0.5 * G * t1**2
    e_b = (G * (t2 - t1) / 2.0) / (G * t1)
    bounce = simulate_ballistic((-0.7, 0.0, hb), (0.30, 0.0, 0.0), t, r, e_b)
    # occluding: constant-speed roll along x behind the box
    x0 = -2.38  # chosen so the box hides the ball from ~2.5 s to ~4.4 s, as in the proposal
    occl =np.stack([x0 + spec.roll_speed * t, np.full(n, spec.roll_y), np.full(n, r)], axis=1)
    return {
        "thrown": MotionGT("thrown", "ballistic", thrown, {"input_contacts_s": [0.70, 1.48], "restitution": e_th}),
        "bounce": MotionGT("bounce", "ballistic", bounce, {"input_contacts_s": [0.58, 1.56], "restitution": round(e_b, 4)}),
        "occlude": MotionGT("occlude", "rolling", occl, {"speed_mps": spec.roll_speed}),
    }


def make_cameras(spec: SceneSpec, width: int, height: int) -> dict[str, Camera]:
    tgt = np.array(spec.target)
    d = spec.cam_distance

    def at(az_deg: float, el_deg: float) -> Camera:
        az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
        eye = tgt + d * np.array([-np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
        return Camera.look_at(eye, tgt, fov_x_deg=spec.fov_x_deg, width=width, height=height)

    front_el = np.degrees(np.arctan2(spec.front_height - tgt[2], d))
    return {
        "front": at(0.0, front_el),
        "1": at(48.0, 8.0),  # 48 deg to the left
        "2": at(-48.0, 8.0),  # 48 deg to the right
        "3": at(0.0, 52.0),  # 52 deg from above
    }


# --------------------------------------------------------------------- rendering
def _wall_color(kind: str, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    if kind != "brick":
        return np.broadcast_to(np.array(WALL_RGB[kind], dtype=float), x.shape + (3,)).copy()
    bw, bh, mortar = 0.26, 0.085, 0.012
    row = np.floor(z / bh)
    xs = x + 0.5 * bw * (row % 2)
    col = np.floor(xs / bw)
    in_mortar = ((xs - col * bw) < mortar) | ((z - row * bh) < mortar)
    rnd = (np.sin(col * 12.9898 + row * 78.233) * 43758.5453) % 1.0
    brick = np.stack([140 + 30 * rnd, 55 + 15 * rnd, 40 + 10 * rnd], axis=-1)
    out = np.where(in_mortar[..., None], np.array([188.0, 182.0, 172.0]), brick)
    return out


def _background(cam: Camera, spec: SceneSpec, wall: str, ss: int, with_box: bool = True):
    """Static background colors and hit distances on the supersampled pixel grid."""
    H, W = cam.height * ss, cam.width * ss
    cs = cam.scaled(W, H)
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = cs.rays(np.stack([uu, vv], axis=-1))
    C = cs.C
    tmin = np.full((H, W), np.inf)
    col = np.broadcast_to(np.array(SKY_RGB, dtype=float), (H, W, 3)).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = -C[2] / d[..., 2]  # floor z = 0
        hit = (tf > 0) & np.isfinite(tf)
        upd = hit & (tf < tmin)
        tmin[upd] = tf[upd]
        P = C + tf[..., None] * d
        shade = 0.85 + 0.15 * np.clip(1 - np.abs(P[..., 0]) / 8, 0, 1)
        col[upd] = (np.array(FLOOR_RGB) * shade[..., None])[upd]
        tw = (spec.wall_y - C[1]) / d[..., 1]  # back wall y = wall_y
        hitw = (tw > 0) & np.isfinite(tw)
        Pw = C + tw[..., None] * d
        hitw &= Pw[..., 2] >= 0
        upd = hitw & (tw < tmin)
        tmin[upd] = tw[upd]
        col[upd] = _wall_color(wall, Pw[..., 0], Pw[..., 2])[upd]
        if with_box:
            lo = np.array(spec.box_center) - 0.5 * np.array(spec.box_size)
            hi = np.array(spec.box_center) + 0.5 * np.array(spec.box_size)
            t1 = (lo - C) / d
            t2 = (hi - C) / d
            tn = np.max(np.minimum(t1, t2), axis=-1)
            tx = np.min(np.maximum(t1, t2), axis=-1)
            hitb = (tx >= tn) & (tx > 0)
            upd = hitb & (tn < tmin)
            tmin[upd] = tn[upd]
            Pb = C + tn[..., None] * d
            face = np.argmin(np.abs(np.stack([Pb[..., 0] - lo[0], Pb[..., 0] - hi[0], Pb[..., 1] - lo[1],
                                              Pb[..., 1] - hi[1], Pb[..., 2] - hi[2]], -1)), axis=-1)
            fshade = np.array([0.80, 0.80, 0.95, 0.70, 1.0])[face]
            col[upd] = (np.array(BOX_RGB) * fshade[..., None])[upd]
    return col, tmin, d, C


def render_clip(cam: Camera, positions: np.ndarray, spec: SceneSpec, wall: str, ball_rgb, with_box: bool,
                ss: int = 2):
    """Render frames (T, H, W, 3) uint8 and the visible share of the ball per frame."""
    bg, tbg, d, C = _background(cam, spec, wall, ss, with_box)
    _, tbg_nobox, _, _ = _background(cam, spec, wall, ss, False) if with_box else (None, tbg, None, None)
    H, W = bg.shape[:2]
    cs = cam.scaled(W, H)
    r = spec.radius
    light = np.array([-0.35, -0.55, 0.76])
    light /= np.linalg.norm(light)
    frames = np.empty((len(positions), cam.height, cam.width, 3), dtype=np.uint8)
    vis = np.zeros(len(positions))
    for k, X in enumerate(positions):
        img = bg.copy()
        uv = cs.project(X[None])[0]
        rad = cs.focal * r / max(cs.depth(X[None])[0] - r, 1e-3)
        m = int(np.ceil(1.6 * rad)) + 2
        u0, u1 = int(max(0, np.floor(uv[0]) - m)), int(min(W, np.ceil(uv[0]) + m))
        v0, v1 = int(max(0, np.floor(uv[1]) - m)), int(min(H, np.ceil(uv[1]) + m))
        if u1 > u0 and v1 > v0:
            dd = d[v0:v1, u0:u1]
            oc = C - X
            b = dd @ oc
            c = oc @ oc - r * r
            disc = b * b - c
            ts = -b - np.sqrt(np.maximum(disc, 0))
            hit_any = (disc >= 0) & (ts > 0)
            hit_nobox = hit_any & (ts < tbg_nobox[v0:v1, u0:u1])
            hit = hit_any & (ts < tbg[v0:v1, u0:u1])
            n_all = hit_nobox.sum()
            vis[k] = hit.sum() / n_all if n_all > 0 else 0.0
            P = C + ts[..., None] * dd
            nrm = (P - X) / r
            lam = 0.30 + 0.70 * np.clip(nrm @ light, 0, 1)
            spec_hl = 0.25 * np.clip(nrm @ light, 0, 1) ** 24
            shaded = np.clip(np.array(ball_rgb) * lam[..., None] + 255 * spec_hl[..., None], 0, 255)
            sub = img[v0:v1, u0:u1]
            sub[hit] = shaded[hit]
        # box filter down to output resolution (anti-aliasing)
        frames[k] = img.reshape(cam.height, ss, cam.width, ss, 3).mean(axis=(1, 3)).round().astype(np.uint8)
    return frames, vis


# ----------------------------------------------------------------------- dataset
def build_synthetic_dataset(out_dir: str | Path, width: int = 640, height: int = 360, ss: int = 2,
                            fps: float = 24.0, num_frames: int = 144, input_frames: int = 48,
                            spec: SceneSpec | None = None, groups: list[str] | None = None,
                            verbose: bool = True) -> Dataset:
    """Render the 18-clip design and write videos + manifest.json to ``out_dir``."""
    spec = spec or SceneSpec()
    out_dir = Path(out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    motions = make_motions(spec, fps, num_frames)
    cams = make_cameras(spec, width, height)
    clips: dict[str, Clip] = {}
    for gid, variants in GROUP_VARIANTS.items():
        if groups and gid not in groups:
            continue
        motion, intervention = gid.split("/")
        for v in variants:
            cid = clip_id(motion, intervention, v)
            ball = BALL_COLORS[v] if intervention == "color" else BALL_COLORS["r"]
            wall = WALL_VARIANTS[v] if intervention == "bg" else WALL_DEFAULT
            cam = cams[v] if intervention == "angle" else cams["front"]
            frames, vis = render_clip(cam, motions[motion].positions, spec, wall, ball,
                                      with_box=(motion == "occlude"), ss=ss)
            path = out_dir / "videos" / f"{cid}.mp4"
            write_video(path, frames, fps=fps)
            clips[cid] = Clip(cid, motion, intervention, v, path, cam, tuple(ball), vis)
            if verbose:
                print(f"  rendered {cid}  (min visible share {vis.min():.2f})")
    ds = Dataset(
        root=out_dir, fps=fps, num_frames=num_frames, input_frames=input_frames, width=width, height=height,
        gravity=np.array([0.0, 0.0, -G]), ball_radius=spec.radius, floor_height=0.0,
        motions={k: m for k, m in motions.items() if any(c.motion == k for c in clips.values())},
        clips=clips, meta={"source": "fairbench.synthetic", "note": "pipeline test data, not the paper dataset"},
    )
    occ = [c for c in clips.values() if c.motion == "occlude"]
    if occ:
        v = occ[0].visible_fraction
        t = np.arange(num_frames) / fps
        hidden = np.where(v < 0.99)[0]
        if len(hidden):
            ds.motions["occlude"].events.update({
                "occlusion_starts_s": round(float(t[hidden[0]]), 3),
                "fully_visible_again_s": round(float(t[hidden[-1] + 1]) if hidden[-1] + 1 < num_frames else -1, 3),
            })
    ds.save(out_dir / "manifest.json")
    return Dataset.load(out_dir / "manifest.json")
