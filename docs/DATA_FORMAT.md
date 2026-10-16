# Dataset format (`manifest.json`)

Everything the harness needs about the 18 clips is in one JSON file next to the videos.
`python -m fairbench import-kubric` writes it from Kubric renders; you can also write it
yourself. `python -m fairbench validate` checks it.

```json
{
  "format": "fairbench-manifest/1",
  "fps": 24, "num_frames": 144, "input_frames": 48,
  "width": 1280, "height": 720,
  "gravity": [0, 0, -9.81],
  "ball_radius": 0.1,
  "floor_height": 0.0,
  "motions": {
    "thrown":  {"kind": "ballistic", "positions": "gt/thrown_positions.npy",
                "events": {"input_contacts_s": [0.70, 1.48]}},
    "bounce":  {"kind": "ballistic", "positions": "gt/bounce_positions.npy",
                "events": {"input_contacts_s": [0.58, 1.56]}},
    "occlude": {"kind": "rolling",   "positions": "gt/occlude_positions.npy",
                "events": {"speed_mps": 0.8}}
  },
  "clips": [
    {"id": "thrown_color_r", "motion": "thrown", "intervention": "color", "variant": "r",
     "video": "videos/thrown_color_r.mp4",
     "camera": {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "R_c2w": [[...], [...], [...]], "C": [x, y, z], "width": 1280, "height": 720},
     "ball_color": [200, 35, 35],
     "visible_fraction": "gt/thrown_color_r_visible.npy"}
  ]
}
```

| Field | Meaning |
| --- | --- |
| `positions` | `(144, 3)` world positions of the ball **center**, one per frame; shared by every clip of the motion (the proposal's check: all clips of a motion share exactly the same 3D trajectory). `.npy`, `.json`, `.csv` or an inline list. |
| `events.input_contacts_s` | floor contacts inside the input; used to check that a model's conditioning window covers the last one (the determined-future condition). |
| clip `id` | `<motion>_<intervention>_<variant>`: motions `thrown`, `bounce`, `occlude`; interventions `color` (variants `r g b`), `bg` (`1 2 3`), `angle` (`1 2 3`). |
| `video` | the full 6 s render: input frames followed by the simulator's continuation (the reference). mp4 or a folder of PNG frames. |
| `camera.K` | pixel intrinsics. **Pixel centers are at +0.5**: the image's top-left corner is (0, 0), so the center of pixel (0, 0) is (0.5, 0.5) and `cx = width / 2`. Kubric uses this convention; OpenCV calibration usually uses `cx = (width - 1) / 2`, which would add a 0.5 px offset (`validate` warns about systematic offsets). |
| `camera.R_c2w` | camera-to-world rotation, **OpenCV camera axes** (x right, y down, z forward). Columns are the camera axes in world coordinates. (`R_w2c` is accepted too.) |
| `camera.C` | camera center in world coordinates (static camera). |
| `visible_fraction` | optional `(144,)` share of the ball visible per frame (the occluder); without it, visibility is "projected disc inside the image". Position metrics use frames with >= 99% visible. |

Frame indexing used by every metric: the *evaluation grid* has index 0 = last conditioning
frame (original frame 47) and 1..96 = generated frames (original 48..143).
