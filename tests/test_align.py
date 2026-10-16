"""Alignment must put any API's output on the benchmark clock."""

import cv2
import numpy as np
import pytest

from fairbench.align import align_output, camera_motion
from fairbench.synthetic import BALL_COLORS, SceneSpec, make_cameras, make_motions, render_clip


@pytest.fixture(scope="module")
def clip_frames():
    spec = SceneSpec()
    pos = make_motions(spec)["thrown"].positions
    cam = make_cameras(spec, 160, 90)["front"]
    frames, _ = render_clip(cam, pos, spec, "grey", BALL_COLORS["r"], with_box=False, ss=2)
    uv = cam.project(pos[:48])
    roi = (int(uv[:, 0].min() - 10), int(uv[:, 1].min() - 10), int(uv[:, 0].max() + 10), int(uv[:, 1].max() + 10))
    return frames, roi


def test_continuation_only(clip_frames):
    frames, roi = clip_frames
    av = align_output(frames[48:], 24.0, frames[:48], 24.0, roi_ref=roi)
    assert not av.info["includes_input"]
    assert np.array_equal(av.frames[0], frames[47]) and np.array_equal(av.frames[1], frames[48])
    assert np.allclose(av.times[:3], [0, 1 / 24, 2 / 24])


def test_replayed_input_is_detected(clip_frames):
    frames, roi = clip_frames
    av = align_output(frames, 24.0, frames[:48], 24.0, roi_ref=roi)
    assert av.info["includes_input"] and av.info["last_input_index_in_output"] == 47
    assert np.array_equal(av.frames[1], frames[48])


def test_lagging_model_is_not_silently_corrected(clip_frames):
    frames, roi = clip_frames
    lagged = frames[47:143]  # first generated frame equals the last input frame
    av = align_output(lagged, 24.0, frames[:48], 24.0, roi_ref=roi)
    assert av.info["first_frame_matches_last_input"] and not av.info["repeats_last_input"]
    assert np.array_equal(av.frames[1], frames[47])  # the lag is kept, so metrics can see it
    av2 = align_output(lagged, 24.0, frames[:48], 24.0, roi_ref=roi, repeats_last_input=True)
    assert np.array_equal(av2.frames[1], frames[48])  # declared API quirk: first frame dropped


def test_other_frame_rate_keeps_time(clip_frames):
    frames, roi = clip_frames
    # a 12 fps continuation: output frame k shows time (k + 1) / 12 s after the input
    out = frames[48:][1::2]
    av = align_output(out, 12.0, frames[:48], 24.0, roi_ref=roi)
    assert np.allclose(av.times[1:4], [1 / 12, 2 / 12, 3 / 12])
    assert np.array_equal(av.frames[1], frames[49])


def test_replay_at_other_frame_rate(clip_frames):
    frames, roi = clip_frames
    # input replayed at 12 fps, then continued: no output frame sits on the last input frame
    out = frames[0::2]
    av = align_output(out, 12.0, frames[:48], 24.0, roi_ref=roi)
    assert av.info["includes_input"] and av.info["replay_starts_at_input_frame"] == 0
    assert np.array_equal(av.frames[0], frames[47])  # frame 0 is the input itself, at t = 0
    assert np.array_equal(av.frames[1], frames[48]) and np.isclose(av.times[1], 1 / 24)
    assert np.allclose(np.diff(av.times[1:5]), 1 / 12)


def test_trimmed_replay(clip_frames):
    frames, roi = clip_frames
    # an API that keeps only the last 28 input frames and returns them before the continuation
    av = align_output(frames[20:], 24.0, frames[:48], 24.0, roi_ref=roi)
    assert av.info["includes_input"] and av.info["replay_starts_at_input_frame"] == 20
    assert np.array_equal(av.frames[1], frames[48]) and np.isclose(av.times[1], 1 / 24)


def _ball_video(wall: str):
    spec = SceneSpec()
    pos = make_motions(spec)["thrown"].positions
    cam = make_cameras(spec, 320, 180)["front"]
    frames, _ = render_clip(cam, pos, spec, wall, BALL_COLORS["r"], with_box=False, ss=1)
    return frames[47:96], cam.project(pos[47:96])


def test_camera_motion_flags_pans_but_not_a_moving_ball():
    frames, uv = _ball_video("brick")
    r = 6.0
    still = camera_motion(frames, exclude_uv=uv, exclude_radius=r)
    assert still["max_median_disp_px"] is not None and still["max_median_disp_px"] < 0.5
    H, W = frames.shape[1:3]
    shift = 0.1 * np.arange(len(frames))  # a slow pan, 4.8 px by the end
    panned = np.stack([cv2.warpAffine(f, np.float32([[1, 0, s], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REFLECT)
                       for f, s in zip(frames, shift)])
    moved = camera_motion(panned, exclude_uv=uv + np.c_[shift, 0 * shift], exclude_radius=r)
    assert abs(moved["max_median_disp_px"] - shift[-1]) < 0.5
    zoomed = np.stack([cv2.warpAffine(f, cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + 0.002 * i), (W, H),
                                      borderMode=cv2.BORDER_REFLECT) for i, f in enumerate(frames)])
    assert camera_motion(zoomed, exclude_uv=uv, exclude_radius=r)["max_median_disp_px"] > 2.0


def test_camera_motion_is_unknown_not_moved_on_plain_walls():
    frames, uv = _ball_video("grey")
    d = camera_motion(frames, exclude_uv=uv, exclude_radius=6.0)["max_median_disp_px"]
    assert d is None or d < 0.5
