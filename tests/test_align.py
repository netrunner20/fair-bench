"""Alignment must put any API's output on the benchmark clock."""

import numpy as np
import pytest

from fairbench.align import align_output
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
