# Metric definitions

All three scores are computed from one tracked ball trajectory per generated video, on
the evaluation grid ($t = 0$: last conditioning frame, $t = 1..T$: generated frames,
$T = 96$). Per-frame distances pass a 5-frame running median (window shrinks at the ends)
before averaging over $t = 1..T$; frames where the ground-truth ball is less than 99%
visible are not compared.

## Shared measurement

**Alignment.** Each output is mapped to the benchmark clock by timestamp (never by frame
index) and to reference pixels by scale; tracks are resampled onto the 24 fps grid by
linear interpolation in time. Replayed input is detected automatically (the first ~0.5 s
of output match the input's frame sequence, from its first frame or, for APIs that keep
only the end of the input, from a later one); the replay fixes the output's clock and is
then dropped. Frame 0 of every aligned video is the last conditioning frame itself, where
the ground truth is exact, and the tracker is initialised there. A continuation that
starts with a copy of the last input frame must be declared per model
(`align.repeats_last_input`), because a lagging or frozen model produces the same thing.

**Camera-motion flag.** Background corners are tracked from frame 0 (Lucas-Kanade,
forward-backward checked, ball masked) and a similarity motion is fitted with RANSAC; a
video is flagged when the median image point moves more than 2 px. Without enough
textured background (plain walls) the check reports "unknown" instead of guessing. The
flag is reported, not used in any score.

**Sphere-center correction.** A sphere of radius $r$ whose center has camera coordinates
$(X, Y, Z)$ projects to an ellipse, not a circle centered on the projected center. Let
$\hat c = c / \lVert c \rVert$ be the unit direction to the center and $\sin\theta = r / \lVert c\rVert$
the half-angle of the tangent cone. Points $d = (x, y, 1)$ on the outline satisfy
$(d \cdot \hat c)^2 = \cos^2\theta \, \lVert d \rVert^2$, i.e. $d^\top M d = 0$ with
$M = \hat c \hat c^\top - \cos^2\theta \, I$. Writing $u = (\hat c_1, \hat c_2)$, the conic's
center is $p_0 = -A^{-1} b$ with $A = u u^\top - \cos^2\theta\, I$ and $b = \hat c_3 u$, which gives

$$p_0 = \frac{\hat c_3\, u}{\hat c_3^2 - \sin^2\theta} = \frac{p_c}{1 - r^2 / Z^2}, \qquad p_c = \frac{u}{\hat c_3} = \left(\frac{X}{Z}, \frac{Y}{Z}\right).$$

So in normalized image coordinates the tracked centroid is corrected by
$p_c = (1 - r^2/Z^2)\, p_0$. The factor is second order in $r/Z$, so the ground-truth
depth of the reference ball is an adequate $Z$ for generated balls. Check
(`tests/test_geometry.py`): an exactly rasterized off-axis sphere has a 1.46 px centroid
bias, 0.002 px after correction.

**Lost ball.** If the ground-truth ball is fully visible but nothing is tracked, the
per-frame distance is set to a penalty (default: 10 one-frame lags of that metric,
`missing_policy: penalty`), and the lost-frame share is reported separately.

## 1. Intervention robustness

Color and background groups (one camera):

$$D_{ab} = \frac{1}{T}\sum_{t=1}^{T} \operatorname{med}_5\!\left(\lVert p_{a,t} - p_{b,t}\rVert \frac{Z_t}{f}\right), \qquad D = \tfrac13\,(D_{12} + D_{13} + D_{23}),$$

with $Z_t$ the true depth and $f$ the focal length (meters at the ball's depth). With $K$
samples per clip, $D_{ab}$ averages all $K^2$ sample pairs, and the same formula on two
samples of the *same* input gives the seed floor.

Camera-angle group (VIMS): for view $k$ with center $C_k$, intrinsics $K_k$ and
camera-to-world rotation $R_k$, the line of sight is $d_{kt} \propto R_k K_k^{-1}[u_{kt}, v_{kt}, 1]^\top$, and

$$\hat X_t = \Big(\sum_k P_{kt}\Big)^{-1} \sum_k P_{kt} C_k, \qquad P_{kt} = I - d_{kt} d_{kt}^\top,$$

$$e_t = \sqrt{\tfrac13 \sum_k \lVert P_{kt}(\hat X_t - C_k)\rVert^2}, \qquad E = \frac1T \sum_{t=1}^{T} \operatorname{med}_5(e_t).$$

0-1 score (proposal's definition for VIMS; the harness uses the same construction for $D$):

$$S = \exp\!\left(-\frac{\max(E - E_{\mathrm{ceil}}, 0)}{E_{\mathrm{ref}} - E_{\mathrm{ceil}}}\right).$$

$E_{\mathrm{ceil}}$: $E$ of ground-truth tracks with Gaussian noise of 0.3 px per axis
(Monte Carlo mean). $E_{\mathrm{ref}}$: $E$ when one view lags by one frame (mean over
which view). Both depend only on the scene (`metrics/calibration.py`); `fairbench
validate` prints them to compare with the proposal's 1.3 mm and 14.7 mm.

## 2. Physical plausibility (Morpheus-style)

Reference: Tragoudaras et al., arXiv 2504.02918, code github.com/physics-from-video/Morpheus
(MIT). Formulas follow `morpheus/scoring/{dynamical_score,physical_score}.py`.

1. **Lift.** Each tracked pixel is intersected with the plane containing the
   ground-truth trajectory (for a straight-line trajectory, the plane through it that is
   most fronto-parallel). Coordinates: $s$ (horizontal, in-plane) and $h$ (height), meters.
2. **Segment.** Floor contacts are sharp local minima of $h$ near the floor (the sampled
   minimum may sit above the floor by up to ~half a frame of fall); resting phases are runs
   of at least 4 frames on the floor. Flights obey $\partial^2 s/\partial t^2 = 0$ and
   $\partial^2 h/\partial t^2 = -g$; rolling stretches keep $h$ constant and allow a constant
   along-track acceleration (friction).
3. **Dynamical score.** $S_{\mathrm{dyn}} = \max(1 - \mathrm{NMSE}, 0)$,
   $\mathrm{NMSE} = \sum_t \lVert y_t - \hat y_t\rVert^2 / \sum_t \lVert y_t - \bar y\rVert^2$.
   `pinn`: per segment a 1-20-20-2 tanh MLP, Adam (lr $10^{-3}$), loss = data MSE +
   $\lambda \times$ physics residual ($\lambda = 1$), up to 20000 epochs with early stopping
   (patience 1000) (Morpheus' setup). The segments of one video are trained as one batch of
   independent networks (identical updates, ~1/S of the time for S segments). Fewer
   epochs underfit multi-bounce tracks (S_dyn ≈ 0.4 at 400 epochs on the true bounce
   trajectory, 0.99 at 2000, 1.00 at 20000).
   `lsq`: the same equations imposed exactly (closed-form least squares, no $\lambda$).
4. **Physical invariance score.** Velocities as in Morpheus (0.7 x local linear
   regression + 0.3 x central difference, then Savitzky-Golay, order 3, window 5).
   Quantities: energy per mass $\tfrac12(v_s^2 + v_h^2) + g(h - h_c)$, vertical (flight) or
   along-track (rolling) acceleration, horizontal velocity. In windows of a quarter of the
   trajectory (inside one segment; step = half a window):
   $S = 1/(1 + \mathrm{std}/|\mathrm{mean}|)$ if $|\mathrm{mean}| \ge 10\,\mathrm{std}$ or
   $|\mathrm{mean}| \ge 10$, else $1/(1 + \mathrm{std})$; best window per quantity; mean over quantities.
5. **Discard** (scores 0): ball lost in >20% of visible frames, more than one ball-like blob
   in >10% of frames, or the ball never gets 10% as far from its start as the true ball.
6. **Extra diagnostic** (not in Morpheus): energy per mass rising by >10% across a floor
   contact, which per-flight fits cannot see.

Differences from Morpheus, for the paper's appendix: calibrated meters/seconds instead of
pixel units with a fixed rescaling; per-flight segmentation (Morpheus fits one PINN with
$\partial^2 y/\partial t^2 = -g$ over a short clip); invariance windows restricted to one flight;
NMSE normalized by the total variance of the 2D trajectory (Morpheus' projectile code
uses the variance of the concatenated x and y series); floor-referenced potential
energy (Morpheus uses the trajectory's minimum height); spatial-extent stillness test
(robust to tracker jitter).

## 3. Reference error

Front camera: $\varepsilon_t = \lVert p_t - p^{\mathrm{sim}}_t\rVert Z_t / f$. Camera-angle group:
distance from the true 3D position to the generated line of sight,
$\varepsilon_t = \lVert P_t (X^{\mathrm{sim}}_t - C)\rVert$. Mean over frames after the median filter;
the 0-1 score uses the same exponential mapping with this metric's own floor and
one-frame-lag reference.

## Tracker floor

`fairbench validate` tracks the 18 reference renders and computes every group's
robustness on them. The three variants of a group show the same motion, so anything
below 1 is measurement error; a group where the floor is low (on the synthetic test data:
the red ball on the brick wall) is one where the tracker is not invariant to the
intervention, and model scores there must be read against the floor.
