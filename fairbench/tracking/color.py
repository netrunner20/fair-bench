"""A fast color-histogram ball tracker.

Used for the synthetic data, unit tests and quick checks. The paper's tracker is SAM 2.1
(:mod:`fairbench.tracking.sam2_tracker`); generated videos can drift in color and
lighting, which this tracker does not handle.

Method: learn P(ball | color) from frame 0 (pixels inside the known ball disc vs. the rest
of the frame), back-project it on every frame, threshold, take connected components and
follow the component closest to the predicted position.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Tracker, TrackResult


class ColorTracker(Tracker):
    name = "color"

    def __init__(self, bins: int = 16, threshold: float = 0.5, min_area_frac: float = 0.15,
                 dup_area_frac: float = 0.3, lost_after: int = 3):
        self.bins = bins
        self.threshold = threshold
        self.min_area_frac = min_area_frac
        self.dup_area_frac = dup_area_frac
        self.lost_after = lost_after

    def _index(self, img: np.ndarray) -> np.ndarray:
        q = (img.astype(np.int32) * self.bins) // 256
        return (q[..., 0] * self.bins + q[..., 1]) * self.bins + q[..., 2]

    def _posterior(self, frame0: np.ndarray, uv: np.ndarray, r: float) -> np.ndarray:
        H, W = frame0.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        dist = np.hypot(xx + 0.5 - uv[0], yy + 0.5 - uv[1])
        ball = dist <= max(0.75 * r, 1.0)
        bg = dist >= 1.6 * r + 2
        idx = self._index(frame0)
        nb = self.bins**3
        hb = np.bincount(idx[ball], minlength=nb).astype(float)
        hg = np.bincount(idx[bg], minlength=nb).astype(float)
        # light smoothing over neighbouring bins so nearby shades still count as ball
        hb3 = hb.reshape(self.bins, self.bins, self.bins)
        hg3 = hg.reshape(self.bins, self.bins, self.bins)
        k = np.array([0.25, 0.5, 0.25])
        for ax in range(3):
            hb3 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, hb3)
            hg3 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, hg3)
        hb, hg = hb3.ravel(), hg3.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            post = np.where(hb + hg > 0, hb / (hb + hg), 0.0)
        return post

    @staticmethod
    def _disc_refine(P: np.ndarray, cen: np.ndarray, r: float) -> np.ndarray:
        """Sub-pixel center by matching a disc of radius r to the ball-probability map.

        More robust than the mask centroid when a few ball pixels are misclassified (dark
        shading) or a few background pixels share the ball's color (e.g. red on brick).
        """
        R = int(np.ceil(r + 1))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        kern = np.clip(r + 0.5 - np.hypot(xx, yy), 0.0, 1.0).astype(np.float32)
        m = R + 3
        x0, y0 = int(np.floor(cen[0])) - m, int(np.floor(cen[1])) - m
        H, W = P.shape
        if x0 < 0 or y0 < 0 or x0 + 2 * m + 1 > W or y0 + 2 * m + 1 > H:
            return cen
        win = P[y0:y0 + 2 * m + 1, x0:x0 + 2 * m + 1].astype(np.float32)
        resp = cv2.filter2D(win, -1, kern, borderType=cv2.BORDER_CONSTANT)
        iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
        if not (0 < ix < resp.shape[1] - 1 and 0 < iy < resp.shape[0] - 1):
            return cen

        def para(a, b, c):
            d = a - 2 * b + c
            return 0.0 if abs(d) < 1e-12 else 0.5 * (a - c) / d

        dx = para(resp[iy, ix - 1], resp[iy, ix], resp[iy, ix + 1])
        dy = para(resp[iy - 1, ix], resp[iy, ix], resp[iy + 1, ix])
        return np.array([x0 + ix + 0.5 + dx, y0 + iy + 0.5 + dy])

    def track(self, frames, init_uv, init_radius_px, ref_rgb=None) -> TrackResult:
        frames = np.asarray(frames)
        T = len(frames)
        uv0 = np.asarray(init_uv, dtype=float)
        r = float(init_radius_px)
        post = self._posterior(frames[0], uv0, r)
        centers = np.full((T, 2), np.nan)
        area = np.zeros(T)
        ncomp = np.zeros(T, dtype=int)
        expected = np.pi * r * r
        last, vel, misses = uv0.copy(), np.zeros(2), 0
        kernel = np.ones((3, 3), np.uint8)
        for t in range(T):
            P = post[self._index(frames[t])]
            mask = (P >= self.threshold).astype(np.uint8)
            if r >= 3:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.min_area_frac * expected]
            ncomp[t] = sum(stats[i, cv2.CC_STAT_AREA] >= self.dup_area_frac * expected for i in cand)
            if not cand:
                misses += 1
                continue
            pred = last + vel * (misses + 1)
            c = np.array([cents[i] + 0.5 for i in cand])  # pixel-center convention
            d = np.linalg.norm(c - pred, axis=1)
            gate = max(6 * r, 3 * np.linalg.norm(vel) + 4 * r)
            if misses >= self.lost_after:
                # re-acquire anywhere (e.g. after an occluder): best area match
                a = np.array([stats[i, cv2.CC_STAT_AREA] for i in cand], dtype=float)
                j = int(np.argmin(np.abs(np.log(a / expected))))
            else:
                j = int(np.argmin(d))
                if d[j] > gate:
                    misses += 1
                    continue
            i = cand[j]
            ys, xs = np.nonzero(lab == i)
            w = P[ys, xs]
            cen = np.array([np.sum(w * (xs + 0.5)), np.sum(w * (ys + 0.5))]) / max(np.sum(w), 1e-9)
            r_est = max(np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi), 0.8 * r, 1.0)
            cen = self._disc_refine(P, cen, r_est)
            if misses == 0 and np.all(np.isfinite(last)):
                vel = 0.5 * vel + 0.5 * (cen - last) if t > 1 else cen - last
            else:
                vel = np.zeros(2)
            centers[t] = cen
            area[t] = stats[i, cv2.CC_STAT_AREA]
            expected = 0.7 * expected + 0.3 * area[t]
            last, misses = cen, 0
        return TrackResult(centers=centers, area=area, n_components=ncomp, info={"tracker": self.name})
