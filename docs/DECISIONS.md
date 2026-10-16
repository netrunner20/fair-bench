# What is decided and what is open

**Decided in the proposal** (the harness implements these as stated):

* V2V only; 2 s (48 frames) in, 4 s (96 frames) out; 1280x720, 24 fps; Kubric renders.
* Three motions (thrown, falling and bouncing, occluding) with determined futures;
  6 intervention groups x 3 variants = 18 clips.
* Color and background robustness: tracked trajectories compared frame by frame, mean
  of the three variant pairs.
* Camera-angle robustness: VIMS (triangulation from tracked centers, sphere-center
  correction, single-frame jump removal, 5-frame running median, frame 0 = last
  conditioning frame, moving cameras flagged), E_ceil from 0.3 px tracker noise, E_ref
  from a one-frame lag.
* Plausibility: Morpheus' dynamical and physical-invariance scores on tracked
  trajectories; generations need only be physically possible.
* Reference error in meters at the ball's depth; point-to-line-of-sight distance for
  the camera-angle group.
* How to combine the 6 group scores and the weights of the 3 scores: chosen after the
  scores are seen.

**Open** (harness default in brackets; change in `configs/`):

| Item | Options | Default |
| --- | --- | --- |
| Model list | see `configs/models.yaml` | none enabled |
| Samples per clip | 1 / 3 / 5 | 3 |
| Prompt | neutral text / empty / scene description without the intervened factor | "Continue this video. The camera stays still." |
| 0-1 mapping for color/background distance and reference error | VIMS-style exponential / raw mm only | exponential (raw also reported) |
| Group combination | mean / per-intervention mean / geometric / worst / median | all reported |
| Final weights | equal / robustness-heavy / geometric / robustness x plausibility | all reported |
| Lost ball | penalty / exclude | penalty = 10 one-frame lags |
| Videos flagged "camera moved" (the check needs background texture; plain walls give "unknown") | report only / exclude / re-register the frames | report only (count per model in the report) |
| Dynamical score | PINN (Morpheus) / exact least squares | PINN, both reported |
| Plausibility combination | mean / dynamical only / product | mean |
| Sub-pixel refinement of SAM 2.1 masks | match the method used for the VIMS validation | probability-weighted centroid |
| Which 3 of the 4 listed walls form the background group | grey, blue, near black, brick | dataset-defined |
