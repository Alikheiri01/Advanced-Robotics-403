# Phase 4 — Interacting Multiple Model (IMM) Tracking

Tracks the robot's motion using an Interacting Multiple Model (IMM) filter
that blends several motion-model hypotheses, so the tracker adapts
automatically to how the robot is actually moving instead of committing to a
single motion assumption.

## What this phase covers

- Three underlying Kalman-filter motion models, each specialized for a
  different kind of motion:
  - **CV** — Constant Velocity ("specialist", tuned to resist spurious
    acceleration)
  - **CA** — Constant Acceleration ("generalist")
  - **CT** — Coordinated Turn (for curved/turning motion)
- Model probability mixing/reparameterization between the three filters at
  every step (the IMM algorithm proper), so the combined estimate
  automatically weights whichever model best explains the recent motion.
- A simulation/test harness (`test_IMM_robot.py`, `move.py`) that drives a
  simulated or real robot trajectory and logs positions for the filter to
  track, plus a visualizer for comparing estimated vs. logged trajectories.

## Structure

| Path | Description |
|---|---|
| `src/IMM.py` | Core IMM filter: motion models, process noise, model mixing |
| `src/IMM_reparameterize.py` | Model-probability mixing/reparameterization step |
| `src/kalman_filter.py` | Underlying Kalman filter implementation used by each motion model |
| `src/move.py` | Drives the robot along a test trajectory |
| `src/test_IMM_robot.py` | End-to-end test harness running the IMM filter on a live/logged trajectory |
| `src/position_visualizer.py` | Plots estimated vs. ground-truth/logged trajectories |
| `data/robot_position_log.csv` | Logged robot positions used for filter evaluation |
| `media/IMMFilterTest.mp4` | Demo recording of the IMM filter tracking the robot |
| `Report.pdf` | Full write-up of methodology and results |

## Running

```bash
python3 src/test_IMM_robot.py
```

`src/IMM.py` can also be run standalone as a self-contained simulation
(it generates its own synthetic trajectory and plots the tracking result).
