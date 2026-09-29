# Phase 1 — Sensor & Motion Model Identification

Establishes the motion and sensor models of the robot inside the **Gazebo**
simulator, which every later phase (localization, SLAM, tracking, planning)
builds on.

## What this phase covers

- Recording raw odometry (`cmd_vel`) and infrared range-sensor data while
  driving the robot through controlled maneuvers (straight-line motion,
  in-place rotation, clockwise/counter-clockwise sweeps).
- Deriving an empirical motion model (how commanded linear/angular velocity
  maps to actual robot displacement, including noise characteristics).
- Deriving an empirical sensor model (range-sensor behavior vs. ground-truth
  distance).
- Visualizing and analyzing the collected data to fit noise parameters used
  later by the particle filter (Phase 2) and SLAM (Phase 3).

## Structure

| Path | Description |
|---|---|
| `src/` | ROS scripts used to drive the robot and log data (`laser_data_recorder.py`, `velocity_data_recorder.py`, `velocity_angular_data_collection.py`) |
| `data/` | Recorded CSV logs (laser ranges, linear velocity, rotation/twist angle estimates) |
| `notebooks/` | Jupyter notebooks used to visualize and analyze the recorded data |
| `Report.pdf` | Full write-up of methodology, experiments, and results |

## Running

The scripts under `src/` are ROS nodes (`rospy`) meant to run against the
simulated robot in Gazebo alongside the rest of this course's ROS workspace.
Start the simulation, then run the desired recorder script, e.g.:

```bash
rosrun <package_name> laser_data_recorder.py
```

Data recorded this way is saved as CSV files matching the format found in
`data/`, which the notebooks in `notebooks/` then load and plot.
