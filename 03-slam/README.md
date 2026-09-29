# Phase 3 — Occupancy-Grid SLAM

Builds a 2D occupancy-grid map of the environment from scratch while
simultaneously tracking the robot's pose, using a log-odds occupancy-grid
mapping approach fed by the robot's range sensor and odometry in Gazebo.

## What this phase covers

- Log-odds occupancy-grid mapping: each cell's occupancy probability is
  updated incrementally from laser hits/misses (`LOG_ODDS_OCC` /
  `LOG_ODDS_FREE`) and clamped to `[LOG_ODDS_MIN, LOG_ODDS_MAX]`.
- A minimum-hit-count threshold before a cell is considered stably occupied,
  reducing noise from spurious sensor readings.
- Periodic map smoothing via convolution to clean up the resulting grid.

## Structure

| Path | Description |
|---|---|
| `src/SLAM.py` | ROS node implementing the occupancy-grid SLAM pipeline |
| `media/SLAM.mp4` | Demo recording of the map being built in real time |
| `Report.pdf` | Full write-up of methodology and results |

## Running

`src/SLAM.py` is a ROS node (`rospy`) that subscribes to odometry and range
sensor topics and publishes the resulting occupancy grid, meant to run
against the simulated robot in Gazebo:

```bash
rosrun <package_name> SLAM.py
```
