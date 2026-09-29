# Phase 5 — Path Planning & Autonomous Navigation

Plans a collision-free path across the mapped environment and drives the
robot along it autonomously, closing the loop from mapping (SLAM) to
goal-directed navigation.

## What this phase covers

- **`map_maker.py`** — builds/loads the occupancy grid the planner searches
  over.
- **`path_planner.py`** — A\* search over the occupancy grid, inflating
  obstacles by the robot's radius (`walkable()`) so the returned path keeps
  a safe clearance from walls.
- **`auto_nav.py`** — a ROS node that follows the planned path with a
  proportional controller, using odometry feedback (position + yaw from the
  quaternion) to drive the robot toward each waypoint in sequence, while
  logging the executed trajectory for later comparison against the planned
  path.

## Structure

| Path | Description |
|---|---|
| `src/map_maker.py` | Builds the occupancy grid used for planning |
| `src/path_planner.py` | A\* path planner with obstacle inflation |
| `src/auto_nav.py` | ROS node that autonomously drives the robot along the planned path |
| `media/PathPlanningAndNavigation.mp4` | Demo recording of autonomous navigation |
| `Report.pdf` / `Report.docx` | Full write-up of methodology and results |

## Running

```bash
# Plan a path (uses map_maker.py internally)
python3 src/path_planner.py

# Autonomously navigate the simulated robot along it
rosrun <package_name> auto_nav.py
```
