# Phase 2 — Particle Filter Localization (Monte Carlo Localization)

Implements Monte Carlo Localization (MCL) with a particle filter to estimate
the robot's pose inside a known map, using the motion and sensor models
identified in [Phase 1](../01-sensor-motion-model), and evaluates the
filter's ability to recover from a **kidnapped-robot** scenario (a sudden,
un-modeled relocation of the robot).

## What this phase covers

- Particle-filter localization against a pre-built occupancy map
  (`maps/map.pgm` / `maps/binary_map.npy`), with adaptive particle count
  (particles are thinned out as the estimate converges) and a likelihood
  field sensor model.
- Two different strategies for detecting and recovering from a kidnapped
  robot, implemented separately in `src/kidnapping_recovery/`.
- A series of incremental filter versions (`src/older_versions/`) kept for
  reference, showing the filter's development from an early to the final
  implementation.
- Isolated unit/test scripts (`src/test_modules/`) used to validate the
  motion model, sensor (weight) model, resampling step, and particle
  initialization/publishing independently of the full filter.

## Structure

| Path | Description |
|---|---|
| `src/particle_filter.py` | Main particle filter node (final version) |
| `src/kidnapping_recovery/` | Two kidnapping-recovery strategies |
| `src/test_modules/` | Standalone tests for individual filter components |
| `src/older_versions/` | Development history of the filter (v2–v9) |
| `maps/` | Occupancy map used for localization (`map.pgm`, `map.yaml`, `binary_map.npy`) and a saved initial particle set |
| `media/` | Demo recordings of the filter converging and recovering from kidnapping |
| `Report.pdf` / `Report.docx` | Full write-up of methodology and results |
| `Assignment-Description.pdf` | Original assignment brief from the course |
| `HOW_TO_RUN.txt` | Original run instructions (ROS/Gazebo commands) |

## Running

This is a ROS package meant to run against the simulated robot in Gazebo. From
`HOW_TO_RUN.txt`:

```bash
# 1. Launch the simulated world
roslaunch anki_description start_world.launch

# 2. Load the map into the ROS map server
rosrun map_server map_server /path/to/anki_description/maps/map.yaml

# 3. Fix the static map -> odom transform
rosrun tf static_transform_publisher 0 0 0 0 0 0 map odom 100

# 4. Visualize (rviz): add Map, PoseArray, and RobotModel displays

# 5. Drive the robot manually (optional)
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/vector/cmd_vel

# 6. Run the particle filter
rosrun anki_description particle_filter.py
```

## Demo

See `media/particle_filter_test.mp4` for convergence, and
`media/kidnapping_test_method1.mp4` / `media/kidnapping_test_method2.mp4` for
the two kidnapping-recovery strategies in action.
