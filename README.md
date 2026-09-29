# Advanced Robotics 403 — Course Project

Semester-long robotics project for the **Advanced Robotics** course, School
of Electrical & Computer Engineering, **University of Tehran** (Spring/Summer
1404 — 2025), built incrementally across five phases: from raw sensor/motion
modeling, through probabilistic localization, SLAM, and multi-model tracking,
up to autonomous path planning and navigation.

The robot is simulated in **Gazebo** (an Anki Vector-based model) and all
online modules are implemented as **ROS** nodes in Python, with NumPy/SciPy
used for the underlying estimation math and Jupyter notebooks for offline
data analysis.

**Team:** Ali Kheiri, Moein Rezaei, Mohammadreza Tavakoli
**Instructors:** Dr. Nasiri, Dr. Moradi

## Project phases

| # | Phase | Summary |
|---|---|---|
| 1 | [Sensor & Motion Model](01-sensor-motion-model) | Identify the robot's empirical motion and sensor (range-finder) models from logged data in Gazebo |
| 2 | [Particle Filter Localization](02-particle-filter-localization) | Monte Carlo Localization (MCL) against a known map, with kidnapped-robot recovery |
| 3 | [SLAM](03-slam) | Log-odds occupancy-grid SLAM, building the map online while tracking pose |
| 4 | [IMM Tracking](04-imm-tracking) | Interacting Multiple Model filter (CV / CA / CT) for robust motion tracking |
| 5 | [Path Planning & Navigation](05-path-planning-navigation) | A\* path planning with obstacle inflation and autonomous waypoint following |

Each phase folder is self-contained and includes its own `README.md`, source
code, a written report (PDF, and DOCX where available), and demo videos
where applicable.

## Tech stack

- **Simulation:** Gazebo, ROS (`rospy`, `tf`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`)
- **Estimation / math:** NumPy, SciPy
- **Analysis / visualization:** Jupyter, Matplotlib
- **Robot platform:** Anki Vector (simulated)

## Repository layout

```
.
├── 01-sensor-motion-model/          # Phase 1: sensor & motion modeling
├── 02-particle-filter-localization/ # Phase 2: MCL + kidnapping recovery
├── 03-slam/                         # Phase 3: occupancy-grid SLAM
├── 04-imm-tracking/                 # Phase 4: IMM motion tracking
└── 05-path-planning-navigation/     # Phase 5: A* planning + autonomous nav
```

## Notes

These projects assume a working ROS + Gazebo setup with the course's
`anki_description` simulation package; they are course deliverables and are
shared here for reference/portfolio purposes rather than as a turnkey
installable package.
