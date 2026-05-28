# AAE 590 - Lie Group Methods for Estimation and Control

Course portfolio for **Lie Group Methods for Estimation and Control**. This repository presents solution notebooks, Kalman/EKF experiments, and a final project on Lie-group-aware multi-camera UAV tracking with particle filtering, EKF variants, hardware calibration studies, and result visualizations.

![Course](https://img.shields.io/badge/course-AAE%20590-1f6feb)
![Focus](https://img.shields.io/badge/focus-Lie%20groups%20%7C%20estimation%20%7C%20control-2ea44f)
![Tools](https://img.shields.io/badge/tools-Python%20%7C%20Jupyter%20%7C%20Modelica%20%7C%20RUMOCA-f97316)
![Status](https://img.shields.io/badge/status-curated%20course%20portfolio-8250df)

## Highlights

| Lie-group particle filter trajectory | Filter RMSE comparison |
|---|---|
| ![Trajectory top view](project/results/figures/trajectory_top_view.png) | ![RMSE comparison](project/results/figures/rmse_bar.png) |

| Hardware calibration ablation | PURT camera layout |
|---|---|
| ![Hardware calibration ablation](project/results/figures/hardware_calibration_ablation_rmse.png) | ![PURT camera layout](project/results/figures/hardware_purt_camera_layout.png) |

| Body-frame EKF | RUMOCA unicycle comparison |
|---|---|
| ![Body-frame EKF animation](coursework/media/body_frame_ekf_animation.gif) | ![RUMOCA unicycle overlay](solutions/ps04/results/problem5_rumoca_overlay.png) |

## What Is Inside

| Area | Contents | Why it is here |
|---|---|---|
| [Problem-set solutions](solutions) | PS04 solution notebooks, Modelica model, RUMOCA comparison script, generated overlay | Shows my implemented solution work without redistributing Purdue prompt PDFs. |
| [Coursework experiments](coursework) | Kalman filtering notebooks and EKF animations | Captures the estimation workflow built during the course. |
| [Final project](project) | Particle filter, EKF baselines, papers, hardware notes, plots, and calibration analysis | Main polished research-style deliverable for the course. |

## Problem-Set Solutions

The `solutions/` folder contains my own solution artifacts where they are available. Course prompt PDFs and assignment handouts are intentionally excluded.

- [PS04 solution workspace](solutions/ps04) - Jupyter notebooks for `SE(2)` and unicycle experiments
- [Modelica unicycle model](solutions/ps04/modelica/Unicycle.mo)
- [RUMOCA comparison script](solutions/ps04/modelica/problem5_rumoca_compare.py)
- [Generated RUMOCA overlay](solutions/ps04/results/problem5_rumoca_overlay.png)

## Coursework

The `coursework/` folder collects additional material from the local course workspace:

- Kalman filtering notebooks from in-class and homework experiments
- EKF visualizations in world-frame and body-frame coordinates

## Final Project

**Particle Filtering on Lie Groups for Multi-Camera UAV Tracking**

The project estimates a UAV's 3D trajectory from fixed calibrated camera measurements. The target state is position and velocity in `R^6`, while the camera geometry is represented with `SE(3)` transforms and pinhole projection. The study compares:

- Multiplicative/error-state EKF style baselines
- Iterated EKF
- Mahalanobis-gated EKF variants
- Bootstrap particle filter with robust multi-camera likelihoods
- Calibration-error ablations motivated by hardware testing

Key artifacts:

- [Final project report](project/paper/particle_filter_on_lie_groups_report.pdf)
- [AIAA-style project paper](project/paper/aiaa_lg_pf_uav_tracking.pdf)
- [Source code](project/code/lg_filter_comparison)
- [Generated project report](project/results/report.md)
- [Hardware calibration analysis](project/results/hardware_calibration_ablation.md)

## Results Snapshot

| Scenario | Best EKF-style RMSE | PF RMSE | Notes |
|---|---:|---:|---|
| Nominal | 0.153 m | 0.183 m | EKF variants are efficient when initialization/noise are benign. |
| High noise | 0.346 m | 0.446 m | PF remains stable but pays a variance/runtime cost. |
| Dropout and outliers | 0.248 m gated | 0.317 m | Gating is essential for EKF robustness. |
| Correct hardware calibration | - | 0.943 m | Simulated PURT run using corrected UP-camera intrinsics. |
| Wrong `camera0.yaml` | - | 6.654 m | Pixel-frame mismatch alone creates multi-meter error. |
| 15 deg, 0.35 m extrinsic error | - | 8.692 m | Marker-frame/optical-frame mismatch is sufficient to explain large raw errors. |

## Hardware And Visuals

| Active marker camera | Hypothesis pipeline |
|---|---|
| ![Active marker camera](project/results/figures/hardware_active_marker_camera.jpeg) | ![Hardware hypothesis pipeline](project/results/figures/hardware_hypothesis_pipeline.png) |

Animations and media are included for quick review:

- [Figure-eight tracking animation](project/results/figures/animated_zoomed_figure_eight.gif)
- [Hardware hypothesis story](project/results/figures/hardware_hypothesis_story.gif)
- [Hardware lab footage GIF](project/results/figures/hw_testing_lab_footage.gif)
- [PURT hardware testing clip](media/hw_testing_dev_day_purt.mp4)

## Repository Layout

```text
solutions/          Problem-set solution notebooks, code, and generated results
coursework/         Kalman filtering notebooks and EKF media
project/code/       Particle filter, EKF, projection, plotting, and hardware scripts
project/paper/      Final reports and paper source
project/results/    Result summaries, CSVs, figures, and generated notes
project/hardware/   Hardware photos and calibration imagery
media/              Short hardware media clips
```

## Run The Project Code

```bash
cd project/code/lg_filter_comparison
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../../../requirements.txt
python run_comparison.py --out-dir ../../results
python pf_particle_sweep.py --out-dir ../../results
python ambiguous_candidate_case.py --out-dir ../../results
```

Some hardware and ROS visualization scripts require ROS 2, `cv_bridge`, `rosbag2_py`, and camera-bag data that is not included in this repository.

To run the PS04 solution notebooks, open the files in `solutions/ps04/notebooks/` with Jupyter. The RUMOCA comparison also expects the optional `rumoca` package and a working Modelica/RUMOCA setup.

## Included And Excluded

Included:

- Problem-set solution notebooks, source code, and generated solution figures
- Kalman filtering notebooks and EKF visualizations from `Desktop/AAE590`
- Final particle-filter project code
- Final reports, result CSVs, plots, hardware images, and selected animations

Excluded:

- Course lecture slides, textbooks, problem statement PDFs, prompt handouts, and assignment sheets
- Python virtual environments, caches, LaTeX build artifacts, raw generated frame folders, and large duplicate exports
- Raw ROS bags and heavyweight local hardware datasets

## Academic Note

This is a personal course portfolio. The repository is intended to show my implementations, reports, experiments, and results, not to redistribute course source material.
