# Particle Filter Project

Final project artifacts for Lie-group-aware multi-camera UAV tracking.

## Structure

- `code/lg_filter_comparison/` - Python implementation
- `paper/` - final reports, paper PDF, and LaTeX source
- `results/` - generated summaries, CSVs, and figures
- `hardware/` - hardware photos and calibration imagery

## Main Scripts

- `run_comparison.py` - runs the EKF/PF comparison across scenarios
- `pf_particle_sweep.py` - evaluates PF particle-count/runtime tradeoffs
- `ambiguous_candidate_case.py` - studies ambiguity under false detections
- `hardware_calibration_ablation.py` - tests calibration failure modes
- `make_gifs.py` and `make_ros_camera_feed_cases.py` - generate result animations

Run scripts from `project/code/lg_filter_comparison` so local imports resolve cleanly.
