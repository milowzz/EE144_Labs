# EE144 — Probabilistic Robot Localization (TurtleBot 4 Lite)

Three progressive labs implementing and comparing Bayesian localization filters on a TurtleBot 4 Lite in ROS 2 / Gazebo. Each lab fuses noisy wheel odometry with LIDAR detections of three known beacons to estimate the robot's pose `(x, y, θ)`, and compares the result against Gazebo ground truth.

**Author:** Emilio Rivas
**Course:** EE144, UC Riverside
**TA:** Georgia Kouvoutsakis

## Overview

| Lab | Filter | Core Idea | Files |
|---|---|---|---|
| 2 | Extended Kalman Filter (EKF) | Linearizes nonlinear motion/measurement models with Jacobians at each step | `ekf.py`, `recorder.py` |
| 3 | Unscented Kalman Filter (UKF) | Propagates 7 sigma points through the *exact* nonlinear model — no Jacobians needed | `ukf.py` |
| 4 | Particle Filter (PF / SIR) | Represents belief as a cloud of weighted particles; resamples to survive non-Gaussian/multi-modal uncertainty | `pf.py` |

All three reuse the same simulated world: a TurtleBot 4 Lite driving a circular path in Gazebo, with three fixed beacons at known positions:

```
Beacon 1 (red):    ( 3.0,  0.0)
Beacon 2 (green):  ( 0.0,  3.0)
Beacon 3 (blue):   (-3.0,  1.5)
```

## Repository Structure

```
.
├── Lab2_EKF/
│   ├── ekf.py              # EKF predict/update implementation
│   ├── recorder.py         # Logs ground truth, odometry, and filter pose to CSV + plot
│   └── Lab2_EE144.pdf       # Lab report
├── Lab3_UKF/
│   ├── ukf.py               # UKF predict/update implementation (unscented transform)
│   └── EE144_lab3_report.pdf
├── Lab4_PF/
│   ├── pf.py                 # Particle filter: predict, update, resample, estimate
│   └── lab4_report_ee144.pdf
└── README.md
```

> Lab 3 and Lab 4 reuse `recorder.py` from Lab 2 unchanged — copy it into each lab's folder (or keep one shared copy) before running.

## Requirements

- ROS 2 (tested with the course-provided TurtleBot 4 Lite / Gazebo simulation setup)
- Python 3
- `numpy`
- `matplotlib`

## Running a Lab

Each lab follows the same four-terminal workflow:

```bash
# Terminal 1 — launch the simulation
bash ~/workspace/lab2_setup.sh        # or the equivalent setup script for the lab
ros2 launch <beacons_world_launch>

# Terminal 2 — start the recorder (logs ground truth / odom / filter estimate)
python3 recorder.py

# Terminal 3 — start the filter node
python3 ekf.py     # or ukf.py / pf.py, depending on the lab

# Terminal 4 — drive the robot in a circle
python3 circle_driver.py
```

When `circle_driver.py` finishes, `Ctrl+C` the recorder terminal to save `trajectory_data.csv` and `trajectory_plot.png` to the current directory.

To test robustness to odometry noise (Labs 3–4), switch the filter's odometry subscription from `/odom` to `/noisy_odom`, published by `noisy_odom_publisher`.

## Lab 2 — Extended Kalman Filter

Implements `EKFLocalization.predict()` and `.update()` in `ekf.py`:

- **Predict** (on `/odom`): propagates `μ = [x, y, θ]` through the unicycle motion model and grows covariance `Σ` via the Jacobian `F_x`/`F_u`.
- **Update** (on `/scan`): linearizes the range-bearing measurement model with Jacobian `H`, computes the Kalman gain, and corrects `μ`/`Σ` whenever a beacon is detected (with Mahalanobis-distance gating to reject outlier associations).

**Result:** the EKF estimate (blue) tracks ground truth (green) closely, snapping back into alignment after each beacon detection, while raw odometry (red dashed) drifts steadily off-course.

## Lab 3 — Unscented Kalman Filter

Implements `UKFLocalization.predict()` and `.update()` in `ukf.py`, reusing the same motion/measurement models from Lab 2 as black-box functions (no Jacobians):

- Builds 7 sigma points (`2n + 1`, `n = 3`) around the current mean via the unscented transform.
- Pushes each sigma point through the nonlinear motion model, then reconstructs a Gaussian (mean + covariance) from the transformed points for the predict step.
- Repeats the same sigma-point propagation through the measurement model for the update step, computing innovation covariance `S`, cross-covariance `T`, and Kalman gain `K = T·S⁻¹`.

**Results:**
- *Clean odometry:* UKF estimate nearly overlaps ground truth.
- *Noisy odometry* (`/noisy_odom`): odometry alone spirals badly off-course; the UKF still tracks the true circular path by leaning on beacon detections.
- *UKF vs. EKF, same noisy input:* the UKF tracked ground truth more tightly than the EKF, since the unscented transform captures nonlinearity to 3rd order vs. the EKF's 1st-order linearization.

## Lab 4 — Particle Filter (SIR)

Implements the `ParticleFilter` class in `pf.py`:

- **Predict:** samples each of `M` particles forward through the motion model with per-particle process noise.
- **Update:** scores each particle's likelihood against detected beacons (combining likelihoods when multiple beacons are seen in one scan) and normalizes weights.
- **Resample:** systematic resampling to fight particle degeneracy; weights reset to `1/M`.
- **Estimate:** weighted mean of the cloud (circular mean for `θ`).

**Results:**
- With `M = 500` particles, the filter converges from a spread-out initial cloud to the true pose within ~1–2 seconds, then tracks ground truth closely for the rest of the circular trajectory.
- Convergence speed depends on how many beacons are visible per scan — seeing 2–3 beacons at once collapses the cloud much faster than a single beacon.
- Dropping to `M = 50` particles produced a nearly identical trajectory — for this 3-dimensional state with 3 well-placed beacons, far fewer particles than 500 are needed to represent the posterior well.

### Filter Comparison (Lab 4 discussion)

| | UKF | Particle Filter |
|---|---|---|
| Belief representation | Single Gaussian (mean + covariance) | Cloud of weighted samples |
| Steady-state tracking | Tighter, smoother | Slightly jittery (resampling noise) |
| Convergence from a poor initial guess | Can diverge if the true state falls outside the linearization region | Robust — particles can represent large/multi-modal uncertainty |
| Best suited for | Good initial guess, unimodal belief | Poor initial guess or non-Gaussian uncertainty |
