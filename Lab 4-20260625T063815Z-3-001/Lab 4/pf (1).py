"""
pf.py — Particle filter localization node for the Turtlebot4
        in the beacons world.

*** STUDENT VERSION ***

In this lab you fill in 10 TODOs, all inside the `ParticleFilter` class.
Each TODO corresponds to a specific equation from Part 1 of the lab
manual.

    __init__():
      TODO 1  -- (init): initialize particle cloud + uniform weights

    predict():
      TODO 2  -- Eq. 1: sample new particles via motion model
      TODO 3  -- add process noise per particle

    update():
      TODO 4  -- compute per-particle innovation against landmark
      TODO 5  -- Eq. 3: compute the Gaussian likelihood per particle
      TODO 6  -- combine likelihoods from multiple detections per scan
      TODO 7  -- Eq. 4: normalize weights to sum to 1

    resample():
      TODO 8  -- Eq. 5: systematic resampling
      TODO 9  -- reset weights to 1/M after resampling

    estimate():
      TODO 10 -- weighted mean of the cloud (circular_mean for theta)

Subscribes:
    /odom   (nav_msgs/Odometry) -- wheel odometry, drives PREDICT
    /scan   (sensor_msgs/LaserScan) -- LIDAR, drives UPDATE

Publishes:
    /pf_pose      (nav_msgs/Odometry)             -- PF mean estimate
    /pf_particles (geometry_msgs/PoseArray)       -- particle cloud
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose


# ============================================================================
#                       KNOWN LANDMARK MAP
# ============================================================================
LANDMARKS = np.array([
    [ 3.0,  0.0],     # red beacon
    [ 0.0,  3.0],     # green beacon
    [-3.0,  1.5],     # blue beacon
])


# ============================================================================
#                       NOISE PARAMETERS  (same as Lab 3)
# ============================================================================
SIGMA_V       = 0.05    # m/s   -- linear velocity noise
SIGMA_W       = 0.05    # rad/s -- angular velocity noise
SIGMA_RANGE   = 0.80    # m     -- LIDAR range noise
SIGMA_BEARING = 0.45    # rad   -- LIDAR bearing noise

MAHALANOBIS_GATE = 4.0   # chi-squared 99% confidence, 2 DOF

# Same +90 deg LIDAR-frame offset as Lab 3.
LIDAR_BEARING_OFFSET = math.pi / 2


# ============================================================================
#                       PF PARAMETERS
# ============================================================================
N_PARTICLES = 50       # M, number of particles
INIT_XY_SPREAD = 1.0     # m, half-width of initial uniform spread
INIT_THETA_SPREAD = 0.2  # rad, initial uniform yaw spread


# ============================================================================
#                       HELPER FUNCTIONS  (provided)
# ============================================================================
def wrap_to_pi(angle):
    """Wrap an angle (scalar or numpy array) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def circular_mean(angles, weights=None):
    """Weighted circular mean of an array of angles.

    Averages on the unit circle: sum the unit vectors, then take atan2
    of the result. Use this instead of np.mean(angles), which breaks
    near the +/- pi wrap.
    """
    if weights is None:
        weights = np.ones_like(angles) / len(angles)
    sin_sum = np.sum(weights * np.sin(angles))
    cos_sum = np.sum(weights * np.cos(angles))
    return math.atan2(sin_sum, cos_sum)


def yaw_from_quaternion(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def motion_model(state, v, omega, dt):
    """Unicycle motion model -- identical to Lab 2/3."""
    x, y, theta = state[0], state[1], state[2]
    if abs(omega) < 1e-6:
        new_x = x + v * dt * math.cos(theta)
        new_y = y + v * dt * math.sin(theta)
        new_theta = theta
    else:
        r = v / omega
        theta_new = theta + omega * dt
        new_x = x + r * (-math.sin(theta) + math.sin(theta_new))
        new_y = y + r * ( math.cos(theta) - math.cos(theta_new))
        new_theta = theta_new
    return np.array([new_x, new_y, wrap_to_pi(new_theta)])


def measurement_model(state, landmark_xy):
    """Range-bearing measurement model -- identical to Lab 3."""
    x, y, theta = state[0], state[1], state[2]
    lx, ly = landmark_xy
    dx = lx - x
    dy = ly - y
    return np.array([math.hypot(dx, dy),
                     wrap_to_pi(math.atan2(dy, dx) - theta)])


# ============================================================================
#                       THE PARTICLE FILTER
# ============================================================================
#
# A note on angles before you start:
#   Particles carry a yaw and detections carry a bearing. Any time you
#   take a difference of two angles -- e.g. (z - z_hat) -- wrap the
#   angular component to [-pi, pi] BEFORE using it. Use `wrap_to_pi`.
#   For averaging angles, use `circular_mean`, never np.mean(angles).
# ============================================================================
class ParticleFilter:
    def __init__(self):
        # Process noise std-devs applied during predict.
        self.sigma_v = SIGMA_V
        self.sigma_w = SIGMA_W

        # Measurement noise covariance for the Gaussian likelihood.
        self.R = np.diag([SIGMA_RANGE ** 2, SIGMA_BEARING ** 2])
        self.R_inv = np.linalg.inv(self.R)
        self.R_det = np.linalg.det(self.R)

        # ----------------------------------------------------------------
        # TODO 1 -- initialize the particle cloud and uniform weights.
        #
        # `self.particles` should be a numpy array of shape (M, 3) where
        # M = N_PARTICLES. Each row is a particle's [x, y, theta].
        #
        #   - x, y: sample uniformly in [-INIT_XY_SPREAD, INIT_XY_SPREAD]
        #   - theta: sample uniformly in [-INIT_THETA_SPREAD,
        #                                  INIT_THETA_SPREAD]
        #
        # `self.weights` should be a numpy array of shape (M,) with every
        # entry equal to 1/M.
        # ----------------------------------------------------------------
        self.particles = np.zeros((N_PARTICLES, 3))
        self.particles[:, 0] = np.random.uniform(-INIT_XY_SPREAD, INIT_XY_SPREAD, N_PARTICLES)
        self.particles[:, 1] = np.random.uniform(-INIT_XY_SPREAD, INIT_XY_SPREAD, N_PARTICLES)
        self.particles[:, 2] = np.random.uniform(-INIT_THETA_SPREAD, INIT_THETA_SPREAD, N_PARTICLES)
        self.weights = np.ones(N_PARTICLES) / N_PARTICLES

    # =================================================================== #
    #                           PREDICT STEP
    # =================================================================== #
    def predict(self, v, omega, dt):
        # ----------------------------------------------------------------
        # TODO 2 -- Eq. 1: sample new particles via motion model.
        # TODO 3 -- add process noise per particle.
        #
        # For each particle m = 0, ..., M-1:
        #   - sample a noisy control:
        #       v_m = v + N(0, sigma_v)
        #       w_m = omega + N(0, sigma_w)
        #   - push particle m through motion_model(particle_m, v_m, w_m, dt)
        #   - overwrite particle m with the result.
        #
        # The process noise (TODO 3) is what gives the cloud its spread.
        # Without it, every particle would move identically and the
        # filter would collapse.
        # ----------------------------------------------------------------
        for m in range(N_PARTICLES):
            v_m = v + np.random.normal(0.0, self.sigma_v)
            w_m = omega + np.random.normal(0.0, self.sigma_w)
            self.particles[m] = motion_model(self.particles[m], v_m, w_m, dt)

    # =================================================================== #
    #                            UPDATE STEP
    # =================================================================== #
    def update(self, detections):
        """Re-weight particles using all detections in this scan.

        `detections` is a list of (range, bearing) tuples already in the
        robot/base frame (cluster_scan does the +pi/2 offset).
        """
        if not detections:
            return

        # Per-particle log-likelihood, summed across detections.
        # We accumulate in log space so multiplying many likelihoods
        # doesn't underflow to zero.
        log_w = np.zeros(N_PARTICLES)

        for (r, b) in detections:
            z = np.array([r, b])

            # Data association: pick the landmark whose predicted
            # measurement is closest to z under a covariance that
            # accounts for BOTH measurement noise (R) AND the spread
            # of the particle cloud's predicted measurements.
            # (Provided -- do not modify.)
            best_idx, best_md2 = -1, float('inf')
            for i, lm in enumerate(LANDMARKS):
                Z_pred = np.zeros((N_PARTICLES, 2))
                for m in range(N_PARTICLES):
                    Z_pred[m] = measurement_model(self.particles[m], lm)
                z_hat = np.array([
                    np.sum(self.weights * Z_pred[:, 0]),
                    circular_mean(Z_pred[:, 1], self.weights),
                ])
                dZ = Z_pred - z_hat
                dZ[:, 1] = wrap_to_pi(dZ[:, 1])
                S = (self.weights[:, None] * dZ).T @ dZ + self.R
                innov = z - z_hat
                innov[1] = wrap_to_pi(innov[1])
                try:
                    md2 = float(innov @ np.linalg.inv(S) @ innov)
                except np.linalg.LinAlgError:
                    md2 = float('inf')
                if md2 < best_md2:
                    best_md2 = md2
                    best_idx = i
            if best_md2 > MAHALANOBIS_GATE:
                continue
            lm = LANDMARKS[best_idx]

            # ------------------------------------------------------------
            # TODO 4 -- compute per-particle innovation against this
            #           landmark.
            # TODO 5 -- Eq. 3: Gaussian likelihood per particle.
            # TODO 6 -- combine likelihoods across detections.
            #
            # For each particle m = 0, ..., M-1:
            #   z_hat_m = measurement_model(particle_m, lm)
            #   nu_m    = z - z_hat_m              <- TODO 4
            #   wrap nu_m[1] (bearing) to [-pi, pi]
            #
            # The Gaussian log-likelihood is:
            #   log p(z | x_m) = -0.5 * nu_m^T R^-1 nu_m  +  const
            # The constant cancels out during normalization, so you can
            # drop it.                              <- TODO 5
            #
            # ADD this log-likelihood to log_w[m].   <- TODO 6
            # (Adding in log space = multiplying likelihoods. Doing
            # this across detections combines them as if they were
            # independent measurements.)
            # ------------------------------------------------------------
            for m in range(N_PARTICLES):
                z_hat_m = measurement_model(self.particles[m], lm)
                nu_m = z - z_hat_m
                nu_m[1] = wrap_to_pi(nu_m[1])
                log_w[m] += -0.5 * (nu_m @ self.R_inv @ nu_m)

        # ----------------------------------------------------------------
        # TODO 7 -- Eq. 4: normalize weights so they sum to 1.
        #
        # Numerical trick: subtract max(log_w) from log_w before
        # exponentiating. This keeps the largest weight at 1.0 and
        # prevents underflow when all log-weights are very negative.
        #
        # Then:
        #   self.weights = exp(log_w)
        #   s = sum(self.weights)
        #   if s > 0: self.weights /= s
        #   else: reset to uniform (1/M)
        # ----------------------------------------------------------------
        log_w -= np.max(log_w)
        self.weights = np.exp(log_w)
        s = np.sum(self.weights)
        if s > 0:
            self.weights /= s
        else:
            self.weights = np.ones(N_PARTICLES) / N_PARTICLES

    # =================================================================== #
    #                          RESAMPLE STEP
    # =================================================================== #
    def resample(self):
        # ----------------------------------------------------------------
        # TODO 8 -- Eq. 5: systematic resampling.
        #
        # Pick one random number r in [0, 1/M). Then compute the
        # cumulative sum of weights. For each i = 0, ..., M-1:
        #   threshold_i = (i + r) / M     (with r picked once, not per i)
        #
        # Walk j up through the cumulative array; every time
        # threshold_i < cumulative[j], select particle j for output
        # position i and move to i+1. Otherwise advance j.
        #
        # This gives M new particles, picked with replacement
        # proportional to the weights, with much lower variance than
        # naive multinomial sampling.
        #
        # TODO 9 -- reset weights to 1/M after resampling.
        # ----------------------------------------------------------------
        M = N_PARTICLES
        positions = (np.arange(M) + np.random.uniform(0.0, 1.0)) / M
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0  # guard against floating-point round-off
        new_particles = np.zeros_like(self.particles)
        i, j = 0, 0
        while i < M:
            if positions[i] < cumulative[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
        self.particles = new_particles
        self.weights = np.ones(M) / M

    # =================================================================== #
    #                          MEAN ESTIMATE
    # =================================================================== #
    def estimate(self):
        # ----------------------------------------------------------------
        # TODO 10 -- weighted mean of the particle cloud.
        #
        # Return np.array([x_hat, y_hat, theta_hat]) where:
        #   x_hat    = sum_m  w_m * particle_m_x
        #   y_hat    = sum_m  w_m * particle_m_y
        #   theta_hat = circular_mean(particle_thetas, weights=self.weights)
        #
        # Do NOT use np.mean(thetas) or sum(w * theta) -- averaging
        # angles directly breaks near the +/- pi wrap.
        # ----------------------------------------------------------------
        x_hat = np.sum(self.weights * self.particles[:, 0])
        y_hat = np.sum(self.weights * self.particles[:, 1])
        theta_hat = circular_mean(self.particles[:, 2], weights=self.weights)
        return np.array([x_hat, y_hat, theta_hat])


# ============================================================================
#                       LIDAR CLUSTERING  (same as Lab 3, provided)
# ============================================================================
def cluster_scan(scan, max_gap=0.15, min_points=3, max_width=0.5):
    ranges = np.array(scan.ranges)
    n = len(ranges)
    if n == 0:
        return []
    angles = scan.angle_min + np.arange(n) * scan.angle_increment
    valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
    clusters, current, prev_xy = [], [], None
    for i in range(n):
        if not valid[i]:
            if len(current) >= min_points:
                clusters.append(current)
            current = []
            prev_xy = None
            continue
        r = ranges[i]
        a = angles[i]
        x = r * math.cos(a)
        y = r * math.sin(a)
        if prev_xy is None:
            current = [(x, y)]
        else:
            if math.hypot(x - prev_xy[0], y - prev_xy[1]) < max_gap:
                current.append((x, y))
            else:
                if len(current) >= min_points:
                    clusters.append(current)
                current = [(x, y)]
        prev_xy = (x, y)
    if len(current) >= min_points:
        clusters.append(current)
    detections = []
    for c in clusters:
        xs = np.array([p[0] for p in c])
        ys = np.array([p[1] for p in c])
        cx, cy = xs.mean(), ys.mean()
        width = math.hypot(xs.max() - xs.min(), ys.max() - ys.min())
        if width > max_width:
            continue
        bearing = wrap_to_pi(math.atan2(cy, cx) + LIDAR_BEARING_OFFSET)
        detections.append((math.hypot(cx, cy), bearing))
    return detections


# ============================================================================
#                       THE ROS NODE  (provided)
# ============================================================================
class PFNode(Node):
    def __init__(self):
        super().__init__('pf_node')
        self.pf = ParticleFilter()
        self.last_odom_time = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Odometry, '/odom',
                                 self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan',
                                 self.scan_callback, sensor_qos)
        self.pose_pub = self.create_publisher(Odometry, '/pf_pose', 10)
        self.particles_pub = self.create_publisher(PoseArray,
                                                   '/pf_particles', 10)
        self.get_logger().info('PF node started.')

    def odom_callback(self, msg):
        # Use header stamp; fall back to node clock if stamp is zero
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            now = self.get_clock().now().to_msg()
            t = now.sec + now.nanosec * 1e-9
        else:
            t = stamp.sec + stamp.nanosec * 1e-9

        if self.last_odom_time is None:
            self.last_odom_time = t
            return
        dt = t - self.last_odom_time
        self.last_odom_time = t
        if dt <= 0.0 or dt > 1.0:
            return

        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        self.pf.predict(v, omega, dt)
        self.publish_pose(stamp)
        self.publish_particles(stamp)

    def scan_callback(self, msg):
        detections = cluster_scan(msg)
        if not detections:
            return
        self.pf.update(detections)
        self.pf.resample()
        self.publish_pose(msg.header.stamp)
        self.publish_particles(msg.header.stamp)

    def publish_pose(self, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        mu = self.pf.estimate()
        msg.pose.pose.position.x = float(mu[0])
        msg.pose.pose.position.y = float(mu[1])
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(float(mu[2]))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        self.pose_pub.publish(msg)

    def publish_particles(self, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        for m in range(N_PARTICLES):
            p = Pose()
            p.position.x = float(self.pf.particles[m, 0])
            p.position.y = float(self.pf.particles[m, 1])
            p.position.z = 0.0
            qx, qy, qz, qw = quaternion_from_yaw(float(self.pf.particles[m, 2]))
            p.orientation.x = qx
            p.orientation.y = qy
            p.orientation.z = qz
            p.orientation.w = qw
            msg.poses.append(p)
        self.particles_pub.publish(msg)


def main():
    rclpy.init()
    node = PFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


main()
