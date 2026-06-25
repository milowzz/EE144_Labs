"""
ukf.py — UKF localization node for the Turtlebot4 in the beacons world.

*** STUDENT VERSION ***

In this lab you fill in 10 TODOs, all inside `UKFLocalization.predict()`
and `UKFLocalization.update()`. Each TODO corresponds to a specific
equation from Part 1 of the lab manual.


Subscribes:
    /odom   (nav_msgs/Odometry) -- wheel odometry, drives PREDICT
    /scan   (sensor_msgs/LaserScan) -- LIDAR, drives UPDATE

Publishes:
    /ukf_pose (nav_msgs/Odometry) -- the UKF's pose estimate
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


# ============================================================================
#                       KNOWN LANDMARK MAP
# ============================================================================
LANDMARKS = np.array([
    [ 3.0,  0.0],     # red beacon
    [ 0.0,  3.0],     # green beacon
    [-3.0,  1.5],     # blue beacon
])


# ============================================================================
#                       NOISE PARAMETERS
# ============================================================================
SIGMA_V       = 0.05    # m/s   -- linear velocity noise
SIGMA_W       = 0.05    # rad/s -- angular velocity noise
SIGMA_RANGE   = 0.80    # m     -- LIDAR range noise
SIGMA_BEARING = 0.45    # rad   -- LIDAR bearing noise

MAHALANOBIS_GATE = 4.0   # chi-squared 99% confidence, 2 DOF

# The /scan frame is rotated +90 degrees relative to base_link.
# tf2_echo showed yaw = +1.571 rad, so convert scan-frame bearings
# into the robot/base frame before using them in the UKF measurement update.
LIDAR_BEARING_OFFSET = math.pi / 2


# ============================================================================
#                       UKF PARAMETERS
# ============================================================================
N_STATE = 3
ALPHA   = 1e-3
BETA    = 2.0
KAPPA   = 0.0
LAMBDA  = ALPHA**2 * (N_STATE + KAPPA) - N_STATE


# ============================================================================
#                       HELPER FUNCTIONS  (provided)
# ============================================================================
def wrap_to_pi(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quaternion(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def motion_model(state, v, omega, dt):
    """Push a single state vector through the unicycle motion model.

    Handles the straight-line case (|omega| ~ 0) separately.
    """
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
    """Return z = [range, bearing] of `landmark_xy` as seen from `state`.

    Bearing is in the robot's body frame and wrapped to [-pi, pi].
    """
    x, y, theta = state[0], state[1], state[2]
    lx, ly = landmark_xy
    dx = lx - x
    dy = ly - y
    return np.array([math.hypot(dx, dy),
                     wrap_to_pi(math.atan2(dy, dx) - theta)])


# ============================================================================
#                       THE UKF CLASS
# ============================================================================
#
# Reminder on angles: any time you take a *difference* of two angles --
# (sigma_i - mu) or (z - z_hat) -- wrap the angular component to [-pi, pi]
# BEFORE using it in further math. Use the `wrap_to_pi` helper.
# ============================================================================
class UKFLocalization:
    def __init__(self):
        # State and covariance
        self.mu = np.array([0.0, 0.0, 0.0])
        self.Sigma = np.diag([0.01, 0.01, 0.01])

        # Process noise Q (added to predicted covariance)
        self.Q = np.diag([SIGMA_V ** 2 * 0.1,
                          SIGMA_V ** 2 * 0.1,
                          SIGMA_W ** 2 * 0.1])

        # Measurement noise R for z = [range, bearing]
        self.R = np.diag([SIGMA_RANGE ** 2, SIGMA_BEARING ** 2])

        # Sigma-point weights
        self.weights_m, self.weights_c = self._compute_weights()

    # ---- provided: sigma-point weights ------------------------------- #
    def _compute_weights(self):
        n = N_STATE
        Wm = np.zeros(2 * n + 1)
        Wc = np.zeros(2 * n + 1)
        Wm[0] = LAMBDA / (n + LAMBDA)
        Wc[0] = LAMBDA / (n + LAMBDA) + (1 - ALPHA**2 + BETA)
        for i in range(1, 2 * n + 1):
            Wm[i] = 1.0 / (2 * (n + LAMBDA))
            Wc[i] = 1.0 / (2 * (n + LAMBDA))
        return Wm, Wc

    # ---- provided: sigma-point generation ---------------------------- #
    def _generate_sigma_points(self, mu, Sigma):
        n = N_STATE
        sigma_points = np.zeros((2 * n + 1, n))
        try:
            L = np.linalg.cholesky((n + LAMBDA) * Sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((n + LAMBDA) * (Sigma + 1e-9 * np.eye(n)))
        sigma_points[0] = mu
        for i in range(n):
            sigma_points[i + 1]     = mu + L[:, i]
            sigma_points[n + i + 1] = mu - L[:, i]
        for i in range(2 * n + 1):
            sigma_points[i, 2] = wrap_to_pi(sigma_points[i, 2])
        return sigma_points

    # =================================================================== #
    #                           PREDICT STEP
    # =================================================================== #
    def predict(self, v, omega, dt):
        n = N_STATE
        sigma_points = self._generate_sigma_points(self.mu, self.Sigma)

        # ----------------------------------------------------------------
        # TODO 1 -- Eq. 2: push each sigma point through motion_model().
        # Build an array `propagated` of shape (2n+1, n).
        # ----------------------------------------------------------------
        propagated = np.zeros_like(sigma_points)
        for i in range(2 * n + 1):
            propagated[i] = motion_model(sigma_points[i], v, omega, dt)

        # ----------------------------------------------------------------
        # TODO 2 -- Eq. 3a: predicted mean
        #     mu_pred = sum_i  Wm[i] * propagated[i]
        # Remember to wrap_to_pi(mu_pred[2]) after the sum.
        # ----------------------------------------------------------------
        mu_pred = np.zeros(n)
        for i in range(2 * n + 1):
            mu_pred += self.weights_m[i] * propagated[i]
        mu_pred[2] = wrap_to_pi(mu_pred[2])

        # ----------------------------------------------------------------
        # TODO 3 -- Eq. 3b: predicted covariance
        #     Sigma_pred = sum_i  Wc[i] * (diff_i)(diff_i)^T  +  Q
        # where diff_i = propagated[i] - mu_pred, with diff_i[2] wrapped
        # to [-pi, pi] before the outer product.
        # ----------------------------------------------------------------
        Sigma_pred = np.zeros((n, n))
        for i in range(2 * n + 1):
            diff = propagated[i] - mu_pred
            diff[2] = wrap_to_pi(diff[2])
            Sigma_pred += self.weights_c[i] * np.outer(diff, diff)
        Sigma_pred += self.Q

        self.mu = mu_pred
        self.Sigma = Sigma_pred

    # =================================================================== #
    #                            UPDATE STEP
    # =================================================================== #
    def update(self, z, landmark_xy):
        n = N_STATE
        sigma_points = self._generate_sigma_points(self.mu, self.Sigma)

        # ----------------------------------------------------------------
        # TODO 4 -- Eq. 4: push each sigma point through measurement_model().
        # Build Z_sigma of shape (2n+1, 2).  Each row is [range, bearing].
        # ----------------------------------------------------------------
        Z_sigma = np.zeros((2 * n + 1, 2))
        for i in range(2 * n + 1):
            Z_sigma[i] = measurement_model(sigma_points[i], landmark_xy)

        # ----------------------------------------------------------------
        # TODO 5 -- Eq. 5a: predicted measurement mean
        #     z_hat = sum_i  Wm[i] * Z_sigma[i]
        # Remember to wrap_to_pi(z_hat[1]) after the sum (it's a bearing).
        # ----------------------------------------------------------------
        z_hat = np.zeros(2)
        for i in range(2 * n + 1):
            z_hat += self.weights_m[i] * Z_sigma[i]
        z_hat[1] = wrap_to_pi(z_hat[1])

        # ----------------------------------------------------------------
        # TODO 6 -- Eq. 5b: innovation covariance
        #     S = sum_i  Wc[i] * (dz_i)(dz_i)^T  +  R
        # where dz_i = Z_sigma[i] - z_hat, with dz_i[1] wrapped to [-pi, pi].
        # ----------------------------------------------------------------
        S = np.zeros((2, 2))
        for i in range(2 * n + 1):
            dz = Z_sigma[i] - z_hat
            dz[1] = wrap_to_pi(dz[1])
            S += self.weights_c[i] * np.outer(dz, dz)
        S += self.R

        # ----------------------------------------------------------------
        # TODO 7 -- Eq. 5c: cross-covariance
        #     T = sum_i  Wc[i] * (dx_i)(dz_i)^T
        # where dx_i = sigma_points[i] - self.mu (wrap dx_i[2]: yaw)
        # and   dz_i = Z_sigma[i] - z_hat        (wrap dz_i[1]: bearing).
        # ----------------------------------------------------------------
        T = np.zeros((n, 2))
        for i in range(2 * n + 1):
            dx = sigma_points[i] - self.mu
            dx[2] = wrap_to_pi(dx[2])
            dz = Z_sigma[i] - z_hat
            dz[1] = wrap_to_pi(dz[1])
            T += self.weights_c[i] * np.outer(dx, dz)

        # ----------------------------------------------------------------
        # TODO 8 -- Eq. 5d: Kalman gain
        #     K = T S^{-1}
        # ----------------------------------------------------------------
        K = T @ np.linalg.inv(S)

        # ----------------------------------------------------------------
        # TODO 9 -- Eq. 5e: update the mean
        #     mu <- mu + K (z - z_hat)
        # Wrap the bearing of the innovation, and wrap self.mu[2] after.
        # ----------------------------------------------------------------
        innov = z - z_hat
        innov[1] = wrap_to_pi(innov[1])
        self.mu = self.mu + K @ innov
        self.mu[2] = wrap_to_pi(self.mu[2])

        # ----------------------------------------------------------------
        # TODO 10 -- Eq. 5f: update the covariance
        #     Sigma <- Sigma - K S K^T
        # ----------------------------------------------------------------
        self.Sigma = self.Sigma - K @ S @ K.T

    # ---- provided: Mahalanobis gating (mirrors update through Eq. 5b) -- #
    def mahalanobis_distance_sq(self, z, landmark_xy):
        n = N_STATE
        sigma_points = self._generate_sigma_points(self.mu, self.Sigma)
        Z_sigma = np.zeros((2 * n + 1, 2))
        for i in range(2 * n + 1):
            Z_sigma[i] = measurement_model(sigma_points[i], landmark_xy)
        z_hat = np.zeros(2)
        for i in range(2 * n + 1):
            z_hat += self.weights_m[i] * Z_sigma[i]
        z_hat[1] = wrap_to_pi(z_hat[1])
        S = np.zeros((2, 2))
        for i in range(2 * n + 1):
            dz = Z_sigma[i] - z_hat
            dz[1] = wrap_to_pi(dz[1])
            S += self.weights_c[i] * np.outer(dz, dz)
        S += self.R
        innov = z - z_hat
        innov[1] = wrap_to_pi(innov[1])
        try:
            return float(innov @ np.linalg.inv(S) @ innov)
        except np.linalg.LinAlgError:
            return float('inf')


# ============================================================================
#                       LIDAR CLUSTERING  (provided)
# ============================================================================
def cluster_scan(scan, max_gap=0.15, min_points=3, max_width=0.5):
    """Group neighboring LIDAR points into beacon detections.

    Returns a list of (range, bearing) tuples in the robot/base frame.
    The +pi/2 offset accounts for the /scan frame being rotated 90deg
    relative to base_link on the Turtlebot4.
    """
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
class UKFNode(Node):
    def __init__(self):
        super().__init__('ukf_node')
        self.ukf = UKFLocalization()
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
        self.pose_pub = self.create_publisher(Odometry, '/ukf_pose', 10)
        self.get_logger().info('UKF node started.')

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

        self.ukf.predict(v, omega, dt)
        self.publish_pose(stamp)

    def scan_callback(self, msg):
        detections = cluster_scan(msg)
        if not detections:
            return

        for (r, b) in detections:
            z = np.array([r, b])
            best_idx, best_md2 = -1, float('inf')
            for i, lm in enumerate(LANDMARKS):
                md2 = self.ukf.mahalanobis_distance_sq(z, lm)
                if md2 < best_md2:
                    best_md2 = md2
                    best_idx = i
            if best_md2 > MAHALANOBIS_GATE:
                continue
            self.ukf.update(z, LANDMARKS[best_idx])

        self.publish_pose(msg.header.stamp)

    def publish_pose(self, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        mu = self.ukf.mu
        msg.pose.pose.position.x = float(mu[0])
        msg.pose.pose.position.y = float(mu[1])
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(float(mu[2]))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        Sigma = self.ukf.Sigma
        cov = [0.0] * 36
        cov[0]  = float(Sigma[0, 0])
        cov[1]  = float(Sigma[0, 1])
        cov[5]  = float(Sigma[0, 2])
        cov[6]  = float(Sigma[1, 0])
        cov[7]  = float(Sigma[1, 1])
        cov[11] = float(Sigma[1, 2])
        cov[30] = float(Sigma[2, 0])
        cov[31] = float(Sigma[2, 1])
        cov[35] = float(Sigma[2, 2])
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)


def main():
    rclpy.init()
    node = UKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


main()
