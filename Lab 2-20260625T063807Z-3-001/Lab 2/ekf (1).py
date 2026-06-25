"""
ekf.py — EKF localization node for the Turtlebot4 in the beacons world.

Subscribes:
    /odom   (nav_msgs/Odometry) -- wheel odometry, drives PREDICT
    /scan   (sensor_msgs/LaserScan) -- LIDAR, drives UPDATE

Publishes:
    /ekf_pose (nav_msgs/Odometry) -- the EKF's pose estimate

WHAT YOU NEED TO DO
-------------------
There are two methods you must complete:

    EKFLocalization.predict(self, v, omega, dt)
    EKFLocalization.update(self, z, landmark_xy)

Each has TODOs and references to the equation numbers from the lab manual.
DO NOT modify the rest of this file -- the ROS plumbing, the LIDAR clustering,
the landmark map, and the publisher are all already set up for you.

USAGE
-----
Once you fill in the two methods, run (in a separate terminal from Gazebo):

    python3 ekf.py

Then run the recorder and the circle driver in their own terminals.
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
# These are the beacon positions in the beacons.sdf world.
# The EKF uses these as ground-truth references for the LIDAR update step.
# DO NOT EDIT unless you also edit the world file to match.
# ============================================================================
LANDMARKS = np.array([
    [ 3.0,  0.0],     # red beacon
    [ 0.0,  3.0],     # green beacon
    [-3.0,  1.5],     # blue beacon
])


# ============================================================================
#                       NOISE PARAMETERS
# These are the standard deviations for the EKF's process and measurement
# noise models. Tune these if the EKF behaves badly:
#   - Increase sigma_v / sigma_w if the filter is too confident in odometry
#   - Increase sigma_range / sigma_bearing if it's too jumpy on measurements
# ============================================================================
SIGMA_V       = 0.05    # m/s   -- linear velocity noise
SIGMA_W       = 0.05    # rad/s -- angular velocity noise
SIGMA_RANGE   = 0.10    # m     -- LIDAR range noise
SIGMA_BEARING = 0.05    # rad   -- LIDAR bearing noise

# Mahalanobis gate for data association: clusters whose squared distance
# to all known landmarks exceeds this are rejected as outliers.
MAHALANOBIS_GATE = 9.21   # chi-squared 99% confidence, 2 DOF


# ============================================================================
#                       HELPER FUNCTIONS
# ============================================================================
def wrap_to_pi(angle):
    """Wrap an angle to the range [-pi, pi]. Critical for bearing math."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quaternion(qx, qy, qz, qw):
    """Extract yaw (rotation around z) from a quaternion."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw):
    """Build a quaternion (x, y, z, w) from a yaw angle."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


# ============================================================================
#                       THE EKF CLASS
# This is where YOU write the math. Two methods to complete: predict, update.
# ============================================================================
class EKFLocalization:
    """
    Pure-math EKF localization. No ROS dependency in this class -- it just
    does linear algebra. The ROS node below calls these methods.
    """

    def __init__(self):
        # State: [x, y, theta], starting at origin.
        self.mu = np.array([0.0, 0.0, 0.0])

        # Covariance: small initial uncertainty (we know we start at origin).
        self.Sigma = np.diag([0.01, 0.01, 0.01])

        # Process noise covariance (in control space [v, omega])
        self.M = np.diag([SIGMA_V ** 2, SIGMA_W ** 2])

        # Measurement noise covariance (in measurement space [r, bearing])
        self.R = np.diag([SIGMA_RANGE ** 2, SIGMA_BEARING ** 2])

    # ------------------------------------------------------------------ #
    #  PREDICT STEP                                                       #
    #  Called whenever new odometry arrives.                              #
    #  Updates self.mu and self.Sigma using the motion model.             #
    # ------------------------------------------------------------------ #
    def predict(self, v, omega, dt):
        """
        Run the EKF predict step.

        Inputs:
          v     : linear velocity (m/s) from odometry
          omega : angular velocity (rad/s) from odometry
          dt    : time elapsed since last predict (seconds)

        Updates self.mu and self.Sigma in place.
        """
        x, y, theta = self.mu

        # ----------------------------------------------------------------
        # TODO 1: Update self.mu using the motion model (Eq. 1a or 1b).
        # ----------------------------------------------------------------
        if abs(omega) < 1e-6:
            # Eq. (1b) -- straight line
            new_x     = x + v * np.cos(theta) * dt
            new_y     = y + v * np.sin(theta) * dt
            new_theta = theta
        else:
            # Eq. (1a) -- circular arc
            theta_new = theta + omega * dt
            new_x     = x + (v / omega) * (-np.sin(theta) + np.sin(theta_new))
            new_y     = y + (v / omega) * ( np.cos(theta) - np.cos(theta_new))
            new_theta = theta_new

        self.mu = np.array([new_x, new_y, wrap_to_pi(new_theta)])

        # ----------------------------------------------------------------
        # TODO 2: Compute the Jacobians F_x (3x3) and F_u (3x2).
        # ----------------------------------------------------------------
        if abs(omega) < 1e-6:
            # Eq. (2c), (2d) -- straight line
            F_x = np.array([[1.0, 0.0, -v * np.sin(theta) * dt],
                            [0.0, 1.0,  v * np.cos(theta) * dt],
                            [0.0, 0.0,  1.0]])
            F_u = np.array([[np.cos(theta) * dt, 0.0],
                            [np.sin(theta) * dt, 0.0],
                            [0.0,                dt ]])
        else:
            # Eq. (2a), (2b) -- turning
            theta_new = theta + omega * dt
            F_x = np.array([[1.0, 0.0, (v/omega) * (-np.cos(theta) + np.cos(theta_new))],
                            [0.0, 1.0, (v/omega) * (-np.sin(theta) + np.sin(theta_new))],
                            [0.0, 0.0, 1.0]])
            F_u = np.array([
                [(-np.sin(theta) + np.sin(theta_new)) / omega,
                 (v/omega**2)*(np.sin(theta) - np.sin(theta_new)) + (v*dt/omega)*np.cos(theta_new)],
                [( np.cos(theta) - np.cos(theta_new)) / omega,
                 (v/omega**2)*(-np.cos(theta) + np.cos(theta_new)) + (v*dt/omega)*np.sin(theta_new)],
                [0.0, dt]
            ])

        # ----------------------------------------------------------------
        # TODO 3: Update self.Sigma using Eq. (2).
        # ----------------------------------------------------------------
        self.Sigma = F_x @ self.Sigma @ F_x.T + F_u @ self.M @ F_u.T

    # ------------------------------------------------------------------ #
    #  UPDATE STEP                                                        #
    #  Called once per detected landmark in each LIDAR scan.              #
    #  Corrects self.mu and shrinks self.Sigma.                           #
    # ------------------------------------------------------------------ #
    def update(self, z, landmark_xy):
        """
        Run the EKF update step for a single observed landmark.

        Inputs:
          z           : np.array([range, bearing]) from the LIDAR
          landmark_xy : (lx, ly) world position of the matched known landmark

        Updates self.mu and self.Sigma in place.
        """
        x, y, theta = self.mu
        lx, ly = landmark_xy

        # Helper quantities (used by Eq. 3 and Eq. 4)
        dx = lx - x
        dy = ly - y
        q  = dx * dx + dy * dy

        # ----------------------------------------------------------------
        # TODO 4: Compute the expected measurement z_hat (Eq. 3).
        # ----------------------------------------------------------------
        z_hat = np.array([np.sqrt(q),
                          wrap_to_pi(np.arctan2(dy, dx) - theta)])

        # ----------------------------------------------------------------
        # TODO 5: Build the 2x3 measurement Jacobian H (Eq. 4).
        # ----------------------------------------------------------------
        sqrt_q = np.sqrt(q)
        H = np.array([[-dx / sqrt_q, -dy / sqrt_q,  0.0],
                      [ dy / q,      -dx / q,      -1.0]])

        # ----------------------------------------------------------------
        # Innovation (residual): how wrong was our prediction?
        # IMPORTANT: the bearing component must be wrapped to [-pi, pi].
        # ----------------------------------------------------------------
        y_innov = z - z_hat
        y_innov[1] = wrap_to_pi(y_innov[1])

        # ----------------------------------------------------------------
        # TODO 6: Compute the innovation covariance S (Eq. 5a).
        # ----------------------------------------------------------------
        S = H @ self.Sigma @ H.T + self.R

        # ----------------------------------------------------------------
        # TODO 7: Compute the Kalman gain K (Eq. 5b).
        # ----------------------------------------------------------------
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        # ----------------------------------------------------------------
        # TODO 8: Update self.mu (Eq. 5c) and wrap its theta component.
        # ----------------------------------------------------------------
        self.mu = self.mu + K @ y_innov
        self.mu[2] = wrap_to_pi(self.mu[2])

        # ----------------------------------------------------------------
        # TODO 9: Update self.Sigma (Eq. 5d).
        # ----------------------------------------------------------------
        I = np.eye(3)
        self.Sigma = (I - K @ H) @ self.Sigma

    # ------------------------------------------------------------------ #
    #  HELPER for data association -- already implemented, don't change  #
    # ------------------------------------------------------------------ #
    def mahalanobis_distance_sq(self, z, landmark_xy):
        """Squared Mahalanobis distance between measurement z and the
        expected measurement of `landmark_xy` from the current pose."""
        x, y, theta = self.mu
        lx, ly = landmark_xy
        dx = lx - x
        dy = ly - y
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        z_hat = np.array([r, wrap_to_pi(np.arctan2(dy, dx) - theta)])
        H = np.array([
            [-dx / r, -dy / r,  0.0],
            [ dy / q, -dx / q, -1.0],
        ])
        S = H @ self.Sigma @ H.T + self.R
        innov = z - z_hat
        innov[1] = wrap_to_pi(innov[1])
        try:
            return float(innov @ np.linalg.inv(S) @ innov)
        except np.linalg.LinAlgError:
            return float('inf')


# ============================================================================
#                       LIDAR CLUSTERING
# Already implemented. Turns a LaserScan into a list of (range, bearing)
# detections by grouping consecutive nearby scan points.
# ============================================================================
def cluster_scan(scan, max_gap=0.15, min_points=3, max_width=0.5):
    """Naive angular clustering. Returns list of (range, bearing) centroids
    in the robot frame, filtering out clusters too wide to be a beacon."""
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
        r_c = math.hypot(cx, cy)
        b_c = math.atan2(cy, cx)
        detections.append((r_c, b_c))
    return detections


# ============================================================================
#                       THE ROS NODE
# Wires everything together: subscribers, the EKF, and the publisher.
# ============================================================================
class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        self.ekf = EKFLocalization()
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

        self.pose_pub = self.create_publisher(Odometry, '/ekf_pose', 10)

        self.get_logger().info('EKF node started.')
        self.get_logger().info(f'  Landmark map: {LANDMARKS.tolist()}')

    def odom_callback(self, msg):
        # Compute dt since the last odom message
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_odom_time is None:
            self.last_odom_time = t
            return
        dt = t - self.last_odom_time
        self.last_odom_time = t
        if dt <= 0.0 or dt > 1.0:
            return

        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        self.ekf.predict(v, omega, dt)
        self.publish_pose(msg.header.stamp)

    def scan_callback(self, msg):
        detections = cluster_scan(msg)
        if not detections:
            return

        # For each detected cluster, find the closest known landmark by
        # Mahalanobis distance and (if it's a good match) run an update.
        for (r, b) in detections:
            z = np.array([r, b])
            best_idx, best_md2 = -1, float('inf')
            for i, lm in enumerate(LANDMARKS):
                md2 = self.ekf.mahalanobis_distance_sq(z, lm)
                if md2 < best_md2:
                    best_md2 = md2
                    best_idx = i
            if best_md2 > MAHALANOBIS_GATE:
                continue
            self.ekf.update(z, LANDMARKS[best_idx])

        self.publish_pose(msg.header.stamp)

    def publish_pose(self, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        mu = self.ekf.mu
        msg.pose.pose.position.x = float(mu[0])
        msg.pose.pose.position.y = float(mu[1])
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(float(mu[2]))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Pack the (x, y, yaw) part of Sigma into the 6x6 covariance.
        Sigma = self.ekf.Sigma
        cov = [0.0] * 36
        cov[0]  = float(Sigma[0, 0])    # xx
        cov[1]  = float(Sigma[0, 1])    # xy
        cov[5]  = float(Sigma[0, 2])    # x-yaw
        cov[6]  = float(Sigma[1, 0])    # yx
        cov[7]  = float(Sigma[1, 1])    # yy
        cov[11] = float(Sigma[1, 2])    # y-yaw
        cov[30] = float(Sigma[2, 0])    # yaw-x
        cov[31] = float(Sigma[2, 1])    # yaw-y
        cov[35] = float(Sigma[2, 2])    # yaw-yaw
        msg.pose.covariance = cov

        self.pose_pub.publish(msg)


def main():
    rclpy.init()
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


main()