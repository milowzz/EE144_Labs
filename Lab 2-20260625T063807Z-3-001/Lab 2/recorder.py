"""
recorder.py — records the robot's trajectory while it drives.

Subscribes to:
  /sim_ground_truth_pose  (Gazebo ground truth)
  /odom                   (wheel odometry, drifts)
  /ekf_pose               (your EKF estimate -- see TODO blocks below)

On Ctrl+C, saves a CSV and a plot of all subscribed sources.

USAGE
-----
Terminal 1: launch Gazebo (bash ~/workspace/lab2_setup.sh, then ros2 launch ...)
Terminal 2: python3 recorder.py
Terminal 3: python3 ekf.py                  (your EKF node)
Terminal 4: python3 circle_driver.py        (drive the robot)

When the circle driver finishes, press Ctrl+C in this recorder terminal.
You'll get trajectory_data.csv and trajectory_plot.png in the current folder.
"""

import os
import csv
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Recorder(Node):
    def __init__(self):
        super().__init__('recorder')

        # Storage for samples. Each entry is (t, x, y).
        self.truth_data = []
        self.odom_data  = []

        # ====================================================================
        # TODO (students): uncomment the line below to also store EKF samples.
        # ====================================================================
        self.ekf_data = []

        self.t0 = time.time()

        # Best-effort QoS for ground truth (Gazebo publishes this way)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- Ground truth subscription ----
        self.create_subscription(
            Odometry, '/sim_ground_truth_pose',
            self.truth_callback, sensor_qos)

        # ---- Odometry subscription ----
        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        # ====================================================================
        # TODO (students): uncomment the EKF subscription below.
        # Your EKF node (ekf.py) should publish on the topic '/ekf_pose'
        # using the message type nav_msgs/Odometry. When that's done,
        # uncomment these three lines so the recorder also collects your
        # EKF estimate.
        # ====================================================================
        self.create_subscription(
            Odometry, '/ekf_pose',
            self.ekf_callback, 10)

        self.get_logger().info(
            'Recorder started. Subscribing to ground truth + odom.')
        self.get_logger().info(
            'Press Ctrl+C to stop and save the plot.')

    def truth_callback(self, msg):
        t = time.time() - self.t0
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.truth_data.append((t, x, y))

    def odom_callback(self, msg):
        t = time.time() - self.t0
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.odom_data.append((t, x, y))

    # ========================================================================
    # TODO (students): uncomment the EKF callback below.
    # ========================================================================
    def ekf_callback(self, msg):
        t = time.time() - self.t0
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ekf_data.append((t, x, y))

    def save_outputs(self):
        out_dir = os.getcwd()
        csv_path  = os.path.join(out_dir, 'trajectory_data.csv')
        plot_path = os.path.join(out_dir, 'trajectory_plot.png')

        # ---- Save CSV ----
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 't', 'x', 'y'])
            for (t, x, y) in self.truth_data:
                writer.writerow(['truth', f'{t:.3f}', f'{x:.4f}', f'{y:.4f}'])
            for (t, x, y) in self.odom_data:
                writer.writerow(['odom',  f'{t:.3f}', f'{x:.4f}', f'{y:.4f}'])

            # ================================================================
            # TODO (students): uncomment to also save EKF samples to CSV.
            # ================================================================
            for (t, x, y) in self.ekf_data:
                writer.writerow(['ekf', f'{t:.3f}', f'{x:.4f}', f'{y:.4f}'])

        print(f'Saved CSV to:  {csv_path}')
        print(f'  truth points: {len(self.truth_data)}, '
              f'odom points: {len(self.odom_data)}')
        # ====================================================================
        # TODO (students): uncomment to also print EKF count.
        # ====================================================================
        print(f'  ekf points:   {len(self.ekf_data)}')

        # ---- Save plot ----
        fig, ax = plt.subplots(figsize=(9, 9))

        if self.truth_data:
            xs = [d[1] for d in self.truth_data]
            ys = [d[2] for d in self.truth_data]
            ax.plot(xs, ys, 'g-', linewidth=2.5,
                    label=f'Ground truth ({len(self.truth_data)} pts)')

        if self.odom_data:
            xs = [d[1] for d in self.odom_data]
            ys = [d[2] for d in self.odom_data]
            ax.plot(xs, ys, 'r--', linewidth=1.5,
                    label=f'Odometry ({len(self.odom_data)} pts)')

        # ====================================================================
        # TODO (students): uncomment the EKF plotting block.
        # This draws the blue EKF line on top of truth (green) and odom (red).
        # ====================================================================
        if self.ekf_data:
            xs = [d[1] for d in self.ekf_data]
            ys = [d[2] for d in self.ekf_data]
            ax.plot(xs, ys, 'b-', linewidth=1.5,
                    label=f'EKF estimate ({len(self.ekf_data)} pts)')

        # Mark known beacon positions
        beacons_x = [3.0,  0.0, -3.0]
        beacons_y = [0.0,  3.0,  1.5]
        ax.plot(beacons_x, beacons_y, 'k*', markersize=15,
                label='Known beacons')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Robot trajectory: ground truth vs. odometry vs. EKF')
        ax.legend(loc='best')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=110)
        plt.close()
        print(f'Saved plot to: {plot_path}')


def main():
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nCtrl+C received, saving outputs...')
    finally:
        node.save_outputs()
        node.destroy_node()
        rclpy.shutdown()


main()