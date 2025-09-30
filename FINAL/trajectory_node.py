import numpy as np
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from transform_helpers.utils import rotmat2q

class YoubotTrajectoryFollower(Node):
    """
    This class represents a ROS2 node responsible for:
    - Following a given joint trajectory of the robot.
    - Publishing the end-effector pose as a transform for tracking purposes.
    """
    def __init__(self):
        super().__init__('youbot_trajectory_follower')

        # Initialise the subscriber
        self.subscription = self.create_subscription(
            JointTrajectory,
            '/EffortJointInterface_trajectory_controller/command',
            self.joint_trajectory_callback,
            5)
        self.subscription # avoid unused variable warning
        
        self.get_logger().info("Trajectory follower node initialised")
        # Initialise the kinematics and transform broadcaster
        self.kdl_kinematics = YoubotKinematicStudent()
        self.tf_broadcaster = TransformBroadcaster(self)
        
    def joint_trajectory_callback(self, msg: JointTrajectory):
        """
        The trigger function: called when a new trajectory message is received.
        """
        self.get_logger().info("Received tracking message")
        self.follow_trajectory(msg)
    
    def follow_trajectory(self, trajectory: JointTrajectory):
        """
        Publishes the corresponding end-effector pose for each point in the trajectory and holds the final pose
        until interrupted.
        """
        prev_time = 0
        last_transform = None
        for i, point in enumerate(trajectory.points):
            # Extract the joint positions from the message
            joint_positions = np.array(point.positions)

            # Calculate the end-effector pose using forward kinematics
            end_effector_transform = self.kdl_kinematics.forward_kinematics(joint_positions.tolist())
            last_transform = end_effector_transform
            # Publish the transform for the current trajectory point
            self.public_transform(end_effector_transform)
            self.get_logger().info(f"Published Transform for Point {i+1}")

            # Sleep for the duration, synchronise with the trajectory timestamps
            # Normalise timestamps of track points to floating point format in seconds
            current_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            sleep_time = current_time - prev_time
            prev_time = current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Hold the last point of the trajectory until interrupted
        if last_transform is None:
            self.get_logger().info("No trajectory points received.")
            return
        self.get_logger().info("Trajectory completed. Holding Node at the last point.")

        try:
            while rclpy.ok():
                self.public_transform(last_transform)
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.get_logger().info("Exiting the trajectory holding loop.")
                    
        
    def public_transform(self, matrix):
        """
        Publishes a transform which represents the pose of the end-effector.
        """
        # Create a TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "end_effector"

        # Extract the translation and rotation from the transformation matrix
        transform.transform.translation.x = matrix[0, 3]
        transform.transform.translation.y = matrix[1, 3]
        transform.transform.translation.z = matrix[2, 3]
        # Convert the rotation matrix to quaternion by using the helper function
        q = rotmat2q(matrix[:3, :3])
        transform.transform.rotation.x = q.x
        transform.transform.rotation.y = q.y
        transform.transform.rotation.z = q.z
        transform.transform.rotation.w = q.w
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    follower = YoubotTrajectoryFollower()
    try:
        follower.get_logger().info("Trajectory Follower Node Running.")
        rclpy.spin(follower)
    except KeyboardInterrupt:
        follower.get_logger().info("Node interrupted by user.")
    finally:
        # Avoid calling shudown multiple times
        if rclpy.ok():
            follower.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

