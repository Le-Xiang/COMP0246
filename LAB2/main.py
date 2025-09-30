import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import numpy as np
from numpy.typing import NDArray
from transform_helpers.utils import rotmat2q  # Import the rotmat2q function
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy


# Modified DH Params for the Franka FR3 robot arm
a_list = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d_list = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
alpha_list = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
theta_list = [0] * len(alpha_list)

DH_PARAMS = np.array([a_list, d_list, alpha_list, theta_list]).T

BASE_FRAME = "base"
FRAMES = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8"]


# def get_transform_n_to_n_minus_one(n: int, theta: float) -> NDArray:
def get_transform_n_to_n_minus_one(n: int, theta:float) -> NDArray:

    # transform_matrix = np.zeros((4,4))

    a = a_list[n]
    d = d_list[n]
    # alpha = alpha_list[n-1]
    alpha = alpha_list[n]
    theta = theta_list[n] + theta  # Add the current joint angle to the base value
    # Construct the transformation matrix using Modified DH Parameters
    transform_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],  # First row (rotation and translation)
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],  # Second row
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],  # Third row
        [0, 0, 0, 1]  # Fourth row (homogeneous coordinate)
    ])

    
    return transform_matrix

class ForwardKinematicCalculator(Node):

    def __init__(self):
        super().__init__('fk_calculator')

        # Define static transformation broadcasters for publishing static transformations from base frame to world coordinate system
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Create dynamic transformation broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)

        # Robot prefixes
        self.prefix = "my_robot/"

        # Publish static transformations of the base frame
        self.publish_base_transform()
        self.publish_base_l0_transform()
        # self.publish_l7_l8_transform()

         # Define a QoS profile (for example, reliable with transient local history)
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,  # Keep the last n messages
            depth=10,  # Queue size of 10 messages
            reliability=QoSReliabilityPolicy.RELIABLE  # Reliable communication
        )
        # Subscriber to the joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',  # Assuming the topic is /joint_states
            self.joint_state_callback,
            qos_profile  # QoS profile，
        )

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Prefix for frame names (robot specific)
        self.prefix = "my_robot/"

    from tf2_ros import StaticTransformBroadcaster

    def publish_base_transform(self):
        # Publish static transformations of base to world frames
        base_transform = TransformStamped()
        base_transform.header.stamp = self.get_clock().now().to_msg()
        base_transform.header.frame_id = "world"  # You can change to another frame of reference
        base_transform.child_frame_id = "my_robot/base"  # This is the basic frame

        # Setting translations and rotations (in this case unit transformations, indicating no offsets)
        base_transform.transform.translation.x = 0.0
        base_transform.transform.translation.y = 0.0
        base_transform.transform.translation.z = 0.0

        base_transform.transform.rotation.x = 0.0
        base_transform.transform.rotation.y = 0.0
        base_transform.transform.rotation.z = 0.0
        base_transform.transform.rotation.w = 1.0

        # Publish static transformations
        self.static_broadcaster.sendTransform(base_transform)
        return base_transform
    
    def publish_base_l0_transform(self):
        base_transform = TransformStamped()
        base_transform.header.stamp = self.get_clock().now().to_msg()
        base_transform.header.frame_id = "my_robot/base"  # Change to the parent frame you want
        base_transform.child_frame_id = f"{self.prefix}fr3_link0"  # It's still the base frame

        # Setting up translations and rotations
        base_transform.transform.translation.x = 0.0
        base_transform.transform.translation.y = 0.0
        base_transform.transform.translation.z = 0.0

        base_transform.transform.rotation.x = 0.0
        base_transform.transform.rotation.y = 0.0
        base_transform.transform.rotation.z = 0.0
        base_transform.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(base_transform)
        return base_transform

    def joint_state_callback(self, msg: JointState):
        # This method is called when a new JointState message is received
        # self.publish_transforms(msg)
        self.publish_cum_transforms(msg)

    def publish_cum_transforms(self, msg: JointState):
        cumulative_transform = np.eye(4)  # Initial transform from base
        eye = np.eye(4)
        cumulative_transform = cumulative_transform @ eye # fr3_link0 - base
        for i, frame in enumerate(FRAMES):
            if frame == "fr3_link0":
                continue
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = f"{self.prefix}{BASE_FRAME}"
            t.child_frame_id = f"{self.prefix}{frame}"

            theta = 0

            if i-1 < len(msg.position):
                theta = msg.position[i-1]

            transform = get_transform_n_to_n_minus_one(i-1, theta)
            cumulative_transform = cumulative_transform @ transform

            quat = rotmat2q(cumulative_transform[:3, :3])

            t.transform.translation.x = cumulative_transform[0, 3]
            t.transform.translation.y = cumulative_transform[1, 3]
            t.transform.translation.z = cumulative_transform[2, 3]

            t.transform.rotation.x = quat.x
            t.transform.rotation.y = quat.y
            t.transform.rotation.z = quat.z
            t.transform.rotation.w = quat.w

            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)

    fk_calculator = ForwardKinematicCalculator()

    rclpy.spin(fk_calculator)

    # Destroy the node explicitly
    fk_calculator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

