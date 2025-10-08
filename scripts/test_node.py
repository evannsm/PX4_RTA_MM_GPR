from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.qos import (QoSProfile,
                       ReliabilityPolicy,
                       HistoryPolicy,
                       DurabilityPolicy) # Import ROS2 QoS policy modules
from mocap4r2_msgs.msg import FullState
from px4_msgs.msg import(
    OffboardControlMode, VehicleCommand, #Import basic PX4 ROS2-API messages for switching to offboard mode
    TrajectorySetpoint, VehicleRatesSetpoint, # Msgs for sending setpoints to the vehicle in various offboard modes
    VehicleStatus, RcChannels #Import PX4 ROS2-API messages for receiving vehicle state information
)


import time
import control
import immrax
import jax.numpy as jnp
import jax

# import numpy as np


class TestNode(Node):
    # print(f"jax version: {jax.__version__}, Numpy version: {np.__version__}")
    # print(f"control version: {control.__version__}")
    def __init__(self, sim):
        super().__init__('test_node')
        print("hi")
        print(f"fjskdlfjasdkl;fjalk;")