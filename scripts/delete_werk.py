import sys
import os
import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy # Import ROS2 QoS policy modules

from px4_msgs.msg import OffboardControlMode, VehicleCommand #Import basic PX4 ROS2-API messages for switching to offboard mode
from px4_msgs.msg import TrajectorySetpoint, ActuatorMotors, VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleRatesSetpoint # Msgs for sending setpoints to the vehicle in various offboard modes
from px4_msgs.msg import VehicleOdometry, VehicleStatus, RcChannels #Import PX4 ROS2-API messages for receiving vehicle state information

import time
import traceback
from typing import Optional

import numpy as np  
import math as m
from scipy.spatial.transform import Rotation as R

from px4_rta_mm_gpr.utilities import test_function
from px4_rta_mm_gpr.jax_nr import NR_tracker_original#, dynamics, predict_output, get_jac_pred_u, fake_tracker, NR_tracker_flat, NR_tracker_linpred
from px4_rta_mm_gpr.utilities import sim_constants # Import simulation constants
from px4_rta_mm_gpr.jax_mm_rta import *

import jax
import jax.numpy as jnp
import immrax as irx
import control
from functools import partial
from Logger import Logger, LogType, VectorLogType, install_shutdown_logging


# Some configurations
jax.config.update("jax_enable_x64", True)
def jit (*args, **kwargs): # A simple wrapper for JAX's jit function to set the backend device
    device = 'cpu'
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)

GP_instantiation_values = jnp.array([[-2, 0.0], #make the second column all zeros
                                    [0, 0.0],
                                    [2, 0.0],
                                    [4, 0.0],
                                    [6, 0.0],
                                    [8, 0.0],
                                    [10, 0.0],
                                    [12, 0.0]]) # at heights of y in the first column, disturbance to the values in the second column
# add a time dimension at t=0 to the GP instantiation values for TVGPR instantiation
actual_disturbance_GP = TVGPR(jnp.hstack((jnp.zeros((GP_instantiation_values.shape[0], 1)), GP_instantiation_values)), 
                                       sigma_f = 5.0, 
                                       l=2.0, 
                                       sigma_n = 0.01,
                                       epsilon=0.1,
                                       discrete=False
                                       )





class OffboardControl(Node):
    def __init__(self, sim: bool) -> None:
        super().__init__('px4_rta_mm_gpr_node')
        test_function()
        # Initialize essential variables
        self.sim: bool = sim
        self.GRAVITY: float = 9.806 # m/s^2, gravitational acceleration

        if self.sim:
            print("Using simulator constants and functions")
            from px4_rta_mm_gpr.utilities import sim_constants # Import simulation constants
            self.MASS = sim_constants.MASS
            self.THRUST_CONSTANT = sim_constants.THRUST_CONSTANT #x500 gazebo simulation motor thrust constant
            self.MOTOR_VELOCITY_ARMED = sim_constants.MOTOR_VELOCITY_ARMED #x500 gazebo motor velocity when armed
            self.MAX_ROTOR_SPEED = sim_constants.MAX_ROTOR_SPEED #x500 gazebo simulation max rotor speed
            self.MOTOR_INPUT_SCALING = sim_constants.MOTOR_INPUT_SCALING #x500 gazebo simulation motor input scaling

        elif not self.sim:
            print("Using hardware constants and functions")
            #TODO: do the hardware version of the above here
            try:
                from px4_rta_mm_gpr.utilities import hardware_constants
                self.MASS = hardware_constants.MASS
            except ImportError:
                raise ImportError("Hardware not implemented yet.")


        # Logging related variables
        self.time_log = []
        self.x_log, self.y_log, self.z_log, self.yaw_log = [], [], [], []
        self.ctrl_comp_time_log = []
        # self.m0_log, self.m1_log, self.m2_log, self.m3_log = [], [], [], [] # direct actuator control logs
        # self.f_log, self.M_log = [], [] # force and moment logs
        self.throttle_log, self.roll_rate_log, self.pitch_rate_log, self.yaw_rate_log = [], [], [], [] # throttle and rate logs
        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                ])

##########################################################################################
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.actuator_motors_publisher = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
        self.vehicle_thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.vehicle_torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_subscriber_callback, qos_profile)
            
        self.offboard_mode_rc_switch_on: bool = True if self.sim else False   # RC switch related variables and subscriber
        print(f"RC switch mode: {'On' if self.offboard_mode_rc_switch_on else 'Off'}")
        self.MODE_CHANNEL: int = 5 # Channel for RC switch to control offboard mode (-1: position, 0: offboard, 1: land)
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" for position v offboard v land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_subscriber_callback, qos_profile
        )
        


    def rc_channel_subscriber_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print('In RC Channel Callback')
        flight_mode = rc_channels.channels[self.MODE_CHANNEL-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on: bool = True if flight_mode >= 0.75 else False


    def adjust_yaw(self, yaw: float) -> float:
        """Adjust yaw angle to account for full rotations and return the adjusted yaw.

        This function keeps track of the number of full rotations both clockwise and counterclockwise, and adjusts the yaw angle accordingly so that it reflects the absolute angle in radians. It ensures that the yaw angle is not wrapped around to the range of -pi to pi, but instead accumulates the full rotations.
        This is particularly useful for applications where the absolute orientation of the vehicle is important, such as in control algorithms or navigation systems.
        The function also initializes the first yaw value and keeps track of the previous yaw value to determine if a full rotation has occurred.

        Args:
            yaw (float): The yaw angle in radians from the motion capture system after being converted from quaternion to euler angles.

        Returns:
            psi (float): The adjusted yaw angle in radians, accounting for full rotations.
        """        
        mocap_psi = yaw
        psi = None

        if not self.mocap_initialized:
            self.mocap_initialized = True
            self.prev_mocap_psi = mocap_psi
            psi = mocap_psi
            return psi

        # MoCap angles are from -pi to pi, whereas the angle state variable should be an absolute angle (i.e. no modulus wrt 2*pi)
        #   so we correct for this discrepancy here by keeping track of the number of full rotations.
        if self.prev_mocap_psi > np.pi*0.9 and mocap_psi < -np.pi*0.9: 
            self.full_rotations += 1  # Crossed 180deg in the CCW direction from +ve to -ve rad value so we add 2pi to keep it the equivalent positive value
        elif self.prev_mocap_psi < -np.pi*0.9 and mocap_psi > np.pi*0.9:
            self.full_rotations -= 1 # Crossed 180deg in the CW direction from -ve to +ve rad value so we subtract 2pi to keep it the equivalent negative value

        psi = mocap_psi + 2*np.pi * self.full_rotations
        self.prev_mocap_psi = mocap_psi
        
        return psi


    def vehicle_odometry_subscriber_callback(self, msg) -> None:
        """Callback function for vehicle odometry topic subscriber."""
        print("==" * 30)
        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2] #+ (2.25 * self.sim) # Adjust z for simulation, new gazebo model has ground level at around -1.39m 

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]


        self.roll, self.pitch, yaw = R.from_quat(msg.q, scalar_first=True).as_euler('xyz', degrees=False)
        self.yaw = self.adjust_yaw(yaw)  # Adjust yaw to account for full rotations
        r_final = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)         # Final rotation object
        self.rotation_object = r_final  # Store the final rotation object for further use

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.full_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw, self.p, self.q, self.r])
        self.nr_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw])
        self.flat_state_vector = np.array([self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0.])
        self.rta_mm_gpr_state_vector_planar = np.array([self.y, self.z, self.vy, self.vz, self.roll])# px, py, h, v, theta = x
        self.output_vector = np.array([self.x, self.y, self.z, self.yaw])
        self.position = np.array([self.x, self.y, self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.quat = self.rotation_object.as_quat()  # Quaternion representation (xyzw)
        self.ROT = self.rotation_object.as_matrix()
        self.omega = np.array([self.p, self.q, self.r])

        print(f"in odom, flat output: {self.output_vector}")

        # if self.odom_counter > 5:
        #     exit(0)
        ODOMETRY_DEBUG_PRINT = False
        if ODOMETRY_DEBUG_PRINT:
            # print(f"{self.full_state_vector=}")
            print(f"{self.nr_state_vector=}")
            # print(f"{self.flat_state_vector=}")
            print(f"{self.output_vector=}")
            print(f"{self.roll = }, {self.pitch = }, {self.yaw = }(rads)")
            # print(f"{self.rotation_object.as_euler('xyz', degrees=True) = } (degrees)")
            # print(f"{self.ROT = }")



# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        print("Updating Logged Data")
        self.time_log.append(data[0])
        self.x_log.append(data[1])
        self.y_log.append(data[2])
        self.z_log.append(data[3])
        self.yaw_log.append(data[4])
        self.ctrl_comp_time_log.append(data[5])
        self.throttle_log.append(data[6])
        self.roll_rate_log.append(data[7])
        self.pitch_rate_log.append(data[8])
        self.yaw_rate_log.append(data[9])



# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main(args=None):
    sim: Optional[bool] = None
    logger = None 
    offboard_control: Optional[OffboardControl] = None

    try:
        print(              
            f"{65 * '='}\n"
            f"Initializing ROS 2 node: '{__name__}' for offboard control\n"
            f"{65 * '='}\n"
        )

        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        sim_val = int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: "))
        if sim_val not in (0, 1):
            raise ValueError(
                            f"\n{65 * '='}\n"
                            f"Invalid input for sim: {sim_val}, expected 0 or 1\n"
                            f"{65 * '='}\n")
        sim = bool(sim_val)
        print(f"{'SIMULATION' if sim else 'HARDWARE'}")

        rclpy.init(args=args)
        offboard_control = OffboardControl(sim)

        filename = sys.argv[1] if len(sys.argv) > 1 else "log.log"
        base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
        base_dir = sys.argv[2] if len(sys.argv) > 2 else base_path
        logger = Logger(filename, base_dir)

        install_shutdown_logging(logger, offboard_control)#, also_shutdown=rclpy.shutdown)# Ensure logs are flushed on Ctrl+C / SIGTERM / normal exit

        rclpy.spin(offboard_control)  # Spin the node to keep it active and processing callbacks

    except KeyboardInterrupt:
        print(
              f"\n{65 * '='}\n"
              f"Keyboard interrupt detected (Ctrl+C), exiting...\n"
              )
    except Exception as e:
        traceback.print_exc()
    finally:
        print("\nNode has shut down.")

if __name__ == '__main__':
    main()