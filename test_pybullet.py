import os, inspect

import numpy as np
import time, math
import subprocess
import pybullet as p2
import pybullet_data
from pybullet_utils import bullet_client as bc

import matplotlib.pyplot as plt

def step(p, action):
    # Send torque to joints (setJointMotorControl2)
    # Parameters:
    #   objUid
    #   jointIndex
    #   controlMode
    #   force
    p.setJointMotorControl2(RRrobot, 0, p.TORQUE_CONTROL, force=action[0])
    p.setJointMotorControl2(RRrobot, 1, p.TORQUE_CONTROL, force=action[1])

    # Simulate one time step
    p.stepSimulation()

    # Return state (getJointState)
    # Parameters:
    #   bodyID
    #   jointIndex
    # Returns:
    #   jointPosition
    #   jointvelocity
    #   jointReactionForces
    #   appliedJointMotorTroque (Only in VELOCITY_CONTROL or POSITION_CONTROL)
    state = p.getJointState(RRrobot, 1)[0:2] + p.getJointState(RRrobot, 0)[0:2]

    # Return state = { q1, q1_d, q2, q2_d }
    return np.array(state)

def render(p, physics_client_id):
    _render_width = 320
    _render_height = 200

    base_pos=[0,0,0]
    _cam_dist = 3
    _cam_pitch = -90
    _cam_yaw = 0
    if (physics_client_id>=0):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=_cam_dist,
            yaw=_cam_yaw,
            pitch=_cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
            aspect=float(_render_width) /
            _render_height,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
            width=_render_width,
            height=_render_height,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
    else:
        px = np.array([[[255,255,255,255]]*_render_width]*_render_height, dtype=np.uint8)
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(np.array(px), (_render_height, _render_width, -1))
    rgb_array = rgb_array[:, :, :3]

    return rgb_array

def joint_to_task_state(state):
    """Given joint angles, return end effector position
    
    """
    # Physical parameters
    l1 = 1
    l2 = 1

    # Unpack state
    q1, q1_d, q2, q2_d = state

    pos = np.zeros(2)
    pos[0] = l1 * cos(q1) + l2 * cos(q1+q2)
    pos[1] = l1 * sin(q1) + l2 * sin(q1+q2)
    
    vel = np.zeros(2)
    vel[0] = -q1_d * l1 * sin(q1) - (q1_d+q2_d) * l2 * sin(q1+q2)
    vel[1] =  q1_d * l1 * cos(q1) + (q1_d+q2_d) * l2 * cos(q1+q2)

    return pos, vel

def compute_force_jacobian(state):
    # Physical parameters
    l1 = 1
    l2 = 1

    # Unpack state
    q1, q1_d, q2, q2_d = state

    # Compute Jacobian
    J = np.zeros((2,2))
    J[0,0] = -l1 * sin(q1) - l2 * sin(q1 + q2)
    J[0,1] = -l2 * sin(q1 + q2)
    J[1,0] =  l1 * cos(q1) + l2 * cos(q1 + q2)
    J[1,1] =  l2 * cos(q1 + q2)

    return J


# Connect to simulation engine
p = bc.BulletClient(connection_mode=p2.GUI)
physics_client_id = p._client

# Reset simulation and load model
p.resetSimulation()
RRrobot = p.loadURDF("RR_planar_robot.urdf", [0, 0, 0])

# Remove all damping
p.changeDynamics(RRrobot, -1, linearDamping=0, angularDamping=0)
p.changeDynamics(RRrobot, 0, linearDamping=0, angularDamping=0)
p.changeDynamics(RRrobot, 1, linearDamping=0, angularDamping=0)

# Disable joint motors to use torque control
p.setJointMotorControl2(RRrobot, 0, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(RRrobot, 1, p.VELOCITY_CONTROL, force=0)

# Set gravity
p.setGravity(0, 0, 0)

# Set simulation time step and disable realtime
timeStep = 0.01
p.setTimeStep(timeStep)
p.setRealTimeSimulation(0)

# Reset joint states
p.resetJointState(RRrobot, 0, targetValue=0, targetVelocity=0)
p.resetJointState(RRrobot, 1, targetValue=0, targetVelocity=0)


# Random control
for i in range(1000):
    # action = [10*(np.random.random() - 0.5), None]
    action = [0, 0.1*math.sin(10*i/100)]
    step(p, action)
    rgb_array = render(p, physics_client_id)

