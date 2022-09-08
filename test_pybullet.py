import os, inspect

import numpy as np
import numpy.linalg as LA
import time
import math
from math import sin, cos
import subprocess
import pybullet as p2
import pybullet_data
from pybullet_utils import bullet_client as bc

import matplotlib.pyplot as plt


GUI_MODE = False

def step(p, action, CONTROL_TYPE='TORQUE'):
    # Limit torque to 5
    action = np.clip(action, -1, 1)

    # Send torque to joints (setJointMotorControl2)
    # Parameters:
    #   objUid
    #   jointIndex
    #   controlMode
    #   force
    if CONTROL_TYPE == 'TORQUE':
        p.setJointMotorControl2(RRrobot, 0, p.TORQUE_CONTROL, force=action[0])
        p.setJointMotorControl2(RRrobot, 1, p.TORQUE_CONTROL, force=action[1])
        
    elif CONTROL_TYPE == 'VELOCITY':
        p.setJointMotorControl2(RRrobot, 0, p.VELOCITY_CONTROL, targetVelocity=action[0])
        p.setJointMotorControl2(RRrobot, 1, p.VELOCITY_CONTROL, targetVelocity=action[1])

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
    state = p.getJointState(RRrobot, 0)[0:2] + p.getJointState(RRrobot, 1)[0:2]

    ee_state = p.getLinkState(RRrobot, 2)[0][0:2]

    # Return state = { q1, q1_d, q2, q2_d }
    return np.array(state), np.array(ee_state)

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
if GUI_MODE:
    p = bc.BulletClient(connection_mode=p2.GUI)
else:
    p = bc.BulletClient(connection_mode=p2.DIRECT)
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

state, ee_state = step(p, [0,0])

sim_length = int(30/timeStep)
log = {
    't': np.linspace(0,10,sim_length),
    'pos': np.zeros((2,sim_length)),
    'vel': np.zeros((2,sim_length)),
    'pos_des': np.zeros((2,sim_length)),
    'vel_des': np.zeros((2,sim_length)),
}

# Random control
for i in range(sim_length):
    # Time
    t = i * timeStep / 2

    # Get force Jacobian
    J = compute_force_jacobian(state)

    # Convert joint variables to cartesian
    pos = ee_state
    pos, vel = joint_to_task_state(state)
    # print("pos\t", pos)
    # print("pos_est\t", pos_est)

    # # Generate trajectory
    # pos_des = np.array([sin(2*t), 1 + 0.5*sin(t)])
    
    # TEST [ 0.5, sin(t) ]
    pos_des = np.array([sin(2*t), 1+0.5*sin(t)])

    # Control Law (PID)
    pos_Kp = 1
    vel_Kp = 5

    # Position Loop
    pos_err = pos_des - pos
    vel_des = pos_Kp * pos_err

    # Velocity Loop
    vel_err = vel_des - vel
    acc_des = vel_Kp * vel_err

    if 1:
        # Convert acceleration to joint ang acc
        q_dd_des = LA.pinv(J) @ acc_des.reshape((2,1))
        action = q_dd_des
        state, ee_state = step(p, action)

    else:
        # [TEST] Use velocity control
        action_vel = LA.pinv(J) @ vel_des.reshape((2,1))
        state, ee_state = step(p, action_vel, 'VELOCITY')

    # Log
    log['pos'][:,i] = pos
    log['vel'][:,i] = vel
    log['pos_des'][:,i] = pos_des
    log['vel_des'][:,i] = vel_des

    if GUI_MODE:
        time.sleep(0.001)
        # rgb_array = render(p, physics_client_id)

    if i % 10 == 0:
        print("\npos_des\t", pos_des)
        print("pos\t", pos)
        print("vel_des\t", vel_des)
        print("vel\t", vel)
        print("acc_des", acc_des)

# Plot Results
_, ax = plt.subplots(2,2)
# Position
ax[0,0].plot(log['t'], log['pos'][0])
ax[0,0].plot(log['t'], log['pos_des'][0])
ax[0,0].set_title('Position')

ax[1,0].plot(log['t'], log['pos'][1])
ax[1,0].plot(log['t'], log['pos_des'][1])

# Velocity
ax[0,1].plot(log['t'], log['vel'][0])
ax[0,1].plot(log['t'], log['vel_des'][0])
ax[0,1].set_title('Velocity')

ax[1,1].plot(log['t'], log['vel'][1])
ax[1,1].plot(log['t'], log['vel_des'][1])

_ = plt.figure()
plt.plot(log['pos'][0], log['pos'][1])
plt.plot(log['pos_des'][0], log['pos_des'][1])

plt.show()

