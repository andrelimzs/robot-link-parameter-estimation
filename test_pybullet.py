import os, inspect

import numpy as np
import time, math
import subprocess
import pybullet as p2
import pybullet_data
from pybullet_utils import bullet_client as bc

import matplotlib.pyplot as plt

def step(p, action):
    x_threshold = 0.4
    theta_threshold_radians = 12 * 2 * math.pi / 360

    force = action[0]

    p.setJointMotorControl2(cartpole, 0, p.TORQUE_CONTROL, force=force)
    p.stepSimulation()

    state = p.getJointState(cartpole, 1)[0:2] + p.getJointState(cartpole, 0)[0:2]
    theta, theta_dot, x, x_dot = state

    done =  x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians
    done = bool(done)
    reward = 1.0
    return np.array(state), reward, done, {}

def render(p, physics_client_id):
    _render_width = 320
    _render_height = 200

    base_pos=[0,0,0]
    _cam_dist = 2
    _cam_pitch = 0.3
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

# Connect to simulation engine
p = bc.BulletClient(connection_mode=p2.GUI)
physics_client_id = p._client

# Reset simulation and load model
p.resetSimulation()
cartpole = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cartpole.urdf"),
                                 [0, 0, 0])
p.changeDynamics(cartpole, -1, linearDamping=0, angularDamping=0)
p.changeDynamics(cartpole, 0, linearDamping=0, angularDamping=0)
p.changeDynamics(cartpole, 1, linearDamping=0, angularDamping=0)
timeStep = 0.02
p.setJointMotorControl2(cartpole, 1, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(cartpole, 0, p.VELOCITY_CONTROL, force=0)
p.setGravity(0, 0, -9.8)
p.setTimeStep(timeStep)
p.setRealTimeSimulation(0)

randstate = np.random.uniform(low=-0.05, high=0.05, size=(4,))
p.resetJointState(cartpole, 1, randstate[0], randstate[1])
p.resetJointState(cartpole, 0, randstate[2], randstate[3])


# Random control
for _ in range(1000):
    action = [np.random.random() - 0.5, None]
    step(p, action)
    time.sleep(0.01)

# Render in plt
# rgb_array = render(p, physics_client_id)
# plt.imshow(rgb_array)
# plt.show()