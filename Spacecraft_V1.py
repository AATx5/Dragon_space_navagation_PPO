import os

import numpy as np
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# body ids in spacecraft.xml: world = 0, target = 1, ship = 2
DEFAULT_CAMERA_CONFIG = {"trackbodyid": 2, "distance": 6.0}

SHIP_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacecraft.xml")


class SpacecraftEnv(MujocoEnv, utils.EzPickle):
    # dt = timestep (0.005) * frame_skip (5) = 0.025 s  ->  40 fps
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(self,
        xml_file: str = SHIP_XML,
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        render_mode=None,
        distance_weight: float = 5.0,
        ctrl_cost_weight: float = 0.05,
        time_penalty: float = 0.2,          
        goal_radius: float = 0.35,          # half of the ship's ~0.7 m body width
        goal_bonus: float = 20.0,           # bonus on the first step inside the goal
        goal_bonus_growth: float = 0.5,     # extra bonus for every further step inside
        goal_bonus_max: float = 100.0,      # cap so the reward cannot grow without limit
        target_speed: float = 0.1,          # m/s, slow drift

        target_range: float = 30.0,          # target starts inside a box of x m
        target_bound: float = 30.0,          # target bounces off a box of +-5 m

        min_target_start_distance: float = 1.0,
        reset_noise_scale: float = 0.0,     # 0 = ship always starts at rest at the origin
        max_stray_distance: float = 20.0,   # episode ends if the ship drifts too far from the target

        termination_penalty: float = 400.0, # subtracted from the reward on the step the episode is terminated

        progress_reward: float = 4.0,       # constant paid on any step the ship got closer to the target
        target_pitch: float = 0.0,          # radians, 0 = ship's nose points along +x axis
        target_yaw: float = 0.0,            # radians, 0 = ship's nose points along +x axis
        pitch_reward_weight: float = 75.0,      # weight for pitch reward
        yaw_reward_weight: float = 75.0,        # weight for yaw reward
        **kwargs,
        ):
        utils.EzPickle.__init__(
            self,
            xml_file=xml_file,
            frame_skip=frame_skip,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            distance_weight=distance_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            time_penalty=time_penalty,
            goal_radius=goal_radius,
            goal_bonus=goal_bonus,
            goal_bonus_growth=goal_bonus_growth,
            goal_bonus_max=goal_bonus_max,
            target_speed=target_speed,
            target_range=target_range,
            target_bound=target_bound,
            max_stray_distance=max_stray_distance,
            termination_penalty=termination_penalty,
            progress_reward=progress_reward,
            min_target_start_distance=min_target_start_distance,
            reset_noise_scale=reset_noise_scale,
            target_pitch=target_pitch,
            target_yaw=target_yaw,
            pitch_reward_weight=pitch_reward_weight,
            yaw_reward_weight=yaw_reward_weight,
            **kwargs,
        )

        self.distance_weight = distance_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.time_penalty = time_penalty
        self.goal_radius = goal_radius
        self.goal_bonus = goal_bonus
        self.goal_bonus_growth = goal_bonus_growth
        self.goal_bonus_max = goal_bonus_max
        self.target_speed = target_speed
        self.target_range = target_range
        self.target_bound = target_bound
        self.min_target_start_distance = min_target_start_distance
        self._reset_noise_scale = reset_noise_scale
        self.max_stray_distance = max_stray_distance
        self.termination_penalty = termination_penalty
        self.progress_reward = progress_reward
        self.target_pitch = target_pitch
        self.target_yaw = target_yaw
        self.pitch_reward_weight = pitch_reward_weight
        self.yaw_reward_weight = yaw_reward_weight
        self.prev_distance = 0.0   # set for real in reset_model

        # set again every episode in reset_model
        self.target_pos = np.zeros(3)
        self.target_vel = np.zeros(3)
        self.steps_in_goal = 0

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            **kwargs,
        )

        # observation (16 numbers, all in the ship's own frame):
        #   orientation quaternion (4), linear velocity (3), angular velocity (3),
        #   target position relative to ship (3), target velocity (3)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64
        )

    def step(self, action):
            # thrusters can only push: keep every thrust between 0 and 1
            action = np.clip(action, 0.0, 1.0)

            # move the target (it is a mocap body, so we just write its position)
            self.target_pos = self.target_pos + self.target_vel * self.dt
            for axis in range(3):
                if abs(self.target_pos[axis]) > self.target_bound:
                    self.target_pos[axis] = np.clip(self.target_pos[axis], -self.target_bound, self.target_bound)
                    self.target_vel[axis] = -self.target_vel[axis]   # bounce back
            self.data.mocap_pos[0] = self.target_pos

            #simulate the environment for a number of frames defined by frame_skip
            self.do_simulation(action, self.frame_skip)

            #observe the enviroment of this time step
            observation = self._get_obs()

            distance = np.linalg.norm(self.data.qpos[0:3] - self.target_pos)

            # count how many steps in a row the ship has stayed inside the goal
            if distance < self.goal_radius:
                self.steps_in_goal += 1
            else:
                self.steps_in_goal = 0

            # did the ship get closer than it was on the previous step?
            closer = distance < self.prev_distance
            self.prev_distance = distance

            # where the nose points (pitch = up/down angle, yaw = left/right angle)
            pitch, yaw = self.get_nose_angles()

            reward = self.get_reward(self.steps_in_goal, action, closer, pitch, yaw)

            #termanation condition: the episode ends when the ship drifts too far away from the target
            pos = self.data.qpos[0:3]
            terminated = bool(np.any(np.abs(pos) > self.target_bound + 5.0))
            
            if terminated:
                reward -= self.termination_penalty

            info = {
            "distance_to_goal": distance,
            "steps_in_goal": self.steps_in_goal,
            "target_x": self.target_pos[0],
            "target_y": self.target_pos[1],
            "target_z": self.target_pos[2],
        }

            if self.render_mode == "human":
                self.render()
            # no terminated condition: the episode ends by the time limit set in gym.register
            return observation, reward, terminated, False, info

    def get_nose_angles(self):
            # the ship's nose is its own +x axis; rotate that axis into the world frame
            nose = np.zeros(3)
            mujoco.mju_rotVecQuat(nose, np.array([1.0, 0.0, 0.0]), self.data.qpos[3:7])

            # yaw   = angle of the nose in the x-y plane (0 = along +x, positive turns toward +y)
            # pitch = angle of the nose above the x-y plane (0 = level, positive = nose up)
            yaw = np.arctan2(nose[1], nose[0])
            pitch = np.arcsin(np.clip(nose[2], -1.0, 1.0))
            return pitch, yaw

    def get_reward(self, steps_in_goal: int, action, closer: bool, pitch: float, yaw: float):
            
            # distance_reward = a constant for every step the ship got closer, otherwise 0
            distance_reward = self.progress_reward if closer else -self.progress_reward

            # ctrl_cost = penalty for burning fuel
            ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

            # inside the goal: the goal bonus (grows the longer the ship stays in)
            # and the pitch / yaw reward. Outside the goal both are 0.
            if steps_in_goal > 0:
                bonus = self.goal_bonus + self.goal_bonus_growth * (steps_in_goal - 1)
                bonus = min(bonus, self.goal_bonus_max)

                # cos is 1 when the nose is exactly on the target angle and -1 when it is opposite
                pitch_reward = self.pitch_reward_weight * np.cos(pitch - self.target_pitch)
                yaw_reward = self.yaw_reward_weight * np.cos(yaw - self.target_yaw)
            else:
                bonus = 0.0
                pitch_reward = 0.0
                yaw_reward = 0.0

            # total reward = distance_reward - ctrl_cost - time_penalty + goal_bonus + pitch_reward + yaw_reward
            reward = distance_reward - ctrl_cost - self.time_penalty + bonus + pitch_reward + yaw_reward

            return reward

    def _get_obs(self):
            position = self.data.qpos[0:3]
            quat = self.data.qpos[3:7]
            lin_vel_world = self.data.qvel[0:3]
            ang_vel = self.data.qvel[3:6]          # already in the ship's own frame

            # rotation matrix: ship frame -> world frame. Its transpose goes world -> ship.
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            rot = rot.reshape(3, 3)

            lin_vel_local = rot.T @ lin_vel_world
            target_local = rot.T @ (self.target_pos - position)
            target_vel_local = rot.T @ self.target_vel

            return np.concatenate((quat, lin_vel_local, ang_vel, target_local, target_vel_local))

    def reset_model(self):
            noise = self._reset_noise_scale

            # ship starts near the origin with a small random tilt and spin
            qpos = self.init_qpos.copy()
            qpos[0:3] += self.np_random.uniform(low=-noise, high=noise, size=3)
            quat = qpos[3:7] + self.np_random.uniform(low=-noise, high=noise, size=4)
            qpos[3:7] = quat / np.linalg.norm(quat)

            qvel = (
            self.init_qvel
            + noise * self.np_random.standard_normal(self.model.nv)
            )

            self.set_state(qpos, qvel)

            # target starts at a random point (not already on top of the ship)
            while True:
                self.target_pos = self.np_random.uniform(
                    low=-self.target_range, high=self.target_range, size=3
                )
                if np.linalg.norm(self.target_pos - qpos[0:3]) > self.min_target_start_distance:
                    break

            # and drifts slowly in a random direction
            direction = self.np_random.standard_normal(3)
            direction = direction / np.linalg.norm(direction)
            self.target_vel = self.target_speed * direction

            self.data.mocap_pos[0] = self.target_pos
            self.steps_in_goal = 0
            self.prev_distance = np.linalg.norm(qpos[0:3] - self.target_pos)

            observation = self._get_obs()

            return observation

    def _get_reset_info(self):
            return {
            "distance_to_goal": np.linalg.norm(self.data.qpos[0:3] - self.target_pos),
            "target_x": self.target_pos[0],
            "target_y": self.target_pos[1],
            "target_z": self.target_pos[2],
        }
