from typing import Optional, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from numba import njit
from pygame import gfxdraw

PI = np.pi
DT = 0.4
SPEED_LIMIT = 4.0
STEERING_LIMIT = PI / 4
N_SAMPLES_LON_ACTION = 2
N_SAMPLES_LAT_ACTION = 5
LON_ACTIONS = np.array([-SPEED_LIMIT, SPEED_LIMIT])
LAT_ACTIONS = np.linspace(-STEERING_LIMIT, STEERING_LIMIT, N_SAMPLES_LAT_ACTION)

STATE_H = 128
STATE_W = 128
SCREEN_H = 400
SCREEN_W = 500

SCALE = 8
STATE_SCALE = np.array(
    [SCREEN_W / SCALE, SCREEN_H / SCALE, 1, 1, SPEED_LIMIT]
    + [SCREEN_W / SCALE, SCREEN_H / SCALE] * 4
)
FPS = 30
RED = (255, 100, 100)
GREEN = (50, 200, 0)
BLUE = (100, 200, 255)
YELLOW = (200, 200, 0)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)

# position[2], heading, speed, length, width, type
INIT_STATE = np.array([0, 10, -PI, 4, 4.8, 1.8, 0])
GOAL_STATE = np.array([7 * 3 / 2, -20, 0, 0, 2, 2, 1])
STATIONARY_STATE = np.array(
    [
        [0, 0, 0, 0, SCREEN_W / SCALE / 2, 0.5, 1],
        [0, (SCREEN_H - 3) / SCALE / 2, 0, 0, SCREEN_W / SCALE, 1, 1],
        [0, -SCREEN_H / SCALE / 2, 0, 0, SCREEN_W / SCALE, 1, 1],
        [(SCREEN_W - 3) / SCALE / 2, 0, 0, 0, 1, SCREEN_H / SCALE, 1],
        [-SCREEN_W / SCALE / 2, 0, 0, 0, 1, SCREEN_H / SCALE, 1],
        [-5 * 3 / 2, -20, PI / 2, 0, 4.8, 1.8, 0],
        [-3 * 3 / 2, -20, PI / 2, 0, 4.8, 1.8, 0],
        [-5 * 3 / 2, -5, -PI / 2, 0, 4.8, 1.8, 0],
        [7 * 3 / 2, -5, -PI / 2, 0, 4.8, 1.8, 0],
        [-3 / 2, 5, PI / 2, 0, 4.8, 1.8, 0],
        [-3 * 3 / 2, 20, -PI / 2, 0, 4.8, 1.8, 0],
        [7 * 3 / 2, 20, -PI / 2, 0, 4.8, 1.8, 0],
    ]
)
MAX_STEPS = 256


@njit("types.none(f8[:], f8[:], f8)", fastmath=True, cache=True)
def kinematic_act(
    action: np.ndarray,
    state: np.ndarray,
    dt: float,
):
    beta = np.arctan(1 / 2 * np.tan(action[1]))
    state[3] = action[0]
    velocity = state[3] * np.array([np.cos(state[2] + beta), np.sin(state[2] + beta)])
    state[0:2] += velocity * dt
    state[2] += state[3] * np.sin(beta) / (state[4] / 2) * dt


@njit("f8[:](f8[:])", fastmath=True, cache=True)
def randomise_state(state: np.ndarray):
    state_copy = state.copy()
    state_copy[0] = np.random.uniform(-SCREEN_W / SCALE / 2, SCREEN_W / SCALE / 2)
    state_copy[1] = np.random.uniform(-SCREEN_H / SCALE / 2, SCREEN_H / SCALE / 2)
    state_copy[2] = np.random.uniform(-PI, PI)
    return state_copy


@njit("b1(f8[:,:], f8, f8[:,:,:], f8[:])", fastmath=True, cache=True)
def collision_check(
    ego: np.ndarray, ego_angle: float, others: np.ndarray, others_angle: np.ndarray
):
    # AABB
    ego_min_x = ego[:, 0].min()
    ego_max_x = ego[:, 0].max()
    ego_min_y = ego[:, 1].min()
    ego_max_y = ego[:, 1].max()
    for i in range(others.shape[0]):
        other_min_x = others[i, :, 0].min()
        other_max_x = others[i, :, 0].max()
        other_min_y = others[i, :, 1].min()
        other_max_y = others[i, :, 1].max()
        if (
            ego_min_x > other_max_x
            or ego_max_x < other_min_x
            or ego_min_y > other_max_y
            or ego_max_y < other_min_y
        ):
            continue
        else:
            # OBB
            def get_rot_mat(angle: float):
                return np.array(
                    [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
                )

            ego_rot_mat = get_rot_mat(ego_angle)
            other_rot_mat = get_rot_mat(others_angle[i])
            rot_mat = np.vstack((ego_rot_mat, other_rot_mat))
            projections_ego = np.dot(np.asfortranarray(ego), rot_mat.T)
            projections_other = np.dot(np.asfortranarray(others[i]), rot_mat.T)
            for j in range(len(rot_mat)):
                min1, max1 = np.min(projections_ego[:, j]), np.max(
                    projections_ego[:, j]
                )
                min2, max2 = np.min(projections_other[:, j]), np.max(
                    projections_other[:, j]
                )
                if max1 < min2 or max2 < min1:
                    continue
                return True
    return False


@njit("f8[:](f8[:], f8)", fastmath=True, cache=True)
def rotate_rad(pos, theta):
    rot_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return rot_matrix.dot(np.asfortranarray(pos))


@njit("f8[:,:](f8[:])", fastmath=True, cache=True)
def compute_vertices(state: np.ndarray):
    """
    shape of state: (7,)
    """
    l, r, t, b = (
        -state[4] / 2,
        state[4] / 2,
        state[5] / 2,
        -state[5] / 2,
    )
    vertices = np.array([[l, b], [l, t], [r, t], [r, b]])

    for i in range(vertices.shape[0]):
        vertices[i, :] = rotate_rad(vertices[i, :], state[2]) + state[:2]
    return vertices


@njit("f8[:,:](f8[:,:])", fastmath=True, cache=True)
def to_pixel(pos: np.ndarray):
    """
    shape of pos: (num_pos, 2)
    """
    pos_pixel = pos.copy()
    pos_pixel[:, 0] = pos_pixel[:, 0] * SCALE + SCREEN_W / 2
    pos_pixel[:, 1] = pos_pixel[:, 1] * SCALE + SCREEN_H / 2
    return pos_pixel


def draw_rectangle(
    surface: pygame.Surface,
    vertices: np.ndarray,
    color=None,
    obj_type: int = 0,
):
    object_surface = pygame.Surface(surface.get_size(), flags=pygame.SRCALPHA)
    if color is None:
        if obj_type == 0:
            color = YELLOW
        elif obj_type == 1:
            color = GREY
    pygame.draw.polygon(
        object_surface,
        color,
        vertices,
        width=2,
    )
    gfxdraw.filled_polygon(object_surface, vertices, color)
    surface.blit(object_surface, (0, 0))


def draw_direction_pattern(
    surface: pygame.Surface,
    state: np.ndarray,
):
    if state[-1] == 1:
        pass
    else:
        state_copy = state.copy()
        state_copy[0] += np.cos(state[2]) * state[4] / 8 * 3
        state_copy[1] += np.sin(state[2]) * state[4] / 8 * 3
        state_copy[4] = state[4] / 4
        vertices = to_pixel(compute_vertices(state_copy))
        draw_rectangle(surface, vertices, RED)


class Parking(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "no_render",
        ],
        "render_fps": FPS,
        "observation_types": [
            "rgb",
            "vector",
        ],
        "action_types": [
            "discrete",
            "multidiscrete",
            "continuous",
            "multicontinuous",
        ],
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        observation_type: Optional[str] = None,
        action_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert render_mode in self.metadata["render_modes"]
        assert observation_type in self.metadata["observation_types"]
        assert action_type in self.metadata["action_types"]
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.action_type = action_type

        if observation_type == "vector":
            self.observation_space = spaces.Box(
                -1.0, 1.0, (STATIONARY_STATE.shape[0] + 2, 13), dtype=np.float32
            )
        elif observation_type == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )

        if action_type == "discrete":
            self.action_space = spaces.Discrete(N_SAMPLES_LAT_ACTION)
        elif action_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete(
                [N_SAMPLES_LON_ACTION, N_SAMPLES_LAT_ACTION]
            )
        elif action_type == "continuous":
            self.action_space = spaces.Box(
                np.array([-STEERING_LIMIT]),
                np.array([STEERING_LIMIT]),
                dtype=np.float32,
            )
        elif action_type == "multicontinuous":
            self.action_space = spaces.Box(
                np.array([-SPEED_LIMIT, -STEERING_LIMIT]),
                np.array([SPEED_LIMIT, STEERING_LIMIT]),
                dtype=np.float32,
            )

        self.screen = None
        self.surf = None
        self.surf_movable = None
        self.surf_stationary = None
        self.clock = None

    def step(self, action: Union[np.ndarray, int]):
        if action is not None:
            if self.action_type == "discrete":
                action = np.array([SPEED_LIMIT, LAT_ACTIONS[action]])
            elif self.action_type == "multidiscrete":
                action = np.array([LON_ACTIONS[action[0]], LAT_ACTIONS[action[1]]])
            elif self.action_type == "continuous":
                action = np.clip(action, -1, 1) * STEERING_LIMIT
                action = np.array([SPEED_LIMIT, action.item()])
            elif self.action_type == "multicontinuous":
                action = np.clip(action, [-1, -1], [1, 1]) * [
                    SPEED_LIMIT,
                    STEERING_LIMIT,
                ]
            kinematic_act(action, self.movable[0], DT)
            self.movable_vertices = compute_vertices(self.movable[0])

        if self.observation_type == "rgb":
            self.obs = self._render("rgb_array", STATE_W, STATE_H)
        elif self.observation_type == "vector":
            self.obs[0, :2] = self.movable[0, :2]
            self.obs[0, 2:4] = [np.cos(self.movable[0, 2]), np.sin(self.movable[0, 2])]
            self.obs[0, 4] = self.movable[0, 3]
            self.obs[0, 5:] = self.movable_vertices.reshape(1, 8)
            self.obs[0, :] /= STATE_SCALE

        reward = self._reward()

        if self.render_mode == "human":
            self.render()
        return self.obs, reward, self.terminated, self.truncated, {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.stationary = STATIONARY_STATE
        self.stationary_vertices = np.zeros((self.stationary.shape[0], 4, 2))
        for i in range(self.stationary.shape[0]):
            self.stationary_vertices[i] = compute_vertices(self.stationary[i])

        self.movable = np.array([randomise_state(INIT_STATE)])
        self.movable_vertices = compute_vertices(self.movable[0])
        while collision_check(
            self.movable_vertices,
            self.movable[0, 2],
            self.stationary_vertices,
            self.stationary[:, 2],
        ):
            self.movable = np.array([randomise_state(INIT_STATE)])
            self.movable_vertices = compute_vertices(self.movable[0])
        self.goal_vertices = compute_vertices(GOAL_STATE)

        if self.observation_type == "vector":
            self.obs = np.zeros(
                (self.movable.shape[0] + self.stationary.shape[0] + 1, 13),
                dtype=np.float32,
            )
            self.obs[1, :2] = GOAL_STATE[:2]
            self.obs[1, 2:5] = [
                np.cos(GOAL_STATE[2]),
                np.sin(GOAL_STATE[2]),
                GOAL_STATE[3],
            ]
            self.obs[1, 5:] = self.goal_vertices.reshape(1, 8)
            self.obs[2:, :2] = self.stationary[:, :2]
            self.obs[2:, 2] = np.cos(self.stationary[:, 2])
            self.obs[2:, 3] = np.sin(self.stationary[:, 2])
            self.obs[2:, 4] = self.stationary[:, 3]
            self.obs[2:, 5:] = self.stationary_vertices.reshape(-1, 8)
            self.obs /= STATE_SCALE

        self.terminated = False
        self.truncated = False
        self.run_steps = 0

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _reward(self):
        reward = 0
        self.run_steps += 1
        if collision_check(
            self.movable_vertices,
            self.movable[0, 2],
            self.stationary_vertices,
            self.stationary[:, 2],
        ):
            self.terminated = True
            reward = -1.0
            return reward
        if collision_check(
            self.movable_vertices,
            self.movable[0, 2],
            np.array([self.goal_vertices]),
            np.array([GOAL_STATE[2]]),
        ):
            self.terminated = True
            reward = 1.0
            return reward
        if self.run_steps == MAX_STEPS:
            self.truncated = True
            reward = -1.0
            return reward
        return reward

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str, rgb_w: int = SCREEN_W, rgb_h: int = SCREEN_H):
        assert mode in self.metadata["render_modes"]

        if mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if mode == "human" or mode == "rgb_array":
            if self.surf_stationary is None:
                self.surf_stationary = pygame.Surface(
                    (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
                )
                for i in range(self.stationary.shape[0]):
                    draw_rectangle(
                        self.surf_stationary,
                        to_pixel(self.stationary_vertices[i]),
                        obj_type=self.stationary[i, -1],
                    )
                    draw_direction_pattern(self.surf_stationary, self.stationary[i])
                draw_rectangle(self.surf_stationary, to_pixel(self.goal_vertices), BLUE)

            if self.surf_movable is None:
                self.surf_movable = pygame.Surface(
                    (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
                )
            self.surf_movable.fill((0, 0, 0, 0))
            draw_rectangle(self.surf_movable, to_pixel(self.movable_vertices), GREEN)
            draw_direction_pattern(self.surf_movable, self.movable[0])

            surf = self.surf_stationary.copy()
            surf.blit(self.surf_movable, (0, 0))
            surf = pygame.transform.flip(surf, False, True)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(BLACK)
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(surf, (rgb_w, rgb_h))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )


if __name__ == "__main__":

    def handle_key_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.QUIT:
                quit = True

    env = Parking(
        render_mode="human", observation_type="vector", action_type="multicontinuous"
    )

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            handle_key_input()
            s, r, terminated, truncated, info = env.step(np.array([0.5, 0.0]))
            total_reward += r
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
