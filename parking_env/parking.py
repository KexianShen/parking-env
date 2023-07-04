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
            rot_mat_func = lambda angle: np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            ego_rot_mat = rot_mat_func(ego_angle)
            other_rot_mat = rot_mat_func(others_angle[i])
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


def compute_vertices(state: np.ndarray):
    l, r, t, b = (
        -state[4] * SCALE / 2,
        state[4] * SCALE / 2,
        state[5] * SCALE / 2,
        -state[5] * SCALE / 2,
    )
    vertices = np.zeros((4, 2))
    for i, vertex in enumerate([(l, b), (l, t), (r, t), (r, b)]):
        vertex = pygame.math.Vector2(vertex).rotate_rad(state[2])
        vertex = np.array(
            [
                vertex[0] + (state[0]) * SCALE + SCREEN_W / 2,
                vertex[1] + (state[1]) * SCALE + SCREEN_H / 2,
            ]
        )
        vertices[i] = vertex
    return vertices


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
        vertices = compute_vertices(state_copy)
        draw_rectangle(surface, vertices, RED)


class Parking(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_rgb_array",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        multidiscrete: bool = False,
        continuous: bool = False,
        multicontinuous: bool = False,
    ) -> None:
        super().__init__()
        self.multidiscrete = multidiscrete
        self.continuous = continuous
        self.multicontinuous = multicontinuous
        if self.multidiscrete:
            self.action_space = spaces.MultiDiscrete(
                [N_SAMPLES_LON_ACTION, N_SAMPLES_LAT_ACTION]
            )
        elif self.continuous:
            self.action_space = spaces.Box(
                np.array([-STEERING_LIMIT]),
                np.array([STEERING_LIMIT]),
                dtype=np.float64,
            )
        elif self.multicontinuous:
            self.action_space = spaces.Box(
                np.array([-SPEED_LIMIT, -STEERING_LIMIT]),
                np.array([SPEED_LIMIT, STEERING_LIMIT]),
                dtype=np.float64,
            )
        else:
            self.action_space = spaces.Discrete(N_SAMPLES_LAT_ACTION)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )
        self.render_mode = render_mode
        self.screen = None
        self.surf = None
        self.surf_movable = None
        self.surf_stationary = None
        self.clock = None

    def step(self, action: Union[np.ndarray, int]):
        if action is not None:
            if self.multidiscrete:
                action = np.array([LON_ACTIONS[action[0]], LAT_ACTIONS[action[1]]])
            elif self.continuous:
                action = np.clip(action, -1, 1) * STEERING_LIMIT
                action = np.array([SPEED_LIMIT, action.item()])
            elif self.multicontinuous:
                action = np.clip(action, [-1, -1], [1, 1]) * [
                    SPEED_LIMIT,
                    STEERING_LIMIT,
                ]
            else:
                action = np.array([SPEED_LIMIT, LAT_ACTIONS[action]])
            kinematic_act(action, self.movable[0], DT)
            self.movable_vertices = compute_vertices(self.movable[0])

        self.state_rgb_array = self._render("state_rgb_array")
        reward = self._reward()
        if self.render_mode == "human":
            self.render()
        return self.state_rgb_array, reward, self.terminated, self.truncated, {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.stationary = np.array(
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

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if self.surf_movable is None:
            self.surf_movable = pygame.Surface(
                (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
            )
        self.surf_movable.fill((0, 0, 0, 0))
        draw_rectangle(self.surf_movable, self.movable_vertices, GREEN)
        draw_direction_pattern(self.surf_movable, self.movable[0])
        if self.surf_stationary is None:
            self.surf_stationary = pygame.Surface(
                (SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA
            )
            for i in range(self.stationary.shape[0]):
                draw_rectangle(
                    self.surf_stationary,
                    self.stationary_vertices[i],
                    obj_type=self.stationary[i, -1],
                )
                draw_direction_pattern(self.surf_stationary, self.stationary[i])
            draw_rectangle(self.surf_stationary, self.goal_vertices, BLUE)
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
            return self._create_image_array(surf, (SCREEN_W, SCREEN_H))
        elif mode == "state_rgb_array":
            return self._create_image_array(surf, (STATE_W, STATE_H))

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

    env = Parking(render_mode="human", multicontinuous=True)

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            handle_key_input()
            s, r, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
            total_reward += r
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
