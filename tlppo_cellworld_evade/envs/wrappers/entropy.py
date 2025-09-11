import gym
import gymnasium
import numpy as np
import gym
import numpy as np
from typing import Tuple, Optional
import copy
class EntropyCalculationWrapper(gym.Wrapper):
    """
    A wrapper that calculates entropy based on:
    1. Distance to hexagonal boundary
    2. Distance to predator (if visible)
    """
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 1.0,  # weight for boundary distance
        beta: float = 1.0,   # weight for predator distance
    ):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        
        # Hexagon parameters (assuming the environment is a regular hexagon)
        self.center = np.array([0.5, 0.5])  # center of hexagon
        self.start_point = np.array([0.0, 0.5])
        self.end_point = np.array([1.0, 0.5])
        self.entropy_coef = 0.001
        
        # Calculate hexagon properties
        self.hex_radius = np.linalg.norm(self.end_point - self.start_point) / 2
        self.hex_vertices = self._compute_hexagon_vertices()

    def _compute_hexagon_vertices(self) -> np.ndarray:
        """
        Compute the vertices of the regular hexagon based on start and end points.
        """
        # Calculate vertices of regular hexagon
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices
        vertices = []
        for angle in angles:
            x = self.center[0] + self.hex_radius * np.cos(angle)
            y = self.center[1] + self.hex_radius * np.sin(angle)
            vertices.append([x, y])
        return np.array(vertices)

    def _distance_to_boundary(self, position: np.ndarray) -> float:
        """
        Calculate the minimum distance from a point to the hexagon boundary.
        
        Args:
            position: Current position [x, y]
            
        Returns:
            float: Minimum distance to boundary
        """
        position = np.array(position)
        
        # Calculate distances to each edge (line segment between vertices)
        distances = []
        for i in range(len(self.hex_vertices)):
            j = (i + 1) % len(self.hex_vertices)
            v1 = self.hex_vertices[i]
            v2 = self.hex_vertices[j]
            
            # Calculate distance to line segment
            line_vec = v2 - v1
            point_vec = position - v1
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len
            point_vec_scaled = point_vec / line_len
            
            t = np.dot(line_unitvec, point_vec_scaled)
            t = max(0, min(1, t))  # Clamp to line segment
            
            nearest = v1 + t * line_vec
            distance = np.linalg.norm(position - nearest)
            distances.append(distance)
            
        return min(distances)

    def _distance_to_predator(self, prey_pos: np.ndarray, predator_pos: np.ndarray) -> float:
        """
        Calculate the distance to predator if visible.
        
        Args:
            prey_pos: Prey position [x, y]
            predator_pos: Predator position [x, y]
            
        Returns:
            float: Distance to predator
        """
        return np.linalg.norm(prey_pos - predator_pos)

    def _calculate_entropy(self, obs: np.ndarray) -> float:
        """
        Calculate the entropy based on boundary distance and predator distance.
        Higher entropy means higher risk (center of map or close to predator).
        Lower entropy means lower risk (near boundary, far from predator).
        
        Args:
            obs: Observation array containing prey and predator positions
            
        Returns:
            float: Calculated entropy value
        """
        # Extract positions from observation
        prey_pos = obs[0:2]
        predator_pos = obs[3:5]
        
        # Calculate boundary component
        boundary_dist = self._distance_to_boundary(prey_pos)
        # Normalize by hexagon radius
        normalized_boundary_dist = boundary_dist / self.hex_radius
        boundary_entropy = normalized_boundary_dist  # Higher distance = higher entropy
        
        # Calculate predator component
        predator_entropy = 0.0
        # print(hasattr(self.env, 'predator_visible'))
        # if hasattr(self.env, 'predator_visible') and self.env.predator_visible():
        #     if not np.all(predator_pos == 0):  # Check if predator position is valid
        #         pred_dist = self._distance_to_predator(prey_pos, predator_pos)
        #         # Normalize by diagonal length and inverse (closer = higher entropy)
        #         normalized_pred_dist = pred_dist / (2 * self.hex_radius)
        #         predator_entropy = 1.0 / (normalized_pred_dist + 1e-5)
        #         print(f"Predator distance: {pred_dist}")
        #         print(f"Predator entropy: {predator_entropy}")
        
        # Combine components with weights
        total_entropy = (
            self.alpha * boundary_entropy +
            self.beta * predator_entropy
        )
        
        return total_entropy

    def step(self, action):
        """
        Execute environment step and calculate entropy.
        """
        obs, reward, done, _, info = self.env.step(action)
        
        # Calculate entropy
        entropy = self._calculate_entropy(obs)

        
        # Add entropy to info dict
        info['entropy'] = entropy
        
        reward = reward - self.entropy_coef * entropy  # Uncomment if you want to use entropy in reward

        # print(f"Entropy: {entropy}")
        # print(f"obs: {obs}")
        return obs, reward, False, info

    def reset(self):
        """
        Reset the environment and calculate initial entropy.
        """
        obs, _ = self.env.reset()
        
        # Calculate initial entropy
        entropy = self._calculate_entropy(obs)
        
        # You might want to store initial entropy or use it somehow
        self._initial_entropy = entropy
        
        return obs


class uncertainty_wrapper_predator(gym.Env):
    def __init__(self, env, cfg):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11 + 1,), dtype=np.float32)
        self.model = env.model


    def see_predator(self):
        # if the predator is visible, then it is a safe zone
        if self.env.model.prey_data.predator_visible:
            return True
        else:
            return False

    def uncertainty_level(self, obs):
        # Get the distance between prey and predator from the observation
        predator_prey_distance = obs[7]
        # Determine uncertainty level based on the distance
        if not self.see_predator():
            return 0
        elif predator_prey_distance < 0.1:
            return 1
        elif predator_prey_distance < 0.3:
            return 0.5
        else:
            return 0.1

    def reset(self, seed=None):
        obs, _ = self.env.reset()
        # add the uncertainty level to the obs
        # set the initial heuristic to 1
        uncertainty = self.uncertainty_level(obs)
        obs = np.append(obs, uncertainty)
        return obs

    def step(self, action):
        # if uncertainty == 1:
        #     action = self.wait_action()
        # else:
        #     action = action
        obs, reward, done, tr, info = self.env.step(action)
        uncertainty = self.uncertainty_level(obs)
        obs = np.append(obs, uncertainty)
        obs = obs.astype(np.float32)
        return obs, reward, False, info

    def render(self, mode='human'):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped
    
class MypreyWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.model = env.model
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,), 
            dtype=np.float32
        )
        

    def see_predator(self):
        # if the predator is visible, then it is a safe zone
        if self.model.prey_data.predator_visible:
            return True
        else:
            return False

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        obs = copy.deepcopy(obs)
        # if self.see_predator():
        #     print('Predator is visible')
        # else:
        #     print('Predator is not visible')
        obs = np.delete(obs, -4)
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self, seed=None):
        obs, _ = self.env.reset()
        obs = obs.astype(np.float32)
        obs = copy.deepcopy(obs)
        obs = np.delete(obs, -4)
        return obs
    
class MypreyWrapper_v2(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.model = env.model
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(3,),  
            dtype=np.float32
        )
    def see_predator(self):
        if self.model.prey_data.predator_visible:
            return True
        else:
            return False

    def step(self, action):
        if action[2] > 0.5:
            wait_pos = self.wait_action()
            action = np.array([wait_pos[0], wait_pos[1]])
        else:
            action = action[:2].copy()
            
        obs, reward, done,  _, info = self.env.step(action)
        obs = obs.astype(np.float32)
        new_obs = obs.copy()
        new_obs = np.delete(new_obs, -4)
        return new_obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self, seed=None):
        obs, _ = self.env.reset()
        obs = obs.astype(np.float32)
        new_obs = obs.copy()
        new_obs = np.delete(new_obs, -4)
        return new_obs
    
    def wait_action(self):
        noise_x = np.random.uniform(-0.02, 0.02)
        noise_y = np.random.uniform(-0.02, 0.02)
        current_x = self.env.model.prey.state.location[0]
        current_y = self.env.model.prey.state.location[1]
        new_x = np.clip(current_x + noise_x, 0.0, 1.0)
        new_y = np.clip(current_y + noise_y, 0.0, 1.0)
        return tuple((new_x, new_y))
    
