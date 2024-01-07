import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt

class BicycleEnv(gym.Env):
    def __init__(self):
        super(BicycleEnv, self).__init__()

        # Action space: Continuous steering angle
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: Bicycle state (e.g., position, velocity, heading)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Environment parameters
        self.dt = 0.1  # Time step
        self.max_steps = 100  # Maximum number of steps

        # Bicycle model parameters
        self.length = 1.0  # Length of the bicycle
        self.velocity = 1.0  # Constant velocity

        # Initial state
        self.reset()

    def reset(self):
        # Reset the state of the environment
        self.steps = 0
        self.state = np.array([0.0, 0.0, 0.0])  # Initial state (x, y, heading)
        return self.state

    def step(self, action):
        # Apply action to the bicycle model
        steering_angle = np.clip(action[0], -1.0, 1.0)
        self.state[2] += self.velocity * np.tan(steering_angle) / self.length

        # Update position based on the bicycle dynamics
        self.state[0] += self.velocity * np.cos(self.state[2]) * self.dt
        self.state[1] += self.velocity * np.sin(self.state[2]) * self.dt

        # Compute reward (example: negative L2 distance to a target)
        target_position = np.array([5.0, 5.0])
        distance_to_target = np.linalg.norm(self.state[:2] - target_position)
        reward = -distance_to_target

        # Check if the episode is done
        done = (self.steps >= self.max_steps) or (distance_to_target < 0.1)

        # Increment the step counter
        self.steps += 1

        return self.state, reward, done, {}

    def render(self):
        # Implement rendering if necessary
        pass

# Create and wrap the environment
env = BicycleEnv()

# Define and train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("ppo_bicycle_model")

# Optional: Load the model back
# model = PPO.load("ppo_bicycle_model")

# Load the trained model
model = PPO.load("ppo_bicycle_model")

# Test the agent and visualize its behavior
num_episodes = 10  # Set the number of episodes to visualize

for _ in range(num_episodes):
    obs = env.reset()
    done = False
    trajectory = []

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _ = env.step(action)
        trajectory.append(env.state[:2].copy())  # Store x, y position

    # Visualize the trajectory
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
    plt.title("Agent's Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

# Close the environment
env.close()
