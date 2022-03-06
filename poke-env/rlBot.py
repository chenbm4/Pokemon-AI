import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf

from poke_env.player.player import Player
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.environment.effect import Effect
from poke_env.environment.status import Status
from poke_env.environment.field import Field
from poke_env.environment.weather import Weather
from poke_env.environment.pokemon_type import PokemonType
from poke_env.data import POKEDEX
from poke_env.data import MOVES
from poke_env.utils import to_id_str

# from tensorflow.python.keras.layers import Dense, Flatten
# from tensorflow.python.keras.models import Sequential

# from rl.agents.dqn import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from gym.spaces import Box, Discrete

# PPO dependencies
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from tensorflow.keras.optimizers import Adam

from poke_env.player.random_player import RandomPlayer
from pokebot import MaxDamagePlayer
from BattleState import GameModel

# NUM_TIMESTEPS = int(2e7)
# SEED = 721
# EVAL_FREQ = 250000
# EVAL_EPISODES = 1000
# LOGDIR = "ppo" # moved to zoo afterwards.

# load pokedex
df = pd.DataFrame.from_dict(POKEDEX).T
df["num"] = df["num"].astype(int)
df.drop(df[df["num"] <= 0].index, inplace=True)
pokemons = df.index.tolist() + ["unknown_pokemon"]

# list of possible abilities
abilities = set(
    [to_id_str(y) for x in df["abilities"].tolist() for y in x.values()]
)
abilities = list(abilities) + ["unknown_ability"]

# load moves
df = pd.DataFrame.from_dict(MOVES).T
moves = df.index.tolist() + ["unknown_move"]

# list of possible boosts
boosts = df["boosts"][~df["boosts"].isnull()].tolist()
boosts = list(set([key for item in boosts for key in item]))
# print(boosts)

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    observation_space = Box(low=-871082, high=871082, shape=(871082,))
    action_space = Discrete(22)

    def getThisPlayer(self):
        return self

    def __init__(self, *args, **kwargs):
        Gen8EnvSinglePlayer.__init__(self)

    def embed_battle(self, battle):
        feature_list = []

        GameModel(feature_list, pokemons, battle, abilities, moves, boosts)
    
        array = np.array(feature_list)
        return array

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

# for self play
# class TrainedRLPlayer(Player):

#     def getThisPlayer(self):
#         return self

#     def __init__(self, player, model, *args, **kwargs):
#         Player.__init__(self, *args, **kwargs)
#         self.model = model
#         self.opponent = player

#     def choose_move(self, battle):
#         state = self.opponent.embed_battle(battle)
#         predictions = self.model.predict([state])[0]
#         action = np.argmax(predictions)
#         return self.opponent._action_to_move(action, battle)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env_player = SimpleRLPlayer(battle_format="gen8randombattle")
env_player = Monitor(env_player, log_dir)
model = PPO('MlpPolicy', env_player, verbose=0)

def training_function(player):
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
    model.learn(total_timesteps=10000, callback=callback)

opponent = RandomPlayer(battle_format="gen8randombattle")
second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

def dqn_evaluation(player):
    player.reset_battles()
    for _ in range(100):
        done = False
        obs = player.reset()
        while not done:
            action = model.predict(obs)[0]
            obs, _, done, _ = player.step(action)
    player.complete_current_battle()

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, 100)
    )

# Training
print("Training:")
env_player.play_against(
    env_algorithm=training_function,
    opponent=opponent,
)

model.save("PPO_10M")

plot_results([log_dir], 10000000, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
plt.show()

#Evaluation
print("\nResults against random player:")
env_player.play_against(
    env_algorithm=dqn_evaluation,
    opponent=opponent,
)

print("\nResults against max player:")
env_player.play_against(
    env_algorithm=dqn_evaluation,
    opponent=second_opponent,
)