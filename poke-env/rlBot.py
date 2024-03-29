import numpy as np
import os
import pandas as pd
import enum

import matplotlib.pyplot as plt

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.env_player import EnvPlayer
from poke_env.data import POKEDEX
from poke_env.data import MOVES
from poke_env.utils import to_id_str

from gym.spaces import Box, Discrete

# PPO dependencies
from stable_baselines3 import PPO, DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

from poke_env.player.random_player import RandomPlayer
from pokebot import MaxDamagePlayer # can change to DamageCalculator
from BattleState import GameModel, SimpleGameModel

class Algorithm(enum.Enum):
    DQN = 0
    PPO = 1

class Opponent(enum.Enum):
    Random = 0
    MaxDamage = 1

class State(enum.Enum):
    Simple = 0
    Complex = 1

TIMESTEPS = 300000
STATE = State.Complex
OPPONENT = Opponent.MaxDamage
ALGORITHM = Algorithm.PPO

# load pokedex
df = pd.DataFrame.from_dict(POKEDEX).T
df["num"] = df["num"].astype(int)
df.drop(df[df["num"] <= 0].index, inplace=True)

# list of possible abilities
ABILITIES_LIST = set(
    [to_id_str(y) for x in df["abilities"].tolist() for y in x.values()]
)
ABILITIES_LIST = list(ABILITIES_LIST) + ["unknown_ability"]

# load moves
df = pd.DataFrame.from_dict(MOVES).T
MOVES_LIST = df.index.tolist() + ["unknown_move"]

# list of possible boosts
BOOSTS_LIST = df["boosts"][~df["boosts"].isnull()].tolist()
BOOSTS_LIST = list(set([key for item in BOOSTS_LIST for key in item]))
# print(boosts)

# Create log dir
LOG_DIR = "tmp/"
os.makedirs(LOG_DIR, exist_ok=True)

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    if STATE == State.Simple:
        observation_space = Box(low=0, high=2, shape=(10,))
    elif STATE == State.Complex:
        observation_space = Box(low=0, high=255, shape=(1769,))
    action_space = Discrete(22)

    def getThisPlayer(self):
        return self

    def __init__(self, *args, **kwargs):
        self.model = None
        Gen8EnvSinglePlayer.__init__(self, *args, **kwargs)

    def embed_battle(self, battle):
        if STATE == State.Simple:
            return SimpleGameModel(battle)
        elif STATE == State.Complex:
            feature_list = []
            GameModel(feature_list, battle, ABILITIES_LIST, MOVES_LIST, BOOSTS_LIST)
            array = np.array(feature_list)
            return array

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )
        
    # this is only run when login.py is run
    def choose_move(self, battle):
        if self.model == None:
            if battle not in self._observations or battle not in self._actions:
                self._init_battle(battle)
            self._observations[battle].put(self.embed_battle(battle))
            action = self._actions[battle].get()

            return self._action_to_move(action, battle)
        else:
            state = self.embed_battle(battle)
            print(state)
            predictions = self.model.predict(np.expand_dims(state, 0))[0] 
            #WARNING: may have different behavior baased on model type
            print(predictions)
            return self._action_to_move(predictions[0], battle)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param LOG_DIR: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, LOG_DIR: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.LOG_DIR = LOG_DIR
        self.save_path = os.path.join(LOG_DIR, f"{STATE.name}_{ALGORITHM.name}_{OPPONENT.name}_{str(TIMESTEPS)}")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path+"_images", exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.LOG_DIR), 'timesteps')
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

if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")
    env_player = Monitor(env_player, LOG_DIR)
    if ALGORITHM == Algorithm.DQN:
        model = DQN("MlpPolicy", env_player, verbose=0)
    elif ALGORITHM == Algorithm.PPO:
        model = PPO("MlpPolicy", env_player, verbose=0)

    def training_function(player):
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, LOG_DIR=LOG_DIR)
            model.learn(total_timesteps=TIMESTEPS, callback=callback)

    if OPPONENT == Opponent.Random:
        opponent = RandomPlayer(battle_format="gen8randombattle")
    elif OPPONENT == Opponent.MaxDamage:
        opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    def evaluation(player):
        player.reset_battles()
        for _ in range(100):
            done = False
            obs = player.reset()
            while not done:
                action = model.predict(obs)[0]
                obs, _, done, _ = player.step(action)
        player.complete_current_battle()

        print(
            "Evaluation: %d victories out of %d episodes"
            % (player.n_won_battles, 100)
        )

    # Training
    print("Training:")
    env_player.play_against(
        env_algorithm=training_function,
        opponent=opponent,
    )
    print("Training Complete!")

    model.load(os.path.join(LOG_DIR, f"{STATE.name}_{ALGORITHM.name}_{OPPONENT.name}_{str(TIMESTEPS)}"))

    plot_results([LOG_DIR], TIMESTEPS, results_plotter.X_TIMESTEPS, f"{ALGORITHM.name} Pokemon Showdown vs {OPPONENT.name} {STATE.name}")
    plt.show()

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    #Evaluation
    print("\nResults against random player:")
    env_player.play_against(
        env_algorithm=evaluation,
        opponent=opponent,
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=evaluation,
        opponent=second_opponent,
    )
