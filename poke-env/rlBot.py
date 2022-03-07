import numpy as np
import os
import pandas as pd

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import matplotlib.pyplot as plt

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.data import POKEDEX
from poke_env.data import MOVES
from poke_env.utils import to_id_str

import gym
from gym.spaces import Box, Discrete
from typing import Any, Dict
import torch
from torch import nn as nn

# from stable_baselines3.common.utils import linear_schedule

# PPO dependencies
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

from poke_env.player.random_player import RandomPlayer
from pokebot import MaxDamagePlayer
from BattleState import GameModel

N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

# ENV_ID = "PokemonEnv"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy"
}

# sample hyperparameters
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

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

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    observation_space = Box(low=0, high=255, shape=(1769,))
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
            battle, fainted_value=1, status_value=0.25, hp_value=0.5, victory_value=30
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

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

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
        self.save_path = os.path.join(log_dir, 'best_model_10M')
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
                    print(f"Saving new best model to best_model_10M")
                  self.model.save('best_model_10M')

        return True

# def objective(trial: optuna.Trial) -> float:

#     kwargs = DEFAULT_HYPERPARAMS.copy()
#     kwargs.update(sample_ppo_params(trial))
#     env_player = SimpleRLPlayer(battle_format="gen8randombattle")
#     env_player = Monitor(env_player, log_dir)
#     model = PPO(env=env_player, **kwargs)

#     eval_callback = TrialEvalCallback(
#         env_player, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
#     )

#     def training_function(player):
#         model.learn(total_timesteps=10000, callback=eval_callback)
    
#     opponent = RandomPlayer(battle_format="gen8randombattle")
#     nan_encountered = False
#     try:
#         env_player.play_against(
#             env_algorithm=training_function,
#             opponent=opponent,
#         )
#     except AssertionError as e:
#         print(e)
#         nan_encountered = True
#     finally:
#         model.env.close()
#         env_player.close()
    
#     if nan_encountered:
#         return float("nan")
    
#     if eval_callback.is_pruned:
#         raise optuna.exceptions.TrialPruned()

#     return eval_callback.last_mean_reward

# if __name__ == "__main__":
#     # Set pytorch num threads to 1 for faster training
#     torch.set_num_threads(1)

#     sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
#     # Do not prune before 1/3 of the max budget is used
#     pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

#     study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
#     try:
#         study.optimize(objective, n_trials=N_TRIALS, timeout=600)
#     except KeyboardInterrupt:
#         pass

#     print("Number of finished trials: ", len(study.trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))

#     print("  User attrs:")
#     for key, value in trial.user_attrs.items():
#         print("    {}: {}".format(key, value))
        
# # Create log dir
# log_dir = "tmp/"
# os.makedirs(log_dir, exist_ok=True)

env_player = SimpleRLPlayer(battle_format="gen8randombattle")
env_player = Monitor(env_player, log_dir)
model = PPO("MlpPolicy", env_player, verbose=0)

def training_function(player):
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        model.learn(total_timesteps=5000000, callback=callback)

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
print("Training Complete!")

model.load("best_model_5M")

plot_results([log_dir], 5000000, results_plotter.X_TIMESTEPS, "PPO Pokemon Showdown vs Random")
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