import numpy as np
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf

from poke_env.player.player import Player
from poke_env.player.env_player import Gen8EnvSinglePlayer

# from tensorflow.python.keras.layers import Dense, Flatten
# from tensorflow.python.keras.models import Sequential

# from rl.agents.dqn import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

from gym.spaces import Box, Discrete

# PPO dependencies
import gym
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common import make_vec_env
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.callbacks import EvalCallback

from tensorflow.keras.optimizers import Adam

from poke_env.player.random_player import RandomPlayer
from pokebot import MaxDamagePlayer

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "ppo" # moved to zoo afterwards.

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    observation_space = Box(low=-17, high=17, shape=(17,))
    action_space = Discrete(22)

    def getThisPlayer(self):
        return self

    def __init__(self, *args, **kwargs):
        Gen8EnvSinglePlayer.__init__(self)

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_damage_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 # rescaling to make learning easier
            if move.type:
                moves_damage_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        
        #active pokemon battle info
        active_stats = battle.active_pokemon.stats
        active_types = battle.active_pokemon.types
        if (battle.can_dynamax):
            active_can_dyna = 1
        else:
            active_can_dyna = 0
        if (battle.can_mega_evolve):
            active_can_mega = 1
        else:
            active_can_mega = 0
        if (battle.can_z_move):
            active_can_z = 1
        else:
            active_can_z = 0
        if (battle.trapped):
            active_trapped = 1
        else:
            active_trapped = 0

        #opponent pokemon battle info
        opponent_active_stats = battle.opponent_active_pokemon.base_stats
        opponent_types = battle.opponent_active_pokemon.types
        opponent_can_dyna = battle.opponent_can_dynamax
        opponent_can_mega = battle.opponent_can_mega_evolve
        opponent_can_z = battle.opponent_can_z_move
        if (battle.opponent_can_dynamax):
            opponent_can_dyna = 1
        else:
            opponent_can_dyna = 0
        if (battle.opponent_can_mega_evolve):
            opponent_can_mega = 1
        else:
            opponent_can_mega = 0
        if (battle.opponent_can_z_move):
            opponent_can_z = 1
        else:
            opponent_can_z = 0
        
        #battle info
        weather = battle.weather


        # Final vector
        return np.concatenate(
            [moves_base_power, moves_damage_multiplier, [remaining_mon_team, remaining_mon_opponent],
            [active_can_dyna], [active_can_mega], [active_can_z], [active_trapped],
            [opponent_can_dyna], [opponent_can_mega], [opponent_can_z]
            ]
        )

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

env_player = SimpleRLPlayer(battle_format="gen8randombattle")

model = PPO('MlpPolicy', env_player, clip_range=0.2, verbose=2)

def training_function(player):
    model.learn(total_timesteps=100000)

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