import numpy as np
import tensorflow as tf

from poke_env.player.env_player import Gen8EnvSinglePlayer

from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.optimizers import Adam

from poke_env.player.random_player import RandomPlayer
from pokebot import MaxDamagePlayer

class SimpleRLPlayer(Gen8EnvSinglePlayer):
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
        active_can_dyna = battle.can_dynamax
        active_can_mega = battle.can_mega_evolve
        active_can_z = battle.can_z_move
        active_trapped = battle.trapped

        #opponent pokemon battle info
        opponent_active_stats = battle.opponent_active_pokemon.base_stats
        opponent_types = battle.oppoenent_active_pokemon.types
        opponent_can_dyna = battle.opponent_can_dynamax
        opponent_can_mega = battle.opponent_can_mega_evolve
        opponent_can_z = battle.opponent_can_z_move
        
        #battle info
        weather = battle.weather


        # Final vector with 10 components
        return np.concatenate(
            [moves_base_power, moves_damage_multiplier, [remaining_mon_team, remaining_mon_opponent],
            active_stats, active_types, active_can_dyna, active_can_mega, active_can_z, active_trapped,
            opponent_active_stats, opponent_types, opponent_can_dyna, opponent_can_mega, opponent_can_z,
            weather
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2,
            hp_value=1,
            victory_value=30,
        )
    
def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps)

    # This call will finished eventual unfinshed battles before returning
    player.complete_current_battle()

env_player = SimpleRLPlayer(battle_format="gen8randombattle")

# Output dimension
n_action = len(env_player.action_space)

model = Sequential()
model.add(Dense(128, activation="elu", input_shape=(1, 10,)))

# Our embedding have shape (1, 10), which affects our hidden layer dimension and output dimension
# Flattening resolve potential issues that would arise otherwise
model.add(Flatten())
model.add(Dense(64, activation="elu"))
model.add(Dense(n_action, activation="linear"))

model.summary(
    line_length=None,
    positions=None,
    print_fn=None,
)

graph = tf.compat.v1.get_default_graph()

memory = SequentialMemory(limit=10000, window_length=1)

# Simple epsilon greedy
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr="eps",
    value_max=1.0,
    value_min=0.05,
    value_test=0,
    nb_steps=10000,
)

# Defining our DQN
dqn = DQNAgent(
    model=model,
    nb_actions=22,
    policy=policy,
    memory=memory,
    nb_steps_warmup=1000,
    gamma=0.5,
    target_model_update=1,
    delta_clip=0.01,
    enable_double_dqn=True,
)

dqn.compile(Adam(learning_rate=0.001), metrics=["mae"])

opponent = RandomPlayer(battle_format="gen8randombattle")
second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

# Training
env_player.play_against(
    env_algorithm=dqn_training,
    opponent=second_opponent,
    env_algorithm_kwargs={"dqn": dqn, "nb_steps": 10000},
)

def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )

# Evaluation
print("Results against random player:")
env_player.play_against(
    env_algorithm=dqn_evaluation,
    opponent=opponent,
    env_algorithm_kwargs={"dqn": dqn, "nb_episodes": 100},
)

print("\nResults against max player:")
env_player.play_against(
    env_algorithm=dqn_evaluation,
    opponent=second_opponent,
    env_algorithm_kwargs={"dqn": dqn, "nb_episodes": 100},
)