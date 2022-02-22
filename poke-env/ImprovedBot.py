# -*- coding: utf-8 -*-
import asyncio
import time

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from poke_env.player_configuration import PlayerConfiguration


class BetterMaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # movetype = move.type
            # multiplier: battle.opponent_active_pokemon.damage_multiplier(move.type)
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power * battle.opponent_active_pokemon.damage_multiplier(move.type))s
            
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)