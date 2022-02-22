import asyncio
import random

from poke_env.player.player import Player

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            #initialize dmg calculator
            dmgCalc = DamageCalculator()

            # Finds the best move among available ones
            best_move = battle.available_moves[0]
            best_move_dmg = 0
            
            for move in battle.available_moves:
                active = battle.active_pokemon
                opponent = battle.opponent_active_pokemon
                
                #check for STAB bonus
                stab = 1
                if move.type in active.types:
                    stab = 1.5
                
                #calculate damage if move is used
                dmgCalc.ResultDamage(active.level, move.base_power, active.stats['atk'], opponent.base_stats['def'], 1, stab, opponent.damage_multiplier(move.type), 1)

                #if damage is higher, assign new best move
                if(dmgCalc.NonCritResult > best_move_dmg):
                    best_move = move
                    best_move_dmg = dmgCalc.NonCritResult

            print("Bot chooses: ", best_move.id)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

class DamageCalculator:
    #minNonCrit = 0
    #maxNonCrit = 0
    NonCritResult = 0
    #minCrit = 0
    #maxCrit = 0
    CritResult = 0

    def ResultDamage(self, level, power, attack, defence, Weather, STAB, Type, Burn):
        randomMulti = random.uniform(0.85, 1.00)

        #  base damage = (((2 * level) / 5 + 2) * power * A / D) / 50 + 2
        base = (((2 * level) / 5 + 2) * power * attack / defence) / 50 + 2

        # baseDamage * Target * Weather * (Badge) * Critical * random * STAB * Type * Burn * other 
        # Target: 1 for 1v1, 0.75 for 2v2
        # Weather: Rainy, Sunny, etc...
        # Badge: not availble after gen2
        # Critical: 1.5 when crit
        # random: RNG number bewteen 0.85 and 1
        # STAB: When using attach type match Pokemon's type, x1.5, unless 1. (Adaptability ability)
        # Type: Type resistance
        # Burn: When attacker burned, physical move times 0.5
        # other: equipment, etc.
        self.NonCritResult = base * Weather * randomMulti * STAB * Type
        #self.minNonCrit = base * Target * Weather * 0.85 * STAB * Type
        #self.maxNonCrit = base * Target * Weather * STAB * Type
        self.CritResult = base * Weather * randomMulti * STAB * Type * 1.5
        #self.minCrit = base * Target * Weather * 0.85 * STAB * Type * 1.5
        #self.maxCrit = base * Target * Weather * STAB * Type * 1.5




        