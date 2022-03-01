import asyncio
import random

from poke_env.player.player import Player

#compares types of active pokemon
def teampreview_performance(mon_a, mon_b):
    # We evaluate the performance on mon_a against mon_b as its type advantage
    a_on_b = b_on_a = -100
    for type_ in mon_a.types:
        if type_:
            a_on_b = max(a_on_b, type_.damage_multiplier(*mon_b.types))
    # We do the same for mon_b over mon_a
    for type_ in mon_b.types:
        if type_:
            b_on_a = max(b_on_a, type_.damage_multiplier(*mon_a.types))
    # Our performance metric is the different between the two
    print("a b:", a_on_b, " b a: ", b_on_a)
    return a_on_b - b_on_a

class MaxDamagePlayer(Player):
    
    def choose_move(self, battle):
        #save current active pokemon
        my_active = battle.active_pokemon
        opponent_active = battle.opponent_active_pokemon

        #calculate matchup
        matchup = teampreview_performance(my_active, opponent_active)
        #if matchup is bad, swap to next pokemon if possible
        if(matchup < 0 and len(battle.available_switches) >= 1):
            return self.create_order(battle.available_switches[0])

        # If the player can attack, it will
        if battle.available_moves:
            #initialize dmg calculator
            dmgCalc = DamageCalculator()

            # Finds the best move among available ones
            best_move = battle.available_moves[0]
            best_move_dmg = 0
            
            print("Available moves:", battle.available_moves)
            for move in battle.available_moves:
                #check for STAB bonus
                stab = 1
                if move.type in my_active.types:
                    stab = 1.5
                
                #calculate damage if move is used
                dmgCalc.ResultDamage(my_active.level, 
                    move.base_power, 
                    my_active.stats['atk'], 
                    opponent_active.base_stats['def'], 
                    1, 
                    stab, 
                    opponent_active.damage_multiplier(move.type), 
                    1)

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