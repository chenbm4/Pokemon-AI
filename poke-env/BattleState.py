import numpy as np

from poke_env.environment.effect import Effect
from poke_env.environment.status import Status
from poke_env.environment.field import Field
from poke_env.environment.weather import Weather
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move import EmptyMove
from poke_env.utils import to_id_str

def MoveModel(feature_list, move, abilities, moves, boosts) -> None:
    move_accuracy = move.accuracy
    if move_accuracy:
        feature_list.append(move_accuracy)
    else:
        feature_list.append(0)

    move_base_pwr = move.base_power
    if move_base_pwr:
        feature_list.append(move_base_pwr / 100)
    else:
        feature_list.append(0)

    move_boosts = move.boosts
    for attr in boosts:
        if move_boosts and (attr in move_boosts):
            feature_list.append(move_boosts[attr])
        else:
            feature_list.append(0)

    move_breaks_prot = 0
    if move.breaks_protect:
        move_breaks_prot = 1
    feature_list.append(move_breaks_prot)

    move_physical = 0
    move_special = 0
    move_status = 0
    if move.category == 1:
        move_physical = 1
    elif move.category == 2:
        move_special = 1
    else:
        move_status = 1
    feature_list.append(move_physical)
    feature_list.append(move_special)
    feature_list.append(move_status)

    move_crit = move.crit_ratio
    if move_crit:
        feature_list.append(move_crit)
    else:
        feature_list.append(0)

    move_current_pp = move.current_pp
    if move_current_pp:
        feature_list.append(move_current_pp / 10)
    else:
        feature_list.append(0)

    move_drain = move.drain
    if move_drain:
        feature_list.append(move_drain)
    else:
        feature_list.append(0)
    
    # move_defphysical = 0
    # move_defspecial = 0
    # move_defstatus = 0
    # if move.defensive_category == 1:
    #     move_defphysical = 1
    # elif move.defensive_category == 2:
    #     move_defspecial = 1
    # else:
    #     move_defstatus = 1
    # feature_list.append(move_defphysical)
    # feature_list.append(move_defspecial)
    # feature_list.append(move_defstatus)

    move_exp_hits = move.expected_hits
    if move_exp_hits:
        feature_list.append(move_exp_hits)
    else:
        feature_list.append(0)

    move_force_switch = 0
    if move.force_switch:
        move_force_switch = 1
    feature_list.append(move_force_switch)

    move_heal = move.heal
    if move_heal:
        feature_list.append(move_heal)
    else:
        feature_list.append(0)
    
    # move_ignore_abil = 0
    # if move.ignore_ability:
    #     move_ignore_abil = 1
    # feature_list.append(move_ignore_abil)

    # move_ignore_def = 0
    # if move.ignore_defensive:
    #     move_ignore_def = 1
    # feature_list.append(move_ignore_def)

    # move_ignore_evas = 0
    # if move.ignore_evasion:
    #     move_ignore_evas = 1
    # feature_list.append(move_ignore_evas)

    # move_ignore_immune = 0
    # if move.ignore_immunity:
    #     move_ignore_immune = 1
    # feature_list.append(move_ignore_immune)

    # move_is_prot_counter = 0
    # if move.is_protect_counter:
    #     move_is_prot_counter = 1
    # feature_list.append(move_is_prot_counter)

    move_is_prot_move = 0
    if move.is_protect_move:
        move_is_prot_move = 1
    feature_list.append(move_is_prot_move)

    move_priority = move.priority
    if move_priority:
        feature_list.append(move_priority)
    else:
        feature_list.append(0)
    
    move_recoil = move.recoil
    if move_recoil:
        feature_list.append(move_recoil)
    else:
        feature_list.append(0)

    move_self_boost = move.self_boost
    for attr in boosts:
        if move_self_boost and (attr in move_self_boost):
            feature_list.append(move_self_boost[attr])
        else:
            feature_list.append(0)

    move_is_sleep_usable = 0
    if move.sleep_usable:
        move_is_sleep_usable = 1
    feature_list.append(move_is_sleep_usable)

    move_status_eff = move.status
    for status in Status:
        if move_status_eff and (move_status_eff == status):
            feature_list.append(1)
        else:
            feature_list.append(0)
        
    move_steals_boosts = 0
    if move.steals_boosts:
        move_steals_boosts = 1
    feature_list.append(move_steals_boosts)

    move_terr = move.terrain
    for field in Field:
        if move_terr and (move_terr == field):
            feature_list.append(1)
        else:
            feature_list.append(0)

    move_type = move.type
    for type in PokemonType:
        if move_type and (move_type == type):
            feature_list.append(1)
        else:
            feature_list.append(0)
    
    move_weather = move.weather
    for weather in Weather:
        if move_weather and (move_weather == weather):
            feature_list.append(1)
        else:
            feature_list.append(0)

def PokemonModel(feature_list, pokemon, battle, abilities, moves, boosts, myTeam) -> None:

    #one hot encoded known abilities
    pkmn_ability = pokemon.ability
    if pkmn_ability == None:
        pkmn_ability = abilities[len(abilities)-1]
    for ability in abilities:
        if to_id_str(pkmn_ability) == ability:
            feature_list.append(1)
        else:
            feature_list.append(0)

    if not myTeam:
        pkmn_poss_abilities = pokemon.possible_abilities
        for ability in abilities:
            found_poss_ability = False
            for poss_ability in pkmn_poss_abilities:
                if to_id_str(poss_ability) == ability:
                    found_poss_ability = True
                    feature_list.append(1)
            if not found_poss_ability:
                feature_list.append(0)

    if myTeam:
        pkmn_stats = pokemon.base_stats
        pkmn_hp = pkmn_stats["hp"]
        feature_list.append(pkmn_hp/100)
        pkmn_atk = pkmn_stats["atk"]
        feature_list.append(pkmn_atk/100)
        pkmn_def = pkmn_stats["def"]
        feature_list.append(pkmn_def/100)
        pkmn_spa = pkmn_stats["spa"]
        feature_list.append(pkmn_spa/100)
        pkmn_spd = pkmn_stats["spd"]
        feature_list.append(pkmn_spd/100)
        pkmn_spe = pkmn_stats["spe"]
        feature_list.append(pkmn_spe/100)
    
    pkmn_boosts = pokemon.boosts
    for attr in boosts:
        if attr:
            feature_list.append(pkmn_boosts[attr])
        else:
            feature_list.append(0)
    
    pkmn_current_hp = pokemon.current_hp
    if pkmn_current_hp:
        feature_list.append(pkmn_current_hp/100)
    else:
        feature_list.append(1)

    pkmn_effects = pokemon.effects # OHE and process
    for effect in Effect:
        if pkmn_effects and effect in pkmn_effects:
            feature_list.append(pkmn_effects[effect])
        else:
            feature_list.append(0)

    pkmn_is_dyn = 0
    if pokemon.is_dynamaxed:
        pkmn_is_dyn = 1
    feature_list.append(pkmn_is_dyn)

    pkmn_preparing = 0
    if pokemon.preparing:
        pkmn_preparing = 1
    feature_list.append(pkmn_preparing)

    pkmn_protect_counter = pokemon.protect_counter
    if pkmn_protect_counter:
        feature_list.append(pkmn_protect_counter)
    else:
        feature_list.append(0)
    
    pkmn_status = pokemon.status
    for status in Status:
        if pkmn_status and pkmn_status == status:
            if pkmn_status == Status.SLP or pkmn_status == Status.TOX:
                feature_list.append(pokemon.status_counter)
            else:
                feature_list.append(1)
        else:
            feature_list.append(0)

    pkmn_type1 = pokemon.type_1
    pkmn_type2 = pokemon.type_2
    for type in PokemonType:
        if pkmn_type1 == type:
            feature_list.append(1)
        else:
            feature_list.append(0)
        if pkmn_type2 == type:
            feature_list.append(1)
        else:
            feature_list.append(0)

    if myTeam:
        pkmn_moves = battle.available_moves
        move_counter = 0
        for move in pkmn_moves:
            MoveModel(feature_list, move, abilities, moves, boosts)
            move_counter += 1

        while move_counter < 4:
            move_counter += 1
            MoveModel(feature_list, EmptyMove("unknown_move"), abilities, moves, boosts) # insert unknown move
        
        if move_counter > 4:
            print(move_counter)
            exit()

def GameModel(feature_list, battle, abilities, moves, boosts) -> None:
    player_can_dyna = 0
    if battle.can_dynamax:
        player_can_dyna = 1
    feature_list.append(player_can_dyna)
    
    player_maybe_trapped = 0
    if battle.maybe_trapped:
        player_maybe_trapped = 1
    feature_list.append(player_maybe_trapped)

    player_trapped = 0
    if battle.trapped:
        player_trapped = 1
    feature_list.append(player_trapped)

    player_force_switch = 0
    if battle.force_switch:
        player_force_switch = 1
    feature_list.append(player_force_switch)

    opponent_can_dyna = 0
    if battle.opponent_can_dynamax:
        opponent_can_dyna = 1
    feature_list.append(opponent_can_dyna)

    battle_weather = battle.weather
    for weather in Weather:
        if weather in battle_weather:
            feature_list.append(battle_weather[weather])
        else:
            feature_list.append(0)
    
    battle_fields = battle.fields
    for field in Field:
        if field in battle_fields:
            feature_list.append(battle_fields[field])
        else:
            feature_list.append(0)

    PokemonModel(feature_list, battle.active_pokemon, battle, abilities, moves, boosts, True)  

    for pokemon in battle.team.values():
        if not pokemon.active:
            pkmn_fainted = 0
            if pokemon.fainted:
                pkmn_fainted = 1
            feature_list.append(pkmn_fainted)

            pkmn_current_hp = pokemon.current_hp
            feature_list.append(pkmn_current_hp)

            pkmn_type1 = pokemon.type_1
            pkmn_type2 = pokemon.type_2
            for type in PokemonType:
                if pkmn_type1 == type:
                    feature_list.append(1)
                else:
                    feature_list.append(0)
                if pkmn_type2 == type:
                    feature_list.append(1)
                else:
                    feature_list.append(0)

    opp_pkmn = battle.opponent_active_pokemon
    PokemonModel(feature_list, opp_pkmn, battle, abilities, moves, boosts, False)

    remaining_mon_opponent = (
        len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6
    )

    feature_list.append(remaining_mon_opponent)


    # mon_counter = 0
    # for pokemon in battle.opponent_team.values():
    #     dbg_count += 1
    #     mon_counter += 1
    #     PokemonModel(feature_list, pokemon, abilities, moves, boosts, True)
    
    # while mon_counter < 6:
    #     dbg_count += 1
    #     mon_counter += 1
    #     PokemonModel(feature_list, Pokemon(), abilities, moves, boosts, False)
    
    # print(dbg_count)
    # if (mon_counter > 6):
    #     exit()

def SimpleGameModel(battle):
    moves_base_power = -np.ones(4)
    moves_dmg_multiplier = np.ones(4)
    for i, move in enumerate(battle.available_moves):
        moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
        if move.type:
            moves_dmg_multiplier[i] = move.type.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
                battle.opponent_active_pokemon.type_2,
            )

    # We count how many pokemons have not fainted in each team
    remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
    remaining_mon_opponent = (
        len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
    )

    # Final vector with 10 components
    return np.concatenate(
        [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
    )
