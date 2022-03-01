import asyncio
import os

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from DamageCalculator import MaxDamagePlayer


# This is a simple helper function to handle user inputs forgivingly.
# This takes an iterable and returns an index that the user has chosen.
# -1 is never returned
def input_options(options):
    result = -1
    while (result not in range(len(options))): # Loop until result is good
        print("Select one of the following options:")
        for i in range(len(options)):
            print(str(i)+": " + str(options[i]))
        try:
            result = int(input("> "))
        except ValueError:
            print("Please enter a number")
            result = -1
    return result

# This is how the user selects 
#   which bot to use, 
#   what account to log into, 
#   which team to use, 
#   what battle format
#   what server
# returns tuple of bot, accountinfo, team, format, server object
# TODO move account information somewhere else for security
# could implement ini reading here
def bot_selector():

    bot_options = (RandomPlayer, MaxDamagePlayer)
    bot_options_strings = ("RandomPlayer", "MaxDamagePlayer")
    bot_index = input_options(bot_options_strings)
    bot = bot_options[bot_index]
    
    account_options = (("CarterBattleBot", "PacifidlogTown"), ("115ABot", "plasma"))
    account_options_strings = ("CarterBattleBot", "115ABot")
    account_index = input_options(account_options_strings)
    account = account_options[account_index]
	
    # currently irrelevant since no extant bots use teams
    team = None
    
    bformat_options = ("gen8randombattle",)
    bformat_options_strings = ("Gen 8 Random Battle",)
    bformat_index = input_options(bformat_options_strings)
    bformat = bformat_options[bformat_index]
    
    server_options = (ShowdownServerConfiguration, None)
    server_options_strings = ("Showdown Server", "Local Server")
    server_index = input_options(server_options_strings)
    server = server_options[server_index]

    return (bot, account, team, bformat, server)

async def main():
    selected = bot_selector()

    # We create a random player
    player = selected[0](
        player_configuration=PlayerConfiguration(selected[1][0], selected[1][1]),
        battle_format=selected[3],
        server_configuration=selected[4],
    )
    print("Successfully Logged In")
    # Sending challenges to 'your_username'
    # Not working on official pokemon showdown servers
    # await player.send_challenges("", n_challenges=1)

    # Accepting one challenge from any user
    # send challenge to bot under 'find a user' button and type in bot's username
    print("Waiting for challenge")
    await player.accept_challenges(None, 1)
    print("Battle finished")
    # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing 5 games on the ladder
    # UNTESTED
    # await player.ladder(5)

    # Print the rating of the player and its opponent after each battle
    #for battle in player.battles.values():
    #    print(battle.rating, battle.opponent_rating)

    # terminates after all battles are finished

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
