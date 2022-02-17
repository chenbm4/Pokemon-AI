import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration


async def random_bot():
    USERNAME = ""
    PASSWORD = ""
    NUM_CHALLENGES = 32

    # Create a player that chooses random moves
    bot = RandomPlayer(PlayerConfiguration("Random Bot", None))
    # bot = RandomPlayer(player_configuration=PlayerConfiguration(USERNAME, PASSWORD),
    #                    server_configuration=ShowdownServerConfiguration)
    
    print("Bot's Username: " + bot.username)

    # Have the bot wait for challenges to accept
    await bot.accept_challenges(None, NUM_CHALLENGES)

asyncio.get_event_loop().run_until_complete(random_bot())