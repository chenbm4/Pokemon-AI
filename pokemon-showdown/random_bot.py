import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

async def random_bot():
    # Create a player that chooses random moves
    bot = RandomPlayer(player_configuration=PlayerConfiguration("Random Bot", None))
    
    print(bot.username)

    # Have the bot wait for challenges to accept
    await bot.accept_challenges(None, 2)

asyncio.get_event_loop().run_until_complete(random_bot())