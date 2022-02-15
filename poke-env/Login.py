import asyncio

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from pokebot import MaxDamagePlayer

async def main():
    # We create a random player
    #logs in as user: 'CarterBattleBot'
    player = MaxDamagePlayer(
        player_configuration=PlayerConfiguration("CarterBattleBot", "PacifidlogTown"),
        battle_format="gen8randombattle",
        server_configuration=ShowdownServerConfiguration,
    )
    
    # Sending challenges to 'your_username'
    # Not working on official pokemon showdown servers
    # await player.send_challenges("TheMagnaCarta", n_challenges=1)

    # Accepting one challenge from any user
    # send challenge to bot under 'find a user' button and type in bot's username
    await player.accept_challenges(None, 1)

    # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing 5 games on the ladder
    # UNTESTED
    # await player.ladder(5)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)

    # terminates after all battles are finished

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())