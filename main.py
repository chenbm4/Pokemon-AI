import asyncio

# import constants
import config


from python_websockets.client import WebsocketClient

async def pokemon_ai(username, password): # move username and password to config file later
    websocket_client = await WebsocketClient.create(username, password)
    await websocket_client.login()
    searching = True
    while True:
        print("listening")
        msg = await websocket_client.listen()
        if(searching):
            searching = await websocket_client.accept_challenge(config.pokemon_mode, None, config.room_name, username)


# runs function pokemon_ai asynchronously until everything is complete
asyncio.get_event_loop().run_until_complete(pokemon_ai("carterbattlebot", "PacifidlogTown"))