import asyncio

# import constants
import config


from python_websockets.client import WebsocketClient
 
async def pokemon_ai(): # move username and password to config file later
    websocket_client = await WebsocketClient.create()
    await websocket_client.login()
    while True:
        print("listening")
        msg = await websocket_client.listen()
        
        #wait for challenge to be sent to the bot from pokemon showdown
        await websocket_client.accept_challenge(config.pokemon_mode, None, config.room_name)


# runs function pokemon_ai asynchronously until everything is complete
asyncio.get_event_loop().run_until_complete(pokemon_ai())
