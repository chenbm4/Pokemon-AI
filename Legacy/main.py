import asyncio

from python_websockets.client import WebsocketClient
from states import State
 
async def pokemon_ai(): # move username and password to config file later
    websocket_client = await WebsocketClient.create()
    state = State.LOGIN
    await websocket_client.login()
    state = State.AWAITING_CHALLENGE
    while True:
        # print("listening")
        # msg = await websocket_client.listen()
        
        #wait for challenge to be sent to the bot from pokemon showdown
        if (state == State.AWAITING_CHALLENGE):
            await websocket_client.accept_challenge()
            state = State.BATTLE

        if (state == State.BATTLE):
            await websocket_client.battle()
            return

# runs function pokemon_ai asynchronously until everything is complete
asyncio.get_event_loop().run_until_complete(pokemon_ai())
