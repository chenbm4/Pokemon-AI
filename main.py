import asyncio

from python_websockets.client import WebsocketClient

async def pokemon_ai(): # move username and password to config file later
    websocket_client = await WebsocketClient.create()
    await websocket_client.login()

    while True:
        await websocket_client.listen()

# runs function pokemon_ai asynchronously until everything is complete
asyncio.get_event_loop().run_until_complete(pokemon_ai())