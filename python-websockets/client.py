
import websockets
import asyncio

# connects and receives a message from the Showdown server
async def listen():
    url = "ws://sim.smogon.com:8000/showdown/websocket"

    async with websockets.connect(url) as ws:
        msg = await ws.recv()
        print(msg)

# runs function listen asynchronously until everything is complete
asyncio.get_event_loop().run_until_complete(listen())
