
import websockets
import asyncio

# the WebsocketClient class we will be using to interface with PS servers
class WebsocketClient:
    websocket = None
    # address = None    not sure why the other guy made this a variable since it should stay the same so commenting out for now
    login_uri = None
    username = None
    password = None
    last_message = None

    # create function creates an instance of a Websocket Client with default values
    @classmethod    # this command makes the next function bound to the class rather than a object instance
    async def create(cls, username, password):
        self = WebsocketClient()
        self.username = username
        self.password = password
        self.address = "ws://sim.smogon.com:8000/showdown/websocket"
        self.websocket = await websockets.connect(self.address)
        self.login_uri = "https://play.pokemonshowdown.com/action.php"   # not sure what this is for yet
        return self

    # receives and prints message from server
    async def listen(self):
        message = await self.websocket.recv()
        print(message) # just printing debug messages for now, can move this to a dedicated debug log later
        return message

    # sends and prints message to server
    async def send(self, room_id, messages):
        # client -> server message protocol: ROOMID|TEXT
        # more details here: https://github.com/smogon/pokemon-showdown/blob/master/PROTOCOL.md
        formatted_message = room_id + "|" + "|".join(messages)
        print(formatted_message)
        await self.websocket.send(formatted_message)
        self.last_message = formatted_message
