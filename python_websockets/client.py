
import websockets
import asyncio
import requests
import json

class LoginError(Exception):
    pass

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
        self.login_uri = "https://play.pokemonshowdown.com/action.php"
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

    # PS servers will send the following string to the client:
    #   "|challstr|(client ID)|(challstr)"
    # In order to log in, we need both client ID and challstr, so this function returns both
    async def get_id_and_challstr(self):
        while True:
            message = await self.listen()
            split_message = message.split('|')
            if split_message[1] == 'challstr':
                return split_message[2], split_message[3]

    # attempts to log the user in using the stored username and password
    async def login(self):
        print("Logging in as " + self.username)
        client_id, challstr = await self.get_id_and_challstr()

        # according to PS github, must make an HTTP POST request to the login_uri
        # with the data act=login&name=USERNAME&pass=PASSWORD&challstr=CHALLSTR
        # requests.post essentially does that for us
        response = requests.post(
            self.login_uri,
            data={
                'act': 'login',
                'name': self.username,
                'pass': self.password,
                'challstr': "|".join([client_id, challstr])
            }
        )

        # status code 200 means the HTTP request has gone through successfully
        if response.status_code == 200:
            # json helps to format the response text in an easily-parsable way for us to use
            response_json = json.loads(response.text[1:])

            if not response_json['actionsuccess']:
                print("Login Unsuccessful")
                raise LoginError("Could not log-in")

            # to finish logging in, must send /trn USERNAME,0,ASSERTION where USERNAME is your desired username 
            # and ASSERTION is data.assertion.

            assertion = response_json.get('assertion')
            message = ["/trn " + self.username + ",0," + assertion]
            print("Successfully logged in")
            await self.send('', message)
        else:
            print.error("Could not log-in\nDetails:\n{}".format(response.content))
            raise LoginError("Could not log-in")