import time
import websockets
import asyncio
import requests
import json
from configparser import ConfigParser

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

    #challenge variables
    acceptingChallenges = True
    battle_format = None
    user_to_challenge = None
    room_name = None

    # create function creates an instance of a Websocket Client with default values
    @classmethod    # this command makes the next function bound to the class rather than a object instance
    async def create(cls):
        self = WebsocketClient()
        config_object = ConfigParser()
        config_object.read("config.ini")
        userinfo = config_object["USERINFO"]
        self.username = userinfo["username"]
        self.password = userinfo["password"]
        #challenge variables
        self.battle_format = userinfo["battle_format"]
        self.user_to_challenge = userinfo["user_to_challenge"]

        serverinfo = config_object["SERVERCONFIG"]
        self.address = serverinfo["address"]
        self.websocket = await websockets.connect(self.address)
        self.login_uri = serverinfo["host"]

        return self

    # receives and prints message from server
    async def listen(self):
        message = await self.websocket.recv()
        print("LOG:",message) # just printing debug messages for now, can move this to a dedicated debug log later
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
    
    # only works for gen8randombattle for now
    # after running main.py, log onto pokemon showdown website and challenge bot using default settings
    async def accept_challenge(self):
        if(self.acceptingChallenges):
            
            # These lines are from example project, and does nothing right now so it is commented out
            #if self.room_name is not None:
            #    await self.join_room(self.room_name)
            # logger.debug("Waiting for a {} challenge".format(battle_format))
            # await self.update_team(team) 

            username = None
            # msg = await self.listen()
            while username is None:
                print("Waiting for challenge")
                msg = await self.listen()
                split_msg = msg.split('|')

                #look for challenge and save user who sent challenge
                if (
                    len(split_msg) == 9 and
                    split_msg[1] == "pm" and
                    split_msg[3].strip() == "!" + self.username and
                    split_msg[4].startswith("/challenge") and
                    split_msg[5] == self.battle_format
                ):
                    username = split_msg[2].strip()

            message = ["/accept " + username]
            await self.send('', message)
            self.acceptingChallenges = False

    """async def challenge_user(self, user_to_challenge, battle_format, team):
        # logger.debug("Challenging {}...".format(user_to_challenge))
        if time.time() - self.last_challenge_time < 10:
            logger.info("Sleeping for 10 seconds because last challenge was less than 10 seconds ago")
            await asyncio.sleep(10)
        await self.update_team(team)
        message = ["/challenge {},{}".format(user_to_challenge, battle_format)]
        await self.send_message('', message)
        self.last_challenge_time = time.time()"""
    
    async def get_battle_tag(self):
        while True:
            msg = await self.listen()
            split_msg = msg.split('|')
            first_msg = split_msg[0]
            if 'battle' in first_msg:
                battle_tag = first_msg.replace('>', '').strip()
                return battle_tag
    
    async def take_turn(self, battle_tag):
        while True:
            msg = await self.listen()
            if "|turn" in msg:    
                move = "default"
                send_message = ["/choose " + move]
                await self.send(battle_tag, send_message)
            elif "|win" in msg or "|tie" in msg:
                return True
    
    async def battle(self):
        battle_tag = await self.get_battle_tag()
        while True:
            game_over = await self.take_turn(battle_tag)
            if game_over:
                return
