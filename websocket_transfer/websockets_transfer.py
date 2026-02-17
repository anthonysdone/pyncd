import asyncio
from dataclasses import dataclass, field
import websockets
from typing import TypedDict, Literal, Any
import json
import data_transfer.json as dtj
import data_structure.Term as fd
import random


type Websocket = Any

'''
The design is as follows:
 - We have a SERVER. The SERVER runs independently from Jupyter Notebook.
 - A CLIENT connects to the SERVER via WebSocket, and can send messages to it.
 - TypeScript Browser connects to the SERVER. It then receives the latest term from the CLIENT.

'''

type Message = HandshakeMessage | DataUpdate | DataRequest | GenericMessage

class HandshakeMessage(TypedDict):
    msgType: Literal['identify']
    clientType: Literal['DiagramClient', 'DataClient']
    clientVersion: str
    clientID: str

class DataUpdate(TypedDict):
    msgType: Literal['dataUpdate']
    data: dtj.JSONDataStructure

class DataRequest(TypedDict):
    msgType: Literal['dataRequest']

class GenericMessage(TypedDict):
    msgType: str

@dataclass
class HandlerInformation:
    socket: Websocket
    clientType: Literal['DiagramClient', 'DataClient'] | None = None

@dataclass
class DataServer:
    data_structure: dtj.JSONDataStructure | None = None
    diagram_clients: dict[str, Websocket] = field(default_factory=dict)
    data_clients: dict[str, Websocket] = field(default_factory=dict)
    message_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    connected_clients: dict[int, HandlerInformation] = field(default_factory=dict)

    async def handler(self, websocket):
        print('Client connected.')

        randomKey = random.randint(0, 2**12)
        handlerInformation = HandlerInformation(socket=websocket)
        self.connected_clients[randomKey] = handlerInformation

        try:
            async for message in websocket:
                handlerInformation, response = await self.process_message(
                    json.loads(message), 
                    handlerInformation)
                self.connected_clients[randomKey] = handlerInformation
                await self.send_to_one(websocket, json.dumps(response))
        finally:
            del self.connected_clients[randomKey]
            print('Client disconnected.')

    async def send_to_one(self, client, message: str):
        await client.send(message)

    async def send_to_diagrams(self, message: str):
        for client in self.connected_clients.values():
            if client.clientType == 'DiagramClient':
                await client.socket.send(message)

    async def process_message(self, 
            msg: Message,
            handlerInformation: HandlerInformation
        ) -> tuple[HandlerInformation, Message]:
        match msg:
            case {'msgType': 'identify', 'clientType': clientType, 'clientVersion': clientVersion, 'clientID': clientID}:
                print(f"Client identified: {clientType} v{clientVersion} (ID: {clientID})")
                handlerInformation.clientType = clientType
                if clientType == 'DiagramClient' and self.data_structure is not None:
                    print('Sending data.')
                    await self.send_to_one(
                        handlerInformation.socket,
                        json.dumps({
                            'msgType': 'dataUpdate',
                            'data': self.data_structure
                        })
                    )
                return handlerInformation, {'msgType': 'Connected'}
            case {'msgType': 'dataUpdate', 'data': data}:
                self.data_structure = data
                print('Data Updated.')
                await self.send_to_diagrams(
                    json.dumps({
                        'msgType': 'dataUpdate',
                        'data': data
                    })
                )
                return handlerInformation, {'msgType': 'DataReceived'}
            case {'msgType': 'dataRequest'}:
                print('Data Requested.')
                if self.data_structure is not None:
                    return handlerInformation, {
                        'msgType': 'dataUpdate',
                        'data': self.data_structure
                    }
                else:
                    return handlerInformation, {'msgType': 'No Data Available'}
            case _:
                raise ValueError('Unknown message type: ' + str(msg))

    async def worker(self):
        while True:
            message = await self.message_queue.get()
            # for client in self.connected_clients:
            #     await client.send(message)

    async def main(self):
        async with websockets.serve(self.handler, "localhost", 8765):
            print("Server started at ws://localhost:8765")
            await self.worker()

@dataclass
class DataClient:
    handshake: str
    data: str

    @classmethod
    async def template(cls, term: fd.GeneralTerm):
        handshake: HandshakeMessage = {
            'msgType': 'identify',
            'clientType': 'DataClient',
            'clientVersion': '0.1.0',
            'clientID': 'unique-client-id-1234'
        }
        data = dtj.TermJSONConverter.export_to_json(term)
        client = cls(
            handshake=json.dumps(handshake),
            data=data)
        await client.main()

    async def main(self):
        try:
            async with websockets.connect("ws://localhost:8765") as websocket:
                await websocket.send(self.handshake)
                connected = await websocket.recv()
                print(f'Received from server: {connected}')
                await websocket.send(json.dumps({
                    'msgType': 'dataUpdate',
                    'data': self.data
                }))
                response = await websocket.recv()
                print(f"Received from server: {response}")
        except Exception as e:
            print(f"Be sure to execute `python run_server.py` before running this client. An error occurred: {e}")

async def send_term(term: fd.GeneralTerm):
    await DataClient.template(term)

# asyncio.run(main())
# print('end.')

if __name__ == '__main__':
    server = DataServer()
    asyncio.run(server.main())