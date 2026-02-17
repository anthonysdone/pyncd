import asyncio
import websocket_transfer.websockets_transfer as wst

server = wst.DataServer()
asyncio.run(server.main())