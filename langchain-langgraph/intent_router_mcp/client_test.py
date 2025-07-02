import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession

async def main():
    async with stdio_client("./mcp_server.py") as transport:
        session = ClientSession(transport=transport)
        await session.initialize()

        result1 = await session.call_tool("Classify Intent", {"input": "I'm feeling anxious and overwhelmed."})
        print("Category:", result1.result)

        result2 = await session.call_tool("Mental Health Response", {"input": "I'm feeling anxious and overwhelmed."})
        print("Mental Agent Response:", result2.result)

asyncio.run(main())
