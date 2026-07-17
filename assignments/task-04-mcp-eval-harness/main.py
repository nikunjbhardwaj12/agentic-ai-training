import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import build_graph

load_dotenv()

async def main():
    print("🤖 MCP Multi-Server Command Line Interface Online (Type 'exit' to quit).")
    print("----------------------------------------------------------------------")
    graph = await build_graph()

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() == "exit":
                break

            if not user_input.strip():
                continue

            config = {"metadata": {"mode": "interactive"}}
            async for event in graph.astream({"messages": [HumanMessage(content=user_input)]}, config=config):
                for node, data in event.items():
                    if "messages" in data:
                        last_msg = data["messages"][-1]
                        if last_msg.content:
                            print(f"\n[{node.upper()}]: {last_msg.content}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn anomaly occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())