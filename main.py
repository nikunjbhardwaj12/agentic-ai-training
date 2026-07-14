import asyncio
import os
import chromadb
from dotenv import load_dotenv

# Force load the .env file relative to this main.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

# Clear system cache to prevent Chroma DB lockups across runs
chromadb.api.client.SharedSystemClient.clear_system_cache()

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import builder, DB_PATH

async def run_local_cli():
    query = input("Ask a question to test Task 03: ")
    thread_id = "test_run_1"
    
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "original_query": query,
        "iteration": 0,
        "max_iterations": 3,
        "messages": []
    }
    
    print("\nStarting Plan-and-Execute pipeline...")
    
    # Open the checkpointer, compile, and stream the run
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        graph = builder.compile(checkpointer=memory)
        
        async for event in graph.astream_events(initial_state, config, version="v2"):
            node = event.get("metadata", {}).get("langgraph_node", "")
            event_type = event.get("event")
            
            if event_type == "on_chain_end" and node:
                output = event.get("data", {}).get("output", {})
                print(f"\n[{node.upper()} COMPLETED]")
                if "sub_tasks" in output:
                    print(f" -> Generated {len(output['sub_tasks'])} sub-tasks:")
                    for task in output['sub_tasks']:
                        print(f"    - [{task['id']}]: {task['question']}")
                if "evaluation" in output:
                    print(f" -> Faithfulness: {output['evaluation']['faithfulness']}/5")
                    print(f" -> Relevance: {output['evaluation']['relevance']}/5")
                    print(f" -> Feedback: {output['evaluation']['feedback']}")
                if "final_answer" in output:
                    print(f"\n[FINAL ANSWER]:\n{output['final_answer']}\n")

if __name__ == "__main__":
    asyncio.run(run_local_cli())