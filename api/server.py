import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Force load the .env file located in the parent directory of api/
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(os.path.dirname(current_dir), ".env")
load_dotenv(dotenv_path=env_path)

from api.schemas import ResearchRequest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import builder, DB_PATH

app = FastAPI(title="Task 03 Multi-Agent Plan & Execute API")

@app.post("/research")
async def research(body: ResearchRequest):
    initial_state = {
        "original_query": body.query,
        "iteration": 0,
        "max_iterations": 3,
        "messages": []
    }
    config = {"configurable": {"thread_id": body.thread_id}}

    async def event_stream():
        # Open checkpointer connection dynamically on call
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
            graph = builder.compile(checkpointer=memory)
            
            async for event in graph.astream_events(initial_state, config, version="v2"):
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                event_type = event.get("event")
                
                # Emit events in real-time matching requested SSE formats
                if event_type == "on_chain_end" and node_name:
                    output = event.get("data", {}).get("output", {})
                    if node_name == "planner_node" and "sub_tasks" in output:
                        yield f"data: {json.dumps({'type': 'planner', 'sub_tasks': output['sub_tasks']})}\n\n"
                    elif node_name == "researcher_node" and "retrieved_docs" in output:
                        yield f"data: {json.dumps({'type': 'researcher', 'chunks_found': len(output['retrieved_docs'])})}\n\n"
                    elif node_name == "aggregator_node" and "aggregated_context" in output:
                        yield f"data: {json.dumps({'type': 'aggregator', 'after_dedup': len(output['aggregated_context'])})}\n\n"
                    elif node_name == "writer_node" and "draft" in output:
                        yield f"data: {json.dumps({'type': 'writer', 'draft_length': len(output['draft'])})}\n\n"
                    elif node_name == "evaluator_node" and "evaluation" in output:
                        yield f"data: {json.dumps({'type': 'evaluator', **output['evaluation'], 'iteration': output.get('iteration', 1)})}\n\n"
                    elif node_name == "accept_final_node" and "final_answer" in output:
                        yield f"data: {json.dumps({'type': 'final', 'answer': output['final_answer']})}\n\n"
                await asyncio.sleep(0.05)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/research/{thread_id}")
async def get_research_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Retrieve checkpoints directly to inspect pipeline states safely
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        graph = builder.compile(checkpointer=memory)
        state = await graph.aget_state(config)
        if not state.values:
            return {"status": "error", "message": "Thread not found"}
        
        is_running = state.next != ()
        return {
            "status": "running" if is_running else "complete",
            "state": state.values
        }