import os
from dotenv import load_dotenv

# 1. Boot up the environment keys immediately before loading other files
load_dotenv()

# 2. Silence the Hugging Face symlinks logging noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import uuid
from rag.ingest import ingest_documents
from graph import graph

def main():
    print("[*] Checking for document indexing updating...")
    ingest_documents()

    query = input("\nEnter your search prompt/question: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    initial_state = {
        "query": query,
        "messages": [],
        "retrieved_docs": [],
        "approved": False,
        "draft": "",
        "final_answer": "",
        "next_agent": ""
    }

    current_inputs = initial_state
    
    while True:
        # Run the graph stream until it finishes or pauses after the researcher
        for event in graph.stream(current_inputs, config, stream_mode="values"):
            pass
        
        state_info = graph.get_state(config)
        current_values = state_info.values
        
        # If there are nodes left to run, we are at our interrupt checkpoint
        if state_info.next:
            docs = current_values.get("retrieved_docs", [])
            
            print("\n=== 🛑 HUMAN REVIEW REQUIRED ===")
            if docs:
                print(f"The researcher found {len(docs)} raw context chunks:")
                for idx, doc in enumerate(docs, 1):
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    print(f"  [{idx}] {content.strip()}")
            else:
                print("No context documents were found by the researcher.")
            print("================================\n")
            
            user_response = input("Approve these references? Enter 'y' (yes), 'n' (no), or 'r <new query>': ").strip()
            
            if user_response.lower() == 'y':
                state_update = {"approved": True}
            elif user_response.lower().startswith('r '):
                new_query = user_response[2:].strip()
                state_update = {"approved": False, "query": new_query, "retrieved_docs": []}
            else:
                print("References rejected. Stopping stream flow execution.")
                state_update = {"approved": False}
                
            # Update the state engine values explicitly
            graph.update_state(config, state_update)
            
            # Set current_inputs to None to tell the graph to resume from its current position
            current_inputs = None
        else:
            # The graph finished its workflow completely
            final_ans = current_values.get("final_answer") or current_values.get("draft")
            if final_ans:
                print("\n=== 📝 FINAL ANSWER OUTPUT ===")
                print(final_ans)
                print("==============================")
            else:
                print("\n[*] Flow complete. No output payload was generated.")
            break

if __name__ == "__main__":
    main()