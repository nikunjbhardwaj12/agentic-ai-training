import uuid
from agent.graph import graph
from langchain_core.messages import HumanMessage

def main():
    print("==========================================================")
    print("🔬 LangGraph Personal Research Agent CLI initialized (Groq)")
    print("==========================================================")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")
    
    # Create a unique thread ID to handle multi-turn chat persistence context
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nAgent: Great! Happy to help whenever you need more research. Goodbye! 👋")
                break
            
            # Execute state machine via structured updates
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "topic": user_input,
                "search_results": [],
                "final_summary": ""
            }
            
            print("\n[Agent Thinking...]")
            
            # Stream events out to watch execution steps inside terminal
            for event in graph.stream(inputs, config, stream_mode="values"):
                # Get the last message in the sequence
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    # Print tool calls visually if they emerge from intermediate state steps
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            print(f" ⚙️  [Calling Tool]: '{tc['name']}' with params {tc['args']}")

            # Final outputs evaluation
            state = graph.get_state(config)
            final_messages = state.values.get("messages", [])
            
            if final_messages:
                print(f"\nAgent:\n{final_messages[-1].content}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()