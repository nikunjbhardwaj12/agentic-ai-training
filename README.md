# Task 01 — Personal Research Agent with LangGraph
A stateful conversational research agent built with LangGraph and the Groq API using the llama-3.3-70b-versatile model. It supports conversational question answering, web search, safe calculations, structured Markdown responses, and session-based memory.

## Features
- **Web Search** — Uses DuckDuckGo to fetch real-time information for user queries.

- **Calculator Tool** — Safely evaluates mathematical expressions.

- **Current Date Tool** — Returns today's date for time-sensitive prompts.

- **State Management** — Uses LangGraph StateGraph to control agent flow.

- **Session Memory** — Uses MemorySaver to preserve context during a chat session.

## Project Structure

assignments/task-01-langgraph-agent/  
│  
├── agent/  
│   ├── __init__.py      # Package initializer  
│   ├── graph.py         # Graph orchestration and conditional routing  
│   ├── nodes.py         # LLM and summarizer node logic  
│   ├── state.py         # AgentState TypedDict definition  
│   └── tools.py         # Built-in tools: search, calculator, date  
│  
├── .env                 # Local environment variables (ignored by git)  
├── .env.example         # Example environment configuration  
├── .gitignore           # Git ignore rules  
├── main.py              # CLI entry point  
├── README.md            # Project documentation  
└── requirements.txt     # Python dependencies  
## Setup
### 1. Create a virtual environment
*bash*  
python -m venv venv
### 2. Activate the environment
Windows (PowerShell)  

*bash*  
.\venv\Scripts\Activate.ps1
macOS / Linux

*bash*  
source venv/bin/activate
### 3. Install dependencies
*bash*  
pip install -r requirements.txt
### 4. Configure environment variables
Create a .env file in the project root and add your Groq API key:

text  
GROQ_API_KEY=your_actual_groq_api_key_here
Run the Project
Start the interactive CLI chatbot with:

*bash*  
python main.py
Example Usage
Ask a factual question

text
What is NASA?
Ask a follow-up question

text
Who founded it?
Exit Commands
Use any of the following commands to stop the program:

text
exit
quit
q
Notes
Make sure the .env file is present and contains a valid GROQ_API_KEY.

Conversation state is maintained during the active session using LangGraph memory.

This project is designed for CLI-based interaction and can be extended with additional tools or interfaces.