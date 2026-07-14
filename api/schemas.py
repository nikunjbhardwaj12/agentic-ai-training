from pydantic import BaseModel

class ResearchRequest(BaseModel):
    query: str
    thread_id: str