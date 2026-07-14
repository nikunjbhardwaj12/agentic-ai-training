from state import PlanExecuteState

def aggregator_node(state: PlanExecuteState) -> dict:
    raw_docs = state.get("retrieved_docs", [])
    
    # Simple deduplication: strip whitespace and keep unique strings
    unique_chunks = []
    seen = set()
    for doc in raw_docs:
        normalized = doc.strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_chunks.append(normalized)

    return {"aggregated_context": unique_chunks}