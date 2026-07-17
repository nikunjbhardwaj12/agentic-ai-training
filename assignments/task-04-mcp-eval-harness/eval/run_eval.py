import os
import sys
import json
import time
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agent.graph import build_graph

load_dotenv()

# Lighter model for grading so it doesn't compete with the 70B agent model's
# daily token quota on Groq's free tier.
GRADER_MODEL = "llama-3.1-8b-instant"


def load_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path, "r") as f:
        return json.load(f)


def call_with_retry(fn, *args, max_retries=5, base_delay=30, **kwargs):
    """Retries a sync call on 429 rate-limit errors with linear backoff.
    Raises immediately on any other exception."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = base_delay * (attempt + 1)
                print(f"  \u23f3 Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Exceeded max retries due to rate limiting.")


async def acall_with_retry(fn, *args, max_retries=3, base_delay=5, **kwargs):
    """Async version of call_with_retry, for graph.ainvoke (MCP tools are async-only).
    Retries on rate limits AND on transient tool_use_failed generation errors —
    both can be intermittent and succeed on a retry."""
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = 30 * (attempt + 1)
                print(f"  \u23f3 Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(wait)
                continue
            if "tool_use_failed" in msg:
                print(f"  \u21bb Malformed tool call, retrying ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(base_delay)
                continue
            raise
    raise RuntimeError("Exceeded max retries.")


def run_grader(question: str, agent_output: str, expected_facts: list):
    """
    LLM-as-a-judge using a rubric.

    Scores:
    - Faithfulness (0-5)
    - Relevance (0-5)
    - Fact Coverage (0-5)

    Passing score: >= 12/15
    """

    api_key = os.getenv("GROQ_API_KEY")

    grader_llm = ChatGroq(
        model=GRADER_MODEL,
        temperature=0,
        groq_api_key=api_key,
    )

    facts = "\n".join(f"- {x}" for x in expected_facts)

    prompt = f"""
You are an impartial evaluation judge.

Evaluate the following response.

USER QUESTION:
{question}

AGENT RESPONSE:
{agent_output}

EXPECTED FACTS:
{facts}

Score the response using this rubric.

1. Faithfulness (0-5)
- Is the response grounded in correct information?
- Does it avoid unsupported claims?

2. Relevance (0-5)
- Does it answer the user's question?
- Does it stay on topic?

3. Fact Coverage (0-5)
- Does it include the important expected facts?

Return ONLY this format:

Faithfulness: <number>
Relevance: <number>
Coverage: <number>
Total: <number>

Do not explain your reasoning.
"""

    result = call_with_retry(grader_llm.invoke, prompt)

    text = result.content.strip()

    faithfulness = 0
    relevance = 0
    coverage = 0

    for line in text.splitlines():

        if line.lower().startswith("faithfulness"):
            faithfulness = int(line.split(":")[1].strip())

        elif line.lower().startswith("relevance"):
            relevance = int(line.split(":")[1].strip())

        elif line.lower().startswith("coverage"):
            coverage = int(line.split(":")[1].strip())

    total = faithfulness + relevance + coverage

    passed = (
        faithfulness >= 3
        and relevance >= 3
        and coverage >= 4   # 4/5 ≈ 80%, satisfies the ≥70% requirement
    )

    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "coverage": coverage,
        "total": total,
        "passed": passed,
    }


async def main():
    print("\U0001f680 Starting evaluation execution run against 12 test cases...")
    graph = await build_graph()
    test_cases = load_dataset()
    passed = 0
    total = len(test_cases)
    results = []

    for idx, tc in enumerate(test_cases, 1):
        question = tc["question"]
        expected_facts = tc["expected_facts"]

        print(f"\n[Test {idx}/{total}] Running: '{question}'")
        inputs = {"messages": [("user", question)]}
        config = {
            "configurable": {"thread_id": f"eval_thread_{idx}"},
            "metadata": {"mode": "eval", "question_id": tc["id"]},
        }

        try:
            output = await acall_with_retry(graph.ainvoke, inputs, config=config)
            agent_response = output["messages"][-1].content

            grading = run_grader( question, agent_response, expected_facts, )
            if grading["passed"]:
                passed += 1
                status = "PASSED"
            else:
                status = "FAILED"

            print(
                f"Faithfulness={grading['faithfulness']}/5 | "
                f"Relevance={grading['relevance']}/5 | "
                f"Coverage={grading['coverage']}/5 | "
                f"Total={grading['total']}/15"
            )
            print(f"Result: {status}")
            results.append({
                "question": question,
                "agent_response": agent_response,
                "status": status,
                "faithfulness": grading["faithfulness"],
                "relevance": grading["relevance"],
                "coverage": grading["coverage"],
                "score": grading["total"],
            })
        except Exception as e:
            print(f"Result: ERROR ({e})")
            import traceback
            traceback.print_exc()
            results.append({
                "question": question,
                "agent_response": f"Error during execution: {e}",
                "status": "ERROR",
                "faithfulness": 0,
                "relevance": 0,
                "coverage": 0,
                "score": 0,
            })
        # Small pacing gap to avoid bursty per-minute rate limits.
        await asyncio.sleep(3)

    accuracy = (passed / total) * 100
    report_path = os.path.join(os.path.dirname(__file__), "report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Execution Report\n\n")
        f.write(f"- **Total Tests:** {total}\n")
        f.write(f"- **Passed Tests:** {passed}\n")
        f.write(f"- **Final System Accuracy:** {accuracy:.2f}%\n\n")
        f.write("## Detailed Results\n\n")
        for idx, res in enumerate(results, 1):
            f.write(f"### Test {idx}: {res['question']}\n")
            f.write(f"- **Status:** {res['status']}\n")
            f.write(f"- **Faithfulness:** {res['faithfulness']}/5\n")
            f.write(f"- **Relevance:** {res['relevance']}/5\n")
            f.write(f"- **Coverage:** {res['coverage']}/5\n")
            f.write(f"- **Overall Score:** {res['score']}/15\n")
            f.write(f"- **Agent Response:**\n> {res['agent_response']}\n\n")

    print(f"\n\u2705 Evaluation complete. Accuracy: {accuracy:.2f}% (Report saved to: {report_path})")

    if accuracy < 80.0:
        print("\u274c System did not meet the 80% passing threshold.")
        sys.exit(1)
    else:
        print("\U0001f389 System exceeded the target threshold!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())