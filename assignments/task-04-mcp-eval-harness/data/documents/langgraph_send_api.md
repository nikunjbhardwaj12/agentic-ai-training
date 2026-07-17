# LangGraph: Send API and State Management

## The Send API

LangGraph's Send API enables dynamic, data-driven task execution loops. Rather than
statically defining every node a graph will visit ahead of time, a node can return
one or more `Send` objects at runtime, each specifying a target node and a distinct
input payload for that node.

This allows for dynamic routing: the number and destination of downstream tasks is
decided based on the current state, not hardcoded in the graph definition. A common
use case is map-reduce operations — for example, splitting a list of documents into
individual items at runtime and using Send to spawn one parallel node execution per
item, then merging (reducing) their outputs back together once all branches complete.

Because each Send call triggers independent, parallel node spawning, LangGraph can
fan out work dynamically based on the size of the input, then fan back in for
aggregation, without the graph author needing to know in advance how many parallel
tasks will be required.

## State Conservation in LangGraph Nodes

Every LangGraph node receives the current graph state and returns a partial update,
rather than mutating state in place. LangGraph applies these updates using
immutable state updates: each node's return value is merged into a new state object
according to the reducer defined for each state key, rather than being edited
directly.

For fields like the `messages` key (typically annotated with `add_messages`), this
means new messages are appended to the existing list rather than replacing it —
message appending, not overwriting.

State keys synchronization ensures that when multiple nodes update different parts
of the state, each key's reducer function determines how to combine the old and new
values consistently. Combined with thread checkpoints, which persist the state of a
graph run at each step under a given thread ID, this design lets LangGraph pause,
resume, replay, or fork a run of a graph at any point exactly as it was.
