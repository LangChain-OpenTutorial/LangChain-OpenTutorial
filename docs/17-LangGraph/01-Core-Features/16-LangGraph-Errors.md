<style>
.custom {
    background-color: #008d8d;
    color: white;
    padding: 0.25em 0.5em 0.25em 0.5em;
    white-space: pre-wrap;       /* css-3 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;
}

pre {
    background-color: #027c7c;
    padding-left: 0.5em;
}

</style>

# Errors

- Author: [seofiled](https://github.com/seofield)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

- Understanding common error types is crucial for building and maintaining robust graph-based workflows. 
- This tutorial introduces key LangGraph errors, demonstrates scenarios that trigger them, and guides you through resolving each error. 
- By intentionally causing these errors and troubleshooting them step-by-step, you’ll gain a practical understanding of LangGraph’s error-handling mechanisms.

### Table of Contents

- [Overview](#overview)
- [GraphRecursionError](#graphrecursionerror)
- [InvalidUpdateError](#invalidupdateerror)
- [MultipleSubgraphsError](#multiplesubgraphserror)

### References

- [LangGraph API Reference: Errors](https://langchain-ai.github.io/langgraph/reference/errors/)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.


**[Note]**

The langchain-opentutorial is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
Check out the  [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain",
    ],
    verbose=False,
    upgrade=False,
)
```

## GraphRecursionError
- The `GraphRecursionError` error is raised when your LangGraph StateGraph exceeds the maximum number of steps during execution. 
- This safeguard prevents infinite loops caused by cyclic dependencies in your graph or overly complex graphs that naturally require many iterations.
- Below is a case where the graph loops infinitely and never ends. In this case, an `GraphRecursionError` prevents the infinite loop.


```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
from typing import TypedDict

# Define the state structure for the graph
class GraphState(TypedDict):
    input: str

# Define the first node
def node_1(state: GraphState):
    print("---Node 1---")
    return {"input": state["input"]}

# Define the second node
def node_2(state: GraphState):
    print("---Node 2---")
    return {"input": state["input"]}

# Build the graph
builder = StateGraph(GraphState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Define the edges between nodes
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_1")

# Compile the graph
graph = builder.compile()

# Execute the graph with a recursion limit to prevent infinite loops
try:
    graph.invoke({"input": "test"}, {"recursion_limit": 10})
except GraphRecursionError as e:
    print(f"GraphRecursionError occurred: {e}")
```

<pre class="custom">---Node 1---
    ---Node 2---
    ---Node 1---
    ---Node 2---
    ---Node 1---
    ---Node 2---
    ---Node 1---
    ---Node 2---
    ---Node 1---
    ---Node 2---
    GraphRecursionError occurred: Recursion limit of 10 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
    For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT
</pre>

If your graph is not an infinite loop and you set appropriate limits, it will work fine.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
from typing import TypedDict

# Define the state structure for the graph
class GraphState(TypedDict):
    input: str

# Define the first node
def node_1(state: GraphState):
    print("---Node 1---")
    return {"input": state["input"]}

# Define the second node
def node_2(state: GraphState):
    print("---Node 2---")
    return {"input": state["input"]}

# Build the graph
builder = StateGraph(GraphState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Define the edges between nodes
builder.add_edge(START, "node_1")  # Start the graph execution with node_1
builder.add_edge("node_1", "node_2")  # Connect node_1 to node_2
builder.add_edge("node_2", END)  # Connect node_2 to the END node

# Compile the graph
graph = builder.compile()

# Execute the graph with a recursion limit
try:
    # Invoke the graph with initial state and recursion limit
    result = graph.invoke({"input": "test"}, {"recursion_limit": 5})
except GraphRecursionError as e:
    print(f"GraphRecursionError occurred: {e}")
print(f"Execution succeeded: {result}")
```

<pre class="custom">---Node 1---
    ---Node 2---
    Execution succeeded: {'input': 'test'}
</pre>

## InvalidUpdateError

The `InvalidUpdateError` is an error that occurs when a channel is updated with an invalid or incompatible set of updates. This typically indicates a mismatch between the updates being applied and the expected format or behavior of the graph.

### INVALID_CONCURRENT_GRAPH_UPDATE

- This error is triggered when multiple nodes concurrently update the same state property in a StateGraph, and no mechanism exists to resolve these conflicts.
- Parallel execution (e.g., fanout) causes multiple nodes to update the same state key.
- LangGraph cannot determine how to merge these updates, resulting in the error.



```python
from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define the state structure for the graph
class State(TypedDict):
    input: str

# Define the first node
def node(state: State):
    return {"input": "value_from_node"}

# Define the second node
def other_node(state: State):
    return {"input": "value_from_other_node"}

# Build the graph
builder = StateGraph(State)
builder.add_node("node", node)
builder.add_node("other_node", other_node)

# Set up parallel execution
builder.add_edge(START, "node")  # Start execution with node
builder.add_edge(START, "other_node")  # Start execution with other_node in parallel
builder.add_edge("node", END)
builder.add_edge("other_node", END)

# Compile the graph
graph = builder.compile()

# Execute the graph
try:
    graph.invoke({"input": ""})
except InvalidUpdateError as e:
    print(f"InvalidUpdateError: {e}")
```

<pre class="custom">InvalidUpdateError: At key 'input': Can receive only one value per step. Use an Annotated key to handle multiple values.
    For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE
</pre>

Use a Reducer to Handle Conflicts

- Reducers are essential for resolving conflicts when multiple nodes in a graph attempt to update the same state key during parallel execution. They define how to combine these updates, ensuring smooth and conflict-free operation.

```python
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# Define the state structure with a reducer
class State(TypedDict):
    # Use a reducer to merge updates by appending to the list
    input: Annotated[list, operator.add]

# Define the first node
def node(state: State):
    return {"input": ["value_from_node"]}

# Define the second node
def other_node(state: State):
    return {"input": ["value_from_other_node"]}

# Build the graph
builder = StateGraph(State)
builder.add_node("node", node)
builder.add_node("other_node", other_node)

# Set up parallel execution
builder.add_edge(START, "node")  # Start execution with node
builder.add_edge(START, "other_node")  # Start execution with other_node in parallel
builder.add_edge("node", END)
builder.add_edge("other_node", END)

# Compile the graph
graph = builder.compile()

# Execute the graph
try:
    result = graph.invoke({"input": []})
except InvalidUpdateError as e:
    print(f"InvalidUpdateError: {e}")
print("Execution succeeded:", result)
```

<pre class="custom">Execution succeeded: {'input': ['value_from_node', 'value_from_other_node']}
</pre>

### INVALID_GRAPH_NODE_RETURN_VALUE

The `INVALID_GRAPH_NODE_RETURN_VALUE` error in LangGraph occurs when a node in a StateGraph returns a value that is not a dictionary. Every node in your graph must return a dictionary with one or more keys that match the defined state schema.




```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import InvalidUpdateError

# Define the state structure for the graph
class State(TypedDict):
    input: str

# Define a node with an invalid return type
def bad_node(state: State):
    # Incorrectly returns a list instead of a dictionary
    return ["bad_node"]

# Build the graph
builder = StateGraph(State)
builder.add_node("bad_node", bad_node)

# Set up the edges
builder.add_edge(START, "bad_node")
builder.add_edge("bad_node", END)

# Compile the graph
graph = builder.compile()

# Execute the graph
try:
    graph.invoke({"input": "test"})
except InvalidUpdateError as e:
    print(f"InvalidUpdateError: {e}")
```

<pre class="custom">InvalidUpdateError: Expected dict, got ['bad_node']
    For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE
</pre>

Ensure the Node Returns a Dictionary

- Nodes must always return a dictionary matching the state schema.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Define the state structure for the graph
class State(TypedDict):
    input: str

# Define a valid node
def fixed_node(state: State):
    # Returns a dictionary with the corrected input
    return {"input": state["input"] + " corrected"}

# Build the graph
builder = StateGraph(State)

# Add the corrected node to the graph
builder.add_node("fixed_node", fixed_node)

# Define the edges
builder.add_edge(START, "fixed_node")
builder.add_edge("fixed_node", END)

# Compile the graph
graph = builder.compile()

# Define the initial state
initial_state = {"input": "test"}

# Execute the graph
try:
    result = graph.invoke({"input": "test"})
except InvalidUpdateError as e:
    print(f"InvalidUpdateError: {e}")
print("Execution succeeded:", result)
```

<pre class="custom">Execution succeeded: {'input': 'test corrected'}
</pre>

## MultipleSubgraphsError

The `MultipleSubgraphsError` occurs when multiple subgraphs are invoked within a single LangGraph node, particularly when checkpointing is enabled for each subgraph. This restriction exists due to how LangGraph handles checkpoint namespacing for subgraphs.

```python
from langgraph.errors import MultipleSubgraphsError
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid

# Define the state structure for the subgraphs
class SubGraphState(TypedDict):
    value: int

# Define the first subgraph node
def subgraph_node_1(state: SubGraphState):
    return {"value": state["value"] + 1}

# Define the second subgraph node
def subgraph_node_2(state: SubGraphState):
    return {"value": state["value"] * 2}

# Build the first subgraph
subgraph_builder_1 = StateGraph(SubGraphState)
subgraph_builder_1.add_node("subgraph_node_1", subgraph_node_1)
subgraph_builder_1.add_edge(START, "subgraph_node_1")
subgraph_builder_1.add_edge("subgraph_node_1", END)

checkpointer_1 = MemorySaver()
subgraph_1 = subgraph_builder_1.compile(checkpointer=checkpointer_1)

# Build the second subgraph
subgraph_builder_2 = StateGraph(SubGraphState)
subgraph_builder_2.add_node("subgraph_node_2", subgraph_node_2)
subgraph_builder_2.add_edge(START, "subgraph_node_2")
subgraph_builder_2.add_edge("subgraph_node_2", END)

checkpointer_2 = MemorySaver()
subgraph_2 = subgraph_builder_2.compile(checkpointer=checkpointer_2)

# Define the main graph state structure
class MainGraphState(TypedDict):
    value: int
    result: int

# Define the main graph node
def main_node(state: MainGraphState):
    # Call two subgraphs within a single node
    subgraph_output_1 = subgraph_1.invoke({"value": state["value"]})
    subgraph_output_2 = subgraph_2.invoke({"value": state["value"]})
    return {"result": subgraph_output_1["value"] + subgraph_output_2["value"]}

# Build the main graph
main_builder = StateGraph(MainGraphState)
main_builder.add_node("main_node", main_node)
main_builder.add_edge(START, "main_node")
main_builder.add_edge("main_node", END)

main_checkpointer = MemorySaver()
main_graph = main_builder.compile(checkpointer=main_checkpointer)

# Configure and execute the graph
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}

try:
    result = main_graph.invoke({"value": 5, "result": 0}, config)
    
except MultipleSubgraphsError as e:
    print(f"MultipleSubgraphsError: {e}")
```

<pre class="custom">MultipleSubgraphsError: Multiple subgraphs called inside the same node
    
    Troubleshooting URL: https://python.langchain.com/docs/troubleshooting/errors/MULTIPLE_SUBGRAPHS/
</pre>

Disable Checkpointing for Subgraphs

- If you do not need to interrupt or resume the subgraphs, disable checkpointing by compiling them with checkpointer=False:
- Best Practices for Avoiding `MultipleSubgraphsError`
  1.	Plan Subgraph Execution: Avoid invoking multiple subgraphs imperatively within a single node. Use modular design with separate nodes.
  2.	Use Checkpointers Judiciously: Only enable checkpointing for subgraphs where interruption and resumption are critical.
  3.	Debug Graph Logic: Visualize your graph or use debugging tools to understand where subgraph calls are being made.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid

# Define the state structure for the subgraphs
class SubGraphState(TypedDict):
    value: int

# Define the first subgraph node
def subgraph_node_1(state: SubGraphState):
    return {"value": state["value"] + 1}

# Define the second subgraph node
def subgraph_node_2(state: SubGraphState):
    return {"value": state["value"] * 2}

# Build the first subgraph
subgraph_builder_1 = StateGraph(SubGraphState)
subgraph_builder_1.add_node("subgraph_node_1", subgraph_node_1)
subgraph_builder_1.add_edge(START, "subgraph_node_1")
subgraph_builder_1.add_edge("subgraph_node_1", END)
subgraph_1 = subgraph_builder_1.compile(checkpointer=False)  # Disable checkpointing

# Build the second subgraph
subgraph_builder_2 = StateGraph(SubGraphState)
subgraph_builder_2.add_node("subgraph_node_2", subgraph_node_2)
subgraph_builder_2.add_edge(START, "subgraph_node_2")
subgraph_builder_2.add_edge("subgraph_node_2", END)
subgraph_2 = subgraph_builder_2.compile(checkpointer=False)  # Disable checkpointing

# Define the state structure for the main graph
class MainGraphState(TypedDict):
    value: int
    result: int

# Define the main graph node
def main_node(state: MainGraphState):
    subgraph_output_1 = subgraph_1.invoke({"value": state["value"]})
    subgraph_output_2 = subgraph_2.invoke({"value": state["value"]})
    return {"result": subgraph_output_1["value"] + subgraph_output_2["value"]}

# Build the main graph
main_builder = StateGraph(MainGraphState)
main_builder.add_node("main_node", main_node)
main_builder.add_edge(START, "main_node")
main_builder.add_edge("main_node", END)

# Compile the main graph
main_graph = main_builder.compile(checkpointer=MemorySaver())

# Config
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),  # Unique thread identifier for execution
    }
}

try:
    result = main_graph.invoke({"value": 5, "result": 0}, config)
except MultipleSubgraphsError as e:
    print(f"MultipleSubgraphsError: {e}")
print("Execution succeeded:", result)
```

<pre class="custom">Execution succeeded: {'value': 5, 'result': 16}
</pre>
