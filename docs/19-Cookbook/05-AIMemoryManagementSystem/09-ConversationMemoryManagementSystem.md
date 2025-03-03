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

# ConversationMemoryManagementSystem

- Author: [syshin0116](https://github.com/syshin0116)
- Design: 
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/05-AIMemoryManagementSystem/09-ConversationMemoryManagementSystem.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/05-AIMemoryManagementSystem/09-ConversationMemoryManagementSystem.ipynb)

## Overview

In modern AI systems, **memory management** is essential for crafting **personalized and context-aware** user experiences. Without the ability to recall prior messages, an AI assistant would quickly become repetitive and less engaging. This updated code demonstrates a robust approach to handling both **short-term** and **long-term** memory in a conversational setting, by integrating:

- A central `Configuration` class for managing runtime parameters (such as `user_id` and model name)
- An `upsert_memory` function for **storing** or **updating** user data in a memory store
- A `call_model` function that **retrieves** context-relevant memories and incorporates them into the system prompt for the model
- A `store_memory` function that **persists** newly identified memories and tool calls
- A `StateGraph` that orchestrates the entire conversation flow, connecting nodes like `call_model` and `store_memory` to streamline user interactions

By leveraging these components, your conversation agent can maintain **deep context** over multiple turns, provide more accurate and engaging responses, and seamlessly update its memory when new information arises. This design illustrates a scalable way to build conversational AI systems that dynamically **remember**, **reason**, and **respond** according to user needs.

### What is Memory?

Memory refers to the capability of an AI system to **store**, **retrieve**, and **use** information. In conversational AI, this typically involves recalling the user’s previous statements, preferences, or relevant context—leading to more **personalized** and **adaptive** interactions.

### Short-term Memory

Short-term memory lets your application remember previous interactions within a single thread or conversation. A thread organizes multiple interactions in a session, similar to the way email groups messages in a single conversation.

LangGraph manages short-term memory as part of the agent's state, persisted via thread-scoped checkpoints. This state can normally include the conversation history along with other stateful data, such as uploaded files, retrieved documents, or generated artifacts. By storing these in the graph's state, the bot can access the full context for a given conversation while maintaining separation between different threads.

Since conversation history is the most common form of representing short-term memory, in the next section, we will cover techniques for managing conversation history when the list of messages becomes long.

### Long-term Memory

Long-term memory extends an AI system's capability to recall information across multiple conversations or sessions. Unlike short-term memory, which focuses on maintaining the context of a single thread, long-term memory stores **persistent information** such as user preferences, key facts, and important events. This enables the system to create a **seamless and personalized user experience** over time.

Long-term memory is typically used to:

- **Recall User Preferences**: For example, remembering a user prefers movie recommendations in a specific genre.
- **Track Progress**: Storing milestones or progress made in ongoing projects or discussions.
- **Adapt Over Time**: Learning about the user’s changing interests or requirements.

In this system, long-term memory is managed through **memory upserts** that save critical user information to a persistent data store. This allows the agent to access, update, and retrieve information beyond a single conversational thread.

### How Long-term Memory Works in LangGraph

In LangGraph, long-term memory is implemented as part of a **persistent data layer**, decoupled from the conversational state. Key components include:

1. **Memory Store**: A database or key-value store where long-term memory records are saved.
2. **Memory Retrieval**: Mechanisms to fetch relevant memories based on the current conversation context.
3. **Memory Updates**: Processes to modify or append new data to existing memory entries when user information evolves.

By linking long-term memory with `call_model` and `store_memory` functions, the system can ensure that the language model has access to **relevant, long-term context**. This makes interactions more coherent and reduces repetitive queries.

#### Challenges with Long-term Memory

- **Scalability**: As the number of users and stored records grows, querying and retrieving relevant memories efficiently becomes a challenge.
- **Relevance Filtering**: Not all past information is useful for every interaction. Filtering out irrelevant data while retaining critical context is key.
- **Data Privacy**: Long-term memory must comply with privacy regulations, ensuring sensitive data is securely handled and stored.


### Short-term vs Long-term Memory

|**Type**|**Scope**|**Purpose**|**Example**|
|---|---|---|---|
|**Short-term Memory**|Single conversational thread|Maintains recent interactions to provide immediate context|Last few user prompts in the current conversation|
|**Long-term Memory**|Multiple threads & sessions|Stores key information and summarized data to maintain continuity across broader conversations|User preferences, important facts, or conversation history|

- **Short-term Memory**: Helps the system focus on the latest messages for immediate context.
- **Long-term Memory**: Enables the agent to recall **past sessions** and user-specific details, creating a more **persistent** experience over time.

### Why use LangGraph's checkpointer?

1. Session Memory & Error Recovery
    
    - Lets you roll back to a previous checkpoint if an error occurs or if you want to resume from a saved state
    - Maintains context across conversations for a more seamless user experience
2. Flexible Database Options & Scalability
    
    - Supports in-memory, SQLite, Postgres, and more, allowing easy scaling as your user base grows
    - Choose the storage method that best fits your project’s needs
3. Human-in-the-Loop & Time Travel
    
    - Pause workflows for human review, then resume where you left off
    - Go back to earlier states (“time travel”) to debug or create alternative paths
4. Ecosystem & Customization
    
    - `LangGraph` v0.2 offers separate checkpointer libraries (e.g., `MemorySaver`, `SqliteSaver`, `PostgresSaver`)
    - Easily build or adapt custom solutions for specific databases or workflows

## Table of Contents

- [Overview](#overview)
    
- [Table of Contents](#table-of-contents)
  
- [Environment Setup](#environment-setup)
    
- [Define System Prompt and Configuration](#define-system-prompt-and-configuration)
    
- [Initialize LLM and Define State Class](#initialize-llm-and-define-state-class)
    
- [Memory Upsert Function](#memory-upsert-function)
    
- [Implement Conversation Flow (call_model, store_memory)](#implement-conversation-flow-call_model-store_memory)
    
- [Define Conditional Edge Logic](#define-conditional-edge-logic)
    
- [Build and Execute StateGraph](#build-and-execute-stategraph)
    
- [Verify Results and View Stored Memories](#verify-results-and-view-stored-memories)
    

### References

- [LangGraph: What-is-memory](https://langchain-ai.github.io/langgraph/concepts/memory/#what-is-memory)
- [LangGraph: memory-template](https://github.com/langchain-ai/memory-template)
- [LangChain-ai: memory-agent](https://github.com/langchain-ai/memory-agent)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">No packages to install.
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "ConversationMemoryManagementSystem",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# import for asynchronous tasks
import asyncio
import nest_asyncio

nest_asyncio.apply()
```

## Define System Prompt and Configuration

This section introduces the `SYSTEM_PROMPT` and the `Configuration` class. They are essential for setting up the system’s behavior and managing environment variables (for example, choosing which language model to use). You can think of `Configuration` as the single source of truth for any settings your application might need.

```python
# Define simple system prompt template used by the chatbot.
SYSTEM_PROMPT = """You are a helpful and friendly chatbot. Get to know the user! \
Ask questions! Be spontaneous! 
{user_info}

System Time: {time}"""
```

```python
import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated


# Define the Configuration class to handle runtime settings, including user ID, model name, and system prompt
@dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )
    system_prompt: str = SYSTEM_PROMPT

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }

        return cls(**{k: v for k, v in values.items() if v})
```

## Initialize LLM and Define State Class

In this part, we configure the `ChatOpenAI` model (using `model` and `temperature` settings) and introduce a `State` class. The `State` class holds the conversation messages, ensuring that **context** is retained and can be easily passed around. This lays the **foundation** for a conversational agent that genuinely “remembers” what has been said.

```python
# Import and initialize the OpenAI-based LLM
from langchain.chat_models import init_chat_model

llm = init_chat_model()
```

```python
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated
from dataclasses import dataclass


# Define the State class to store the list of messages in the conversation
@dataclass(kw_only=True)
class State:
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""
```

```python
# Define a utility function to split model provider and model name from a string
def split_model_and_provider(fully_specified_name: str) -> dict:
    """Initialize the configured chat model."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return {"model": model, "provider": provider}
```

## Memory Upsert Function

Here, we focus on the `upsert_memory` function. This function is responsible for storing or updating (**upserting**) user-specific data. By preserving user context across conversations—like interests, preferences, or corrections—you can give your application a more **persistent and personalized** feel.

```python
import uuid
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore


# Define a function to upsert (create or update) memory in the database
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: Optional[uuid.UUID] = None,
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    user_id = Configuration.from_runnable_config(config).user_id
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"
```

## Implement Conversation Flow (call_model, store_memory)

Next, we implement two important functions for our conversation flow:

1. `call_model`: Takes the current conversation `State`, retrieves relevant memories, and then sends them along with user messages to the LLM.
2. `store_memory`: Processes the model’s **tool calls**—in this case, requests to store data—and updates the memory store accordingly.

By combining these two functions, the model not only uses past **context** but also augments it with new information in real time.

```python
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore


# Define function to process the user's state and update memory based on conversation context
async def call_model(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    configurable = Configuration.from_runnable_config(config)

    # Retrieve the most recent memories for context
    memories = await store.asearch(
        ("memories", configurable.user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format memories for inclusion in the prompt
    formatted = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    # Prepare the system prompt with user memories and current time
    sys = configurable.system_prompt.format(
        user_info=formatted, time=datetime.now().isoformat()
    )
    print("system_msg:", sys)

    # Invoke the language model with the prepared prompt and tools
    msg = await llm.bind_tools([upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        {"configurable": split_model_and_provider(configurable.model)},
    )
    return {"messages": [msg]}
```

```python
# Define function to process tool calls and store memories in the memory store
async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    # Extract tool calls from the last message
    tool_calls = state.messages[-1].tool_calls

    # Concurrently execute all upsert_memory calls
    saved_memories = await asyncio.gather(
        *(upsert_memory(**tc["args"], config=config, store=store) for tc in tool_calls)
    )

    # Format the results of memory storage operations
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}
```

## Define Conditional Edge Logic

```python
# Define a function to determine the next step in the conversation flow
def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if msg.tool_calls:
        # Route to store_memory if there are tool calls
        return "store_memory"
    # Otherwise, finish
    return END
```

## Build and Execute StateGraph

In this section, we construct a `StateGraph` to define the flow of the conversation. We specify which node (for instance, `call_model`) leads to which next step (for example, `store_memory`). Once the graph is set, we run sample conversations to see how the system **dynamically** manages user input, retrieves relevant memories, and updates them when necessary.

```python
# Initialize and define the StateGraph, specifying nodes and edges for conversation flow
builder = StateGraph(State, config_schema=Configuration)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response
builder.add_edge("store_memory", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"
```

```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Visualize the compiled StateGraph as a Mermaid diagram
display(
    Image(
        graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
```


    
![png](./img/output_24_0.png)
    


## Verify Results and View Stored Memories

Finally, we examine the stored memories to confirm that our system has correctly captured the user’s context. You can look into the final conversation state (using `graph.get_state`) and see how messages and memories have been organized. This is a great point to do some **debugging** if anything seems amiss, ensuring that your memory mechanism works just as intended.

```python
# Prepare a sample conversation to test the memory agent

conversation = [
    "Hello, I'm Charlie. I work as a software engineer and I'm passionate about AI. Remember this.",
    "I specialize in machine learning algorithms and I'm currently working on a project involving natural language processing.",
    "My main goal is to improve sentiment analysis accuracy in multi-lingual texts. It's challenging but exciting.",
    "We've made some progress using transformer models, but we're still working on handling context and idioms across languages.",
    "Chinese and English have been the most challenging pair so far due to their vast differences in structure and cultural contexts.",
]
```

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Initialize an in-memory store and compile the graph with a memory saver checkpoint
mem_store = InMemoryStore()

graph = builder.compile(store=mem_store, checkpointer=MemorySaver())
user_id = "test-user"  # temporary user ID for testing
config = {
    "configurable": {
        "thread_id": 1,  # temporary thread ID for testing
    },
    "user_id": user_id,
}
```

```python
# Process each message in the conversation asynchronously using the compiled graph
for content in conversation:
    async for chunk in graph.astream(
        {"messages": [("user", content)]},
        config=config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
```

<pre class="custom">================================ Human Message =================================
    
    Hello, I'm Charlie. I work as a software engineer and I'm passionate about AI. Remember this.
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    
    System Time: 2025-01-12T15:54:11.951177
    ================================== Ai Message ==================================
    Tool Calls:
      upsert_memory (call_f3b7nFwOGZZtCzSfqcRyUqVo)
     Call ID: call_f3b7nFwOGZZtCzSfqcRyUqVo
      Args:
        content: Charlie is a software engineer and passionate about AI.
        context: Charlie introduced themselves and shared their profession and interest.
    ================================= Tool Message =================================
    
    Stored memory 83b21302-d28d-4a7d-b0a9-45f6ab790e37
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:13.275082
    ================================== Ai Message ==================================
    
    Hello Charlie! It's great to meet you. I've noted that you're a software engineer with a passion for AI. What kind of AI projects are you working on or interested in?
    ================================ Human Message =================================
    
    I specialize in machine learning algorithms and I'm currently working on a project involving natural language processing.
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:14.199902
    ================================== Ai Message ==================================
    Tool Calls:
      upsert_memory (call_Ti9pY6RDSywIrTEw50FX1tBA)
     Call ID: call_Ti9pY6RDSywIrTEw50FX1tBA
      Args:
        content: Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.
        context: Charlie shared their area of specialization and current project focus.
    ================================= Tool Message =================================
    
    Stored memory 1d086522-0510-43df-b119-96e66bc8f9d0
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    [1d086522-0510-43df-b119-96e66bc8f9d0]: {'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:15.287356
    ================================== Ai Message ==================================
    
    That sounds fascinating! Natural language processing is such an intriguing field. Are there any particular challenges or goals you're focusing on in your current project?
    ================================ Human Message =================================
    
    My main goal is to improve sentiment analysis accuracy in multi-lingual texts. It's challenging but exciting.
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    [1d086522-0510-43df-b119-96e66bc8f9d0]: {'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:16.151716
    ================================== Ai Message ==================================
    Tool Calls:
      upsert_memory (call_bSldwiluscIOp2RmsiHacmHc)
     Call ID: call_bSldwiluscIOp2RmsiHacmHc
      Args:
        content: Charlie's main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.
        context: Charlie described the primary goal and challenge of their current project involving natural language processing.
    ================================= Tool Message =================================
    
    Stored memory 427e0614-8768-44f4-8a01-f01c3cf933d3
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    [1d086522-0510-43df-b119-96e66bc8f9d0]: {'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'} (similarity: None)
    [427e0614-8768-44f4-8a01-f01c3cf933d3]: {'content': "Charlie's main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.", 'context': 'Charlie described the primary goal and challenge of their current project involving natural language processing.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:18.056567
    ================================== Ai Message ==================================
    
    Improving sentiment analysis accuracy in multi-lingual texts sounds like a challenging yet rewarding task. The nuances of different languages can definitely make it complex. How do you tackle the challenges of working with multiple languages? Are there any particular techniques or tools you find especially helpful?
    ================================ Human Message =================================
    
    We've made some progress using transformer models, but we're still working on handling context and idioms across languages.
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    [1d086522-0510-43df-b119-96e66bc8f9d0]: {'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'} (similarity: None)
    [427e0614-8768-44f4-8a01-f01c3cf933d3]: {'content': "Charlie's main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.", 'context': 'Charlie described the primary goal and challenge of their current project involving natural language processing.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:19.171699
    ================================== Ai Message ==================================
    
    Transformer models are indeed powerful for capturing context, but idioms can be tricky since they often don't translate directly. It's impressive that you're tackling such a complex issue! Are there any specific languages you're focusing on, or are you working with a broad range?
    ================================ Human Message =================================
    
    Chinese and English have been the most challenging pair so far due to their vast differences in structure and cultural contexts.
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    [1d086522-0510-43df-b119-96e66bc8f9d0]: {'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'} (similarity: None)
    [427e0614-8768-44f4-8a01-f01c3cf933d3]: {'content': "Charlie's main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.", 'context': 'Charlie described the primary goal and challenge of their current project involving natural language processing.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:20.297475
    ================================== Ai Message ==================================
    
    Chinese and English do pose a unique set of challenges due to their linguistic and cultural differences. It's intriguing to see how you're addressing these complexities. Have you found any particular strategies effective in bridging these differences, or are there cultural nuances that you're still exploring?
</pre>

```python
from pprint import pprint

# Search and check stored memories for the user
namespace = ("memories", user_id)
memories = mem_store.search(namespace)
pprint(memories)
```

<pre class="custom">[Item(namespace=['memories', 'test-user'], key='83b21302-d28d-4a7d-b0a9-45f6ab790e37', value={'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'}, created_at='2025-01-12T06:54:13.274005+00:00', updated_at='2025-01-12T06:54:13.274006+00:00', score=None),
     Item(namespace=['memories', 'test-user'], key='1d086522-0510-43df-b119-96e66bc8f9d0', value={'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'}, created_at='2025-01-12T06:54:15.285755+00:00', updated_at='2025-01-12T06:54:15.285763+00:00', score=None),
     Item(namespace=['memories', 'test-user'], key='427e0614-8768-44f4-8a01-f01c3cf933d3', value={'content': "Charlie's main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.", 'context': 'Charlie described the primary goal and challenge of their current project involving natural language processing.'}, created_at='2025-01-12T06:54:18.054776+00:00', updated_at='2025-01-12T06:54:18.054780+00:00', score=None)]
</pre>

```python
# Test memory recall by asking a question. Notice the memory the agent recalls.
async for chunk in graph.astream(
    {"messages": [("user", "Whats my name?")]},
    config=config,
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
```

<pre class="custom">================================ Human Message =================================
    
    Whats my name?
    system_msg: You are a helpful and friendly chatbot. Get to know the user! Ask questions! Be spontaneous! 
    
    <memories>
    [83b21302-d28d-4a7d-b0a9-45f6ab790e37]: {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'} (similarity: None)
    [1d086522-0510-43df-b119-96e66bc8f9d0]: {'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'} (similarity: None)
    [427e0614-8768-44f4-8a01-f01c3cf933d3]: {'content': "Charlie's main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.", 'context': 'Charlie described the primary goal and challenge of their current project involving natural language processing.'} (similarity: None)
    </memories>
    
    System Time: 2025-01-12T15:54:21.511105
    ================================== Ai Message ==================================
    
    Your name is Charlie.
</pre>

```python
# Retrieve and print the final state of the conversation
graph.get_state(config).values["messages"]
```




<pre class="custom">[HumanMessage(content="Hello, I'm Charlie. I work as a software engineer and I'm passionate about AI. Remember this.", additional_kwargs={}, response_metadata={}, id='58ecf148-30bf-402e-8554-707cda5df3e5'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_f3b7nFwOGZZtCzSfqcRyUqVo', 'function': {'arguments': '{"content":"Charlie is a software engineer and passionate about AI.","context":"Charlie introduced themselves and shared their profession and interest."}', 'name': 'upsert_memory'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 37, 'prompt_tokens': 233, 'total_tokens': 270, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c10af4fa-75c9-4a8d-8acc-c9ad89343b42-0', tool_calls=[{'name': 'upsert_memory', 'args': {'content': 'Charlie is a software engineer and passionate about AI.', 'context': 'Charlie introduced themselves and shared their profession and interest.'}, 'id': 'call_f3b7nFwOGZZtCzSfqcRyUqVo', 'type': 'tool_call'}], usage_metadata={'input_tokens': 233, 'output_tokens': 37, 'total_tokens': 270, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     ToolMessage(content='Stored memory 83b21302-d28d-4a7d-b0a9-45f6ab790e37', id='e33883d4-617f-4263-8ce5-2f452cec0839', tool_call_id='call_f3b7nFwOGZZtCzSfqcRyUqVo'),
     AIMessage(content="Hello Charlie! It's great to meet you. I've noted that you're a software engineer with a passion for AI. What kind of AI projects are you working on or interested in?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 37, 'prompt_tokens': 374, 'total_tokens': 411, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-71394dc8-9891-4991-a712-4687219040dc-0', usage_metadata={'input_tokens': 374, 'output_tokens': 37, 'total_tokens': 411, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     HumanMessage(content="I specialize in machine learning algorithms and I'm currently working on a project involving natural language processing.", additional_kwargs={}, response_metadata={}, id='8f044596-eeff-40f4-92a8-24fa751af3ba'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ti9pY6RDSywIrTEw50FX1tBA', 'function': {'arguments': '{"content":"Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.","context":"Charlie shared their area of specialization and current project focus."}', 'name': 'upsert_memory'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 435, 'total_tokens': 481, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-8f6e1dab-c3f7-4a06-9af6-819a76d1177f-0', tool_calls=[{'name': 'upsert_memory', 'args': {'content': 'Charlie specializes in machine learning algorithms and is currently working on a project involving natural language processing.', 'context': 'Charlie shared their area of specialization and current project focus.'}, 'id': 'call_Ti9pY6RDSywIrTEw50FX1tBA', 'type': 'tool_call'}], usage_metadata={'input_tokens': 435, 'output_tokens': 46, 'total_tokens': 481, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     ToolMessage(content='Stored memory 1d086522-0510-43df-b119-96e66bc8f9d0', id='563e8710-170b-446e-bee5-a0bdd1c91cce', tool_call_id='call_Ti9pY6RDSywIrTEw50FX1tBA'),
     AIMessage(content="That sounds fascinating! Natural language processing is such an intriguing field. Are there any particular challenges or goals you're focusing on in your current project?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 30, 'prompt_tokens': 582, 'total_tokens': 612, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-5abf4334-5fa2-4dbe-93bd-4cb6b3110e58-0', usage_metadata={'input_tokens': 582, 'output_tokens': 30, 'total_tokens': 612, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     HumanMessage(content="My main goal is to improve sentiment analysis accuracy in multi-lingual texts. It's challenging but exciting.", additional_kwargs={}, response_metadata={}, id='c6a9869a-3d21-491f-9fd4-59eed893ab4c'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_bSldwiluscIOp2RmsiHacmHc', 'function': {'arguments': '{"content":"Charlie\'s main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.","context":"Charlie described the primary goal and challenge of their current project involving natural language processing."}', 'name': 'upsert_memory'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 54, 'prompt_tokens': 639, 'total_tokens': 693, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a927456-ea41-4ec6-af8a-936f340bc8e8-0', tool_calls=[{'name': 'upsert_memory', 'args': {'content': "Charlie's main goal in their current project is to improve sentiment analysis accuracy in multi-lingual texts.", 'context': 'Charlie described the primary goal and challenge of their current project involving natural language processing.'}, 'id': 'call_bSldwiluscIOp2RmsiHacmHc', 'type': 'tool_call'}], usage_metadata={'input_tokens': 639, 'output_tokens': 54, 'total_tokens': 693, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     ToolMessage(content='Stored memory 427e0614-8768-44f4-8a01-f01c3cf933d3', id='1cc43a3f-bd34-4cc4-a0d9-e12b241a92be', tool_call_id='call_bSldwiluscIOp2RmsiHacmHc'),
     AIMessage(content='Improving sentiment analysis accuracy in multi-lingual texts sounds like a challenging yet rewarding task. The nuances of different languages can definitely make it complex. How do you tackle the challenges of working with multiple languages? Are there any particular techniques or tools you find especially helpful?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 56, 'prompt_tokens': 804, 'total_tokens': 860, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-6d35cedc-258d-4154-b6d3-8f2745e05602-0', usage_metadata={'input_tokens': 804, 'output_tokens': 56, 'total_tokens': 860, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     HumanMessage(content="We've made some progress using transformer models, but we're still working on handling context and idioms across languages.", additional_kwargs={}, response_metadata={}, id='b9ac5c3b-ad4d-4e60-bced-ec0de8a71024'),
     AIMessage(content="Transformer models are indeed powerful for capturing context, but idioms can be tricky since they often don't translate directly. It's impressive that you're tackling such a complex issue! Are there any specific languages you're focusing on, or are you working with a broad range?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 52, 'prompt_tokens': 887, 'total_tokens': 939, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-858ab2a9-9b1c-4a69-a311-3eb59ac4e358-0', usage_metadata={'input_tokens': 887, 'output_tokens': 52, 'total_tokens': 939, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     HumanMessage(content='Chinese and English have been the most challenging pair so far due to their vast differences in structure and cultural contexts.', additional_kwargs={}, response_metadata={}, id='e7975af6-0cef-4fba-b164-b50bfd9a2b9e'),
     AIMessage(content="Chinese and English do pose a unique set of challenges due to their linguistic and cultural differences. It's intriguing to see how you're addressing these complexities. Have you found any particular strategies effective in bridging these differences, or are there cultural nuances that you're still exploring?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 52, 'prompt_tokens': 967, 'total_tokens': 1019, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-a65311a7-e301-4422-b2b0-ceaf5b8f230a-0', usage_metadata={'input_tokens': 967, 'output_tokens': 52, 'total_tokens': 1019, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     HumanMessage(content='Whats my name?', additional_kwargs={}, response_metadata={}, id='21b8072f-ee35-45f3-8558-63a726605f67'),
     AIMessage(content='Your name is Charlie.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 1029, 'total_tokens': 1036, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_703d4ff298', 'finish_reason': 'stop', 'logprobs': None}, id='run-15b1800c-2b57-445f-8d6a-5bfdd2e17424-0', usage_metadata={'input_tokens': 1029, 'output_tokens': 7, 'total_tokens': 1036, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]</pre>


