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

# Functional API

- Author: [Yejin Park](https://github.com/ppakyeah)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial covers LangGraph's Functional API, focusing on workflow automation with `@entrypoint` and `@task` decorators.

Key features include state management, parallel processing, and human-in-the-loop capabilities.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Functional API](#functional-api)
- [Use Cases](#use-cases)

### References

- [LangGraph: Functional API Document](https://langchain-ai.github.io/langgraph/concepts/functional_api/)
- [LangGraph: Functional API Tutorial](https://github.com/langchain-ai/langgraph/blob/f239b39060096ab2c8bff0d6303781efee174a5c/docs/docs/tutorials/functional_api/functional_api_test.ipynb)
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
        "langchain_core",
        "langgraph",
        "langchain-openai",
    ],
    verbose=False,
    upgrade=True,
)
```

You can set API keys in a `.env` file or set them manually.

[Note] If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "15-LangGraph-Functional-API",
        }
    )
```

## Functional API
The Functional API is a programming interface provided by LangGraph that extends existing Python functions with advanced features such as state management, parallel processing, and memory management, all while requiring minimal code modifications.

### Core Components
The Functional API uses two primitives to define workflows:
1. `@entrypoint` Decorator
- Defines the entry point of a workflow
- Automates state management and checkpointing
- Manages streaming and interruption points

```python
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint


@entrypoint(checkpointer=MemorySaver())
def calculate_sum(numbers: list[int]) -> int:
    """A simple workflow that sums numbers"""
    return sum(numbers)

config = {
    "configurable": {
        "thread_id": str(uuid4())
    }
}

calculate_sum.invoke([1, 2, 3, 4, 5], config)
```




<pre class="custom">15</pre>



2. `@task` Decorator
- Defines units of work that can be executed asynchronously
- Handles retry policies and error handling
- Supports parallel processing

```python
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import task

@task()
def multiply_number(num: int) -> int:
    """Simple task that multiplies a number by 2"""
    return num * 2

@entrypoint(checkpointer=MemorySaver())
def calculate_multiply(num: int) -> int:
    """A simple workflow that multiplies two numbers"""
    future = multiply_number(num)
    return future.result()

config = {
    "configurable": {
        "thread_id": str(uuid4())
    }
}
calculate_multiply.invoke(3, config)
```




<pre class="custom">6</pre>



## Use Cases

### Asynchronous and Parallel Processing

Long-running tasks can significantly impact application performance.

The Functional API allows you to execute tasks asynchronously and in parallel, improving efficiency especially for I/O-bound operations like LLMs API calls.

The `@task` decorator makes it easy to convert regular functions into asynchronous tasks.

```python
from langgraph.func import task
import time

@task()
def process_number(n: int) -> int:
    """Simulates processing by waiting for 1 second"""
    time.sleep(1)
    return n * 2

@entrypoint()
def parallel_processing(numbers: list[int]) -> list[int]:
    """Processes multiple numbers in parallel"""
    # Start all tasks
    futures = [process_number(n) for n in numbers]
    return [f.result() for f in futures]

parallel_processing.invoke([1, 2, 3, 4, 5])
```




<pre class="custom">[2, 4, 6, 8, 10]</pre>



### Interrupts and Human Intervention

Some workflows require human oversight or intervention at critical points.

The Functional API provides built-in support for human-in-the-loop processes through its interrupt mechanism.

This allows you to pause execution, get human input, and continue processing based on that input.

```python
from uuid import uuid4
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


@task()
def step_1(input_query):
    """Append bar."""
    return f"{input_query} bar"


@task()
def human_feedback(input_query):
    """Append user input."""
    feedback = interrupt(f"Please provide feedback: {input_query}")
    return f"{input_query} {feedback}"


@task()
def step_3(input_query):
    """Append qux."""
    return f"{input_query} qux"

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def graph(input_query):
    result_1 = step_1(input_query).result()
    feedback = interrupt(f"Please provide feedback: {result_1}")

    result_2 = f"{input_query} {feedback}"
    result_3 = step_3(result_2).result()

    return result_3

config = {"configurable": {"thread_id": str(uuid4())}}
for event in graph.stream("foo", config):
    print(event)
    print("\n")
```

<pre class="custom">{'step_1': 'foo bar'}
    
    
    {'__interrupt__': (Interrupt(value='Please provide feedback: foo bar', resumable=True, ns=['graph:f550c5f8-67e0-6c57-9206-c10c7affc896'], when='during'),)}
    
    
</pre>

```python
# Continue execution
for event in graph.stream(Command(resume="baz"), config):
    print(event)
    print("\n")
```

<pre class="custom">{'step_3': 'foo baz qux'}
    
    
    {'graph': 'foo baz qux'}
    
    
</pre>

### Automated State Management

The Functional API automatically handles state persistence and restoration between function calls.

This is particularly useful in conversational applications where maintaining context is crucial.

You can focus on your business logic while LangGraph handles the complexities of state management.

```python
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.func import entrypoint
from langgraph.graph import add_messages


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

checkpointer = MemorySaver()

# Set a checkpointer to enable persistence.
@entrypoint(checkpointer=checkpointer)
def conversational_agent(messages: list[BaseMessage], *, previous: list[BaseMessage] = None):
    # Add previous messages from short-term memory to the current messages
    if previous is not None:
        messages = add_messages(previous, messages)

    # Get agent's response based on conversation history.
    llm_response = llm.invoke(
         [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + messages
    )

    # Add agent's messages to conversation history
    messages = add_messages(messages, llm_response)

    return messages
```

```python
# Config
config = {
    "configurable": {
        "thread_id": str(uuid4())
    }
}

# Run with checkpointer to persist state in memory
messages = conversational_agent.invoke([HumanMessage(content="Hi. I'm currently creating a tutorial, named LangChain OpenTutorial.")], config)
for m in messages:
    m.pretty_print()
```

<pre class="custom">================================ Human Message =================================
    
    Hi. I'm currently creating a tutorial, named LangChain OpenTutorial.
    ================================== Ai Message ==================================
    
    That sounds like a great project! How can I assist you with your LangChain OpenTutorial? Are you looking for help with content, examples, or something else?
</pre>

```python
# Checkpoint state
agent_state = conversational_agent.get_state(config)
for m in agent_state.values:
    m.pretty_print()
```

<pre class="custom">================================ Human Message =================================
    
    Hi. I'm currently creating a tutorial, named LangChain OpenTutorial.
    ================================== Ai Message ==================================
    
    That sounds like a great project! How can I assist you with your LangChain OpenTutorial? Are you looking for help with content, examples, or something else?
</pre>

```python
# Continue with the same thread
messages = conversational_agent.invoke([HumanMessage(content="Do you remember the name of my tutorial that I'm now working on?")], config)
for m in messages:
    m.pretty_print()
```

<pre class="custom">================================ Human Message =================================
    
    Hi. I'm currently creating a tutorial, named LangChain OpenTutorial.
    ================================== Ai Message ==================================
    
    That sounds like a great project! How can I assist you with your LangChain OpenTutorial? Are you looking for help with content, examples, or something else?
    ================================ Human Message =================================
    
    Do you remember the name of my tutorial that I'm now working on?
    ================================== Ai Message ==================================
    
    Yes, you mentioned that you are creating a tutorial named "LangChain OpenTutorial." How can I assist you further with it?
</pre>
