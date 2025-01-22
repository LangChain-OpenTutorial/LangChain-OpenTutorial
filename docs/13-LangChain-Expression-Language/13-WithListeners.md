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

# WithListeners

- Author: [Donghak Lee](https://github.com/stsr1284)
- Design:
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial covers the implementation and usage of `with_listeners()` in `Runnable`.

`with_listeners()` binds lifecycle listeners to a Runnable, returning a new Runnable. This allows you to connect event listeners to the data flow, enabling tracking, analysis, and debugging during execution.

The `with_listeners()` function provides the ability to add event listeners to a Runnable object. Listeners are functions that are called when specific events occur, such as start, end, or error.

This function is useful in the following scenarios:

- Logging the start and end of data processing

- Triggering notifications on errors

- Printing debugging information

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [with_listeners](#with_listeners)
- [with_alisteners](#with_alisteners)
- [RootListenersTracer](#rootlistenerstracer)

### References

- [LangChain with_listeners](https://python.langchain.com/v0.2/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_listeners)
- [LangChain RootListenersTracer](https://python.langchain.com/v0.2/api_reference/core/tracers/langchain_core.tracers.root_listeners.RootListenersTracer.html)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_openai",
        "datetime",
    ],
    verbose=False,
    upgrade=False,
)
```

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "WithListeners",
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

## with_listeners

`with_listeners()` takes a list of listener functions and returns a new `Runnable` object. Listener functions respond to start, end, and error events.

Using event listeners allows you to observe the data flow, and you can flexibly register them using `with_listeners()`.

Here is an example implementation of the function.

```python
from langchain_core.runnables import RunnableLambda
import time


# Define tasks for each Runnable
def stepOne(message):
    time.sleep(1)  # Wait for 1 second
    return f"Step 1 completed with message {message}"


def stepTwo(message):
    time.sleep(2)  # Wait for 2 seconds
    return f"Step 2 completed with message {message}"


# Define listener functions
def fnStart(runObj):
    print(f"Start: {runObj.inputs}")


def fnEnd(runObj):
    print(f"End: {runObj.outputs}")


def fnError(runObj):
    print(f"Error: {runObj.error}")


# Define each Runnable
runnable1 = RunnableLambda(stepOne).with_listeners(
    on_start=fnStart, on_end=fnEnd, on_error=fnError
)

runnable2 = RunnableLambda(stepTwo).with_listeners(
    on_start=fnStart, on_end=fnEnd, on_error=fnError
)

# Chain connection
chain = runnable1 | runnable2

# Execute
chain.invoke("Hello, World!")
```

<pre class="custom">Start: {'input': 'Hello, World!'}
    End: {'output': 'Step 1 completed with message Hello, World!'}
    Start: {'input': 'Step 1 completed with message Hello, World!'}
    End: {'output': 'Step 2 completed with message Step 1 completed with message Hello, World!'}
</pre>




    'Step 2 completed with message Step 1 completed with message Hello, World!'



You can also register events in the chain of LCEL using `with_listeners()`.

```python
def chainStart(runObj):
    print(f"Chain Start: {runObj.inputs}")


def chainEnd(runObj):
    print(f"Chain End: {runObj.outputs}")


chain_with_listeners = chain.with_listeners(
    on_start=chainStart, on_end=chainEnd, on_error=fnError
)

chain_with_listeners.invoke("Hello, World!")
```

<pre class="custom">Chain Start: {'input': 'Hello, World!'}
    Start: {'input': 'Hello, World!'}
    End: {'output': 'Step 1 completed with message Hello, World!'}
    Start: {'input': 'Step 1 completed with message Hello, World!'}
    End: {'output': 'Step 2 completed with message Step 1 completed with message Hello, World!'}
    Chain End: {'output': 'Step 2 completed with message Step 1 completed with message Hello, World!'}
</pre>




    'Step 2 completed with message Step 1 completed with message Hello, World!'



## with_alisteners

Bind asynchronous lifecycle listeners to a Runnable, returning a new Runnable.

on_start: Asynchronously called before the Runnable starts running.
on_end: Asynchronously called after the Runnable finishes running.
on_error: Asynchronously called if the Runnable throws an error.

The Run object contains information about the run, including its id, type, input, output, error, start_time, end_time, and any tags or metadata added to the run.

```python
import asyncio


async def testRunnable(time_to_sleep: int):
    print(f"Runnable[{time_to_sleep}s]: starts at {time.strftime('%S')}")
    await asyncio.sleep(time_to_sleep)
    print(f"Runnable[{time_to_sleep}s]: ends at {time.strftime('%S')}")


async def fnStart(runObj):
    print(f"runnable{runObj.inputs['input']}: {time.strftime('%S')}")
    await asyncio.sleep(3)
    print(f"runnable{runObj.inputs['input']}: {time.strftime('%S')}")


async def fnEnd(runObj):
    print(f"runnable{runObj.inputs['input']}: {time.strftime('%S')}")
    await asyncio.sleep(2)
    print(f"runnable{runObj.inputs['input']}: {time.strftime('%S')}")


runnable = RunnableLambda(testRunnable).with_alisteners(on_start=fnStart, on_end=fnEnd)


async def concurrentRuns():
    await asyncio.gather(runnable.ainvoke(2), runnable.ainvoke(3))


await concurrentRuns()
```

<pre class="custom">runnable2: 25
    runnable3: 25
    runnable2: 28
    runnable3: 28
    Runnable[2s]: starts at 28
    Runnable[3s]: starts at 28
    Runnable[2s]: ends at 30
    runnable2: 30
    Runnable[3s]: ends at 31
    runnable3: 31
    runnable2: 32
    runnable3: 33
</pre>

## RootListenersTracer

You can directly bind `RootListenersTracer` to a Runnable using `RunnableBinding` to register event listeners. This is the internal code of `with_listeners()`.

`RootListenersTracer` calls listeners on run start, end, and error.

```python
from langchain_core.tracers.root_listeners import RootListenersTracer
from langchain_core.runnables.base import RunnableBinding
from langchain_openai import ChatOpenAI


# Define listener functions
def fnStart(runObj):
    print(f"Start: {runObj.inputs}")


def fnEnd(runObj):
    print(f"End: {runObj.outputs}")


def fnError(runObj):
    print(f"End: {runObj.error}")


# # LLM and chain setup
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

model_with_listeners = RunnableBinding(
    bound=model,
    config_factories=[
        lambda config: {
            "callbacks": [
                RootListenersTracer(
                    config=config,
                    on_start=fnStart,
                    on_end=fnEnd,
                    on_error=fnError,
                )
            ],
        }
    ],
)

model_with_listeners.invoke("Tell me the founding year of Google")
```

<pre class="custom">Start: {'messages': [[{'lc': 1, 'type': 'constructor', 'id': ['langchain', 'schema', 'messages', 'HumanMessage'], 'kwargs': {'content': 'Tell me the founding year of Google', 'type': 'human'}}]]}
    End: {'generations': [[{'text': 'Google was founded in the year 1998.', 'generation_info': {'finish_reason': 'stop', 'logprobs': None}, 'type': 'ChatGeneration', 'message': {'lc': 1, 'type': 'constructor', 'id': ['langchain', 'schema', 'messages', 'AIMessage'], 'kwargs': {'content': 'Google was founded in the year 1998.', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 11, 'prompt_tokens': 14, 'total_tokens': 25, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'id': 'run-6f335bec-171d-47a8-a508-85bb52307e10-0', 'usage_metadata': {'input_tokens': 14, 'output_tokens': 11, 'total_tokens': 25, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}, 'tool_calls': [], 'invalid_tool_calls': []}}}]], 'llm_output': {'token_usage': {'completion_tokens': 11, 'prompt_tokens': 14, 'total_tokens': 25, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c'}, 'run': None, 'type': 'LLMResult'}
</pre>




    AIMessage(content='Google was founded in the year 1998.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 14, 'total_tokens': 25, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}, id='run-6f335bec-171d-47a8-a508-85bb52307e10-0', usage_metadata={'input_tokens': 14, 'output_tokens': 11, 'total_tokens': 25, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



### Using tracers without LCEL
You can directly use `on_llm_start()` and `on_llm_end()` of `RootListenersTracer` to handle events.

```python
from langchain_core.tracers.schemas import Run
import uuid
from datetime import datetime, timezone


# Modify user-defined listener functions
def onStart(run: Run):
    print(
        f"[START] Run ID: {run.id}, Start time: {run.start_time}\nInput: {run.inputs}"
    )


def onEnd(run: Run):
    # Safely handle output
    print(f"[END] Run ID: {run.id}, End time: {run.end_time}\nOutput: {run.outputs}")


def onError(run: Run):
    print(f"[ERROR] Run ID: {run.id}, Error message: {run.error}")


# Create RootListenersTracer
tracer = RootListenersTracer(
    config={}, on_start=onStart, on_end=onEnd, on_error=onError
)

# Set up LLM
llm = ChatOpenAI()

# Input text
input_text = "What is the founding year of Google?"

try:
    # Create and initialize Run object at the start of execution
    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)

    # Create Run object (including only required fields)
    run = Run(
        id=run_id,
        start_time=start_time,
        execution_order=1,
        serialized={},
        inputs={"input": input_text},
        run_type="llm",
    )

    # Call tracer at the start of execution
    tracer.on_llm_start(serialized={}, prompts=[input_text], run_id=run_id)

    # Execute the actual Runnable
    result = llm.generate([input_text])

    # Update Run object
    run.end_time = datetime.now(timezone.utc)
    run.outputs = result

    # Call tracer at the end of execution
    tracer.on_llm_end(response=result, run_id=run_id)

except Exception as e:
    run.error = str(e)
    run.end_time = datetime.now(timezone.utc)
    tracer.on_llm_error(error=e, run_id=run_id)
    print(f"Error occurred: {str(e)}")
```

<pre class="custom">[START] Run ID: a76a54b6-8173-4173-b063-ebe107e52dd3, Start time: 2025-01-12 05:32:32.311749+00:00
    Input: {'prompts': ['What is the founding year of Google?']}
    [END] Run ID: a76a54b6-8173-4173-b063-ebe107e52dd3, End time: 2025-01-12 05:32:32.898851+00:00
    Output: {'generations': [[{'text': 'Google was founded on September 4, 1998.', 'generation_info': {'finish_reason': 'stop', 'logprobs': None}, 'type': 'ChatGeneration', 'message': {'lc': 1, 'type': 'constructor', 'id': ['langchain', 'schema', 'messages', 'AIMessage'], 'kwargs': {'content': 'Google was founded on September 4, 1998.', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 13, 'prompt_tokens': 15, 'total_tokens': 28, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'id': 'run-d0c3617b-05c1-4e34-8fa5-eba2ed0f2748-0', 'usage_metadata': {'input_tokens': 15, 'output_tokens': 13, 'total_tokens': 28, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}, 'tool_calls': [], 'invalid_tool_calls': []}}}]], 'llm_output': {'token_usage': {'completion_tokens': 13, 'prompt_tokens': 15, 'total_tokens': 28, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo'}, 'run': [{'run_id': UUID('d0c3617b-05c1-4e34-8fa5-eba2ed0f2748')}], 'type': 'LLMResult'}
</pre>
