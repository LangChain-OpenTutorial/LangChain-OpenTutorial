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

# RunnableRetry

- Author: [PangPangGod](https://github.com/pangpanggod)
- Design: []()
- Peer Review : []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/12-RunnableRetry.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/12-RunnableRetry.ipynb)

## Overview

This tutorial covers how to use `RunnableRetry` to handle retry logic effectively in LangChain workflows.  
We'll demonstrate how to configure and use `RunnableRetry` with examples that showcase custom retry policies to make your workflow resilient to failures.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [What is RunnableRetry](#what-is-runnableretry)
- [Why Use RunnableRetry](#why-use-runnableretry)
- [Base RunnableRetry Example](#base-runnableretry-example)
- [RunnableRetry Bind with Chains](#runnableretry-bind-with-chains)

### References

- [LangChain API Reference: RunnableRetry](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.retry.RunnableRetry.html)
- [LangChain OpenTutorial: PydanticOutputParser](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/01-PydanticOuputParser.ipynb)
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

<pre class="custom">
    [notice] A new release of pip is available: 23.2.1 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain_core",
        "langchain_openai",
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
        "LANGCHAIN_PROJECT": "RunnableRetry",
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



## What is RunnableRetry

`RunnableRetry` is a utility provided by LangChain that allows you to add retry mechanisms for individual `Runnable` objects.  
Instead of wrapping your entire workflow in retry logic, you can apply retry policies at the level of specific tasks.   
This helps you handle transient issues, such as network errors or intermittent failures, without restarting the entire workflow.

## Why Use RunnableRetry

By using `RunnableRetry`, you can:

- **Avoid wrapping the entire workflow with retry logic**: Instead of restarting the entire process during frequent network calls or API failures, you can retry individual `Runnable` units.
- **Implement retries per task**: This enables more efficient task recovery and makes workflows more robust.
- **Flexible implementation**: You can implement retries using `.with_retry()` or define a custom retry strategy by creating a `RunnableRetry` with specific events, such as exception types and exponential backoff.

## Base RunnableRetry Example

Below is a simple example to demonstrate the effectiveness of `RunnableRetry`.  
In this example, we simulate a task with a chance of failure and use `RunnableRetry` to automatically retry it up to a maximum attempts.

```python
import random
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry

# Define a simple function with a chance of failure.
def random_number(input):
    number = random.randint(1, 3)
    if number == 1:
        print("Success! The number is 1.")
    else:
        print(f"Failed! The number is {number}.")
        raise ValueError

# Bind the function to RunnableLambda
runnable = RunnableLambda(random_number)

# Then bind it to RunnableRetry
runnable_with_retries = RunnableRetry(
    bound=runnable,
    retry_exception_types=(ValueError,),
    max_attempt_number=5,
    wait_exponential_jitter=True,
)

# In this example, there is no need for input, but LangChain's Runnable requires an input argument.
# TypeError: RunnableRetry.invoke() missing 1 required positional argument: 'input'
input=None
runnable_with_retries.invoke(input)
```

<pre class="custom">Failed! The number is 3.
    Success! The number is 1.
</pre>

or you can simply implemented with `.with_retry()` method.

```python
# Bind the function to RunnableLambda
runnable = RunnableLambda(random_number)

# with .with_retry(), no need to bind with Runnableretry(bind= ...)
runnable_with_retries = runnable.with_retry(
    retry_if_exception_type=(ValueError,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

input=None
runnable_with_retries.invoke(None)
```

<pre class="custom">Failed! The number is 3.
    Failed! The number is 2.
    Success! The number is 1.
</pre>

## RunnableRetry Bind with Chains

In this example, we’ll take it a step further and demonstrate how to construct a Chain using `ChatOpenAI`. The example will show not just the basic chain setup but also how to enhance it by incorporating `RunnableRetry` for robust error handling and `PydanticOutputParser` for structured output validation.

### Components Used:
- `RunnableRetry`: Automatically retries failed tasks to handle transient issues, such as API call failures or timeouts.
- `PydanticOutputParser`: Ensures the output is parsed and validated against a defined schema, making the workflow more reliable and predictable.

with `PydanticOutputParser`, check our another tutorial [here](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/01-PydanticOuputParser.ipynb).

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables.retry import RunnableRetry
from langchain_core.prompts import PromptTemplate

# Note: Each model provider may have different error classes for API-related failures.
# For example, OpenAI uses "InternalServerError", while other providers may define different exceptions.
from openai import InternalServerError

# first, define model and prompt
model = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate.from_template("tell me a joke about {topic}.")

# bind with RunnableRetry
model_bind_with_retry = RunnableRetry(
    bound=model,
    retry_exception_types=(InternalServerError,),
    max_attempt_number=3,
    wait_exponential_jitter=True,
)
```

```python
chain = prompt | model_bind_with_retry
chain.invoke("programming")
```




<pre class="custom">AIMessage(content='Why do programmers prefer dark mode?\n\nBecause light attracts bugs!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 14, 'total_tokens': 27, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_d02d531b47', 'finish_reason': 'stop', 'logprobs': None}, id='run-56bd0842-7465-496b-a4a9-b4e1545dd1da-0', usage_metadata={'input_tokens': 14, 'output_tokens': 13, 'total_tokens': 27, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



or you can use `.with_retry()` with Runnables.

```python
chain = prompt | model.with_retry(retry_if_exception_type=(InternalServerError,), stop_after_attempt=3, wait_exponential_jitter=True)
chain.invoke("bear")
```




<pre class="custom">AIMessage(content='Why do bears have hairy coats?\n\nBecause they look silly in sweaters!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 14, 'total_tokens': 29, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0aa8d3e20b', 'finish_reason': 'stop', 'logprobs': None}, id='run-a73176c2-8550-40bf-8284-acb56a302c07-0', usage_metadata={'input_tokens': 14, 'output_tokens': 15, 'total_tokens': 29, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>


