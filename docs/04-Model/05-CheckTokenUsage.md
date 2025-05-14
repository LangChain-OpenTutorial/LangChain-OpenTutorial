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

# Check Token Usage

- Author: [Haseom Shin](https://github.com/IHAGI-c)
- Peer Review : [Teddy Lee](https://github.com/teddylee777), [Sooyoung](https://github.com/sooyoung-wind)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/05-CheckTokenUsage.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/05-CheckTokenUsage.ipynb)
## Overview

This tutorial covers how to track and monitor token usage with ```LangChain``` and ```OpenAI API```.

```Token usage tracking``` is crucial for managing API costs and optimizing resource utilization.

In this tutorial, we will create a simple example to measure and monitor token consumption during OpenAI API calls using LangChain's ```CallbackHandler```.

![example](./img/04-CheckTokenUsage-example-flow-token-usage.png)

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Implementing Check Token Usage](#implementing-check-token-usage)
- [Monitoring Token Usage](#monitoring-token-usage)

### References

- [OpenAI API Pricing](https://openai.com/api/pricing/)
- [Token Usage Guide](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
- [LangChain Python API Reference](https://python.langchain.com/api_reference/community/callbacks/langchain_community.callbacks.manager.get_openai_callback.html)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_openai",
        "langchain_community",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 24.2 -> 24.3.1
    [notice] To update, run: pip install --upgrade pip
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
        "LANGCHAIN_PROJECT": "04-CheckTokenUsage",
    }
)
```

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



Let's setup ```ChatOpenAI``` with ```gpt-4o``` model.

```python
from langchain_openai import ChatOpenAI

# Load the model
llm = ChatOpenAI(model_name="gpt-4o")
```

## Implementing Check Token Usage

if you want to check token usage, you can use ```get_openai_callback``` function.

```python
# callback to track it
from langchain_community.callbacks.manager import get_openai_callback

with get_openai_callback() as cb:
    result = llm.invoke("where is the capital of United States?")
    print(cb)
```

<pre class="custom">Tokens Used: 28
    	Prompt Tokens: 15
    		Prompt Tokens Cached: 0
    	Completion Tokens: 13
    		Reasoning Tokens: 0
    Successful Requests: 1
    Total Cost (USD): $0.00016749999999999998
</pre>

```python
# callback to track it
with get_openai_callback() as cb:
    result = llm.invoke("where is the capital of United States?")
    print(f"Total tokens used: \t\t{cb.total_tokens}")
    print(f"Tokens used in prompt: \t\t{cb.prompt_tokens}")
    print(f"Tokens used in completion: \t{cb.completion_tokens}")
    print(f"Cost: \t\t\t\t${cb.total_cost}")
```

<pre class="custom">Total tokens used: 		28
    Tokens used in prompt: 		15
    Tokens used in completion: 	13
    Cost: 				$0.00016749999999999998
</pre>

## Monitoring Token Usage

Token usage monitoring is crucial for managing costs and resources when using the OpenAI API. LangChain provides an easy way to track this through ```get_openai_callback()```.

In this section, we'll explore token usage monitoring through three key scenarios:

1. **Single Query Monitoring**: 
   - Track token usage for individual API calls
   - Distinguish between prompt and completion tokens
   - Calculate costs

2. **Multiple Queries Monitoring**:
   - Track cumulative token usage across multiple API calls
   - Analyze total costs

> **Note**: Token usage monitoring is currently only supported for OpenAI API.

```python
# 1. Single Query Monitoring
print("1. Single Query Monitoring")
print("-" * 40)

with get_openai_callback() as cb:
    response = llm.invoke("What is the capital of France?")
    print(f"Response: {response.content}")
    print("-" * 40)
    print(f"Token Usage Statistics:")
    print(f"Prompt Tokens: \t\t{cb.prompt_tokens}")
    print(f"Completion Tokens: \t{cb.completion_tokens}")
    print(f"Total Tokens: \t\t{cb.total_tokens}")
    print(f"Cost: \t\t\t${cb.total_cost:.4f}\n")
```

<pre class="custom">1. Single Query Monitoring
    ----------------------------------------
    Response: The capital of France is Paris.
    ----------------------------------------
    Token Usage Statistics:
    Prompt Tokens: 		14
    Completion Tokens: 	8
    Total Tokens: 		22
    Cost: 			$0.0001
    
</pre>

```python
# 2. Multiple Queries Monitoring
print("2. Multiple Queries Monitoring")
print("-" * 40)

with get_openai_callback() as cb:
    # First query
    response1 = llm.invoke("What is Python?")
    # Second query
    response2 = llm.invoke("What is JavaScript?")

    print(f"Response 1: {response1.content[:100]}...")
    print("-" * 40)
    print(f"Response 2: {response2.content[:100]}...")
    print("-" * 40)
    print("Cumulative Statistics:")
    print(f"Total Prompt Tokens: \t\t{cb.prompt_tokens}")
    print(f"Total Completion Tokens: \t{cb.completion_tokens}")
    print(f"Total Tokens: \t\t\t{cb.total_tokens}")
    print(f"Total Cost: \t\t\t${cb.total_cost:.4f}\n")
```

<pre class="custom">2. Multiple Queries Monitoring
    ----------------------------------------
    Response 1: Python is a high-level, interpreted programming language known for its readability, simplicity, and ...
    ----------------------------------------
    Response 2: JavaScript is a high-level, dynamic, untyped, and interpreted programming language that is widely us...
    ----------------------------------------
    Cumulative Statistics:
    Total Prompt Tokens: 		23
    Total Completion Tokens: 	596
    Total Tokens: 			619
    Total Cost: 			$0.0060
    
</pre>
