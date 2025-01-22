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

# RunnableLambda

- Author: [Kenny Jung](https://www.linkedin.com/in/kwang-yong-jung)
- Design:
- Peer Review: [Junseong Kim](https://www.linkedin.com/in/%EC%A4%80%EC%84%B1-%EA%B9%80-591b351b2/), [Haseom Shin](https://github.com/IHAGI-c)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/03-RunnableLambda.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/03-RunnableLambda.ipynb)


## Overview

`RunnableLambda` provides a way to **integrate custom functions** into your LangChain pipeline.

This allows you to **define your own custom functions** and execute them using `RunnableLambda` as part of their workflow.

For example, you can build your own logic and execute the functions, performing different tasks such as:
- Data preprocessing,
- Calculations,
- Interactions with external APIs, and
- Any other custom logic you need in your chain.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [How to Execute Custom Functions](#how-to-execute-custom-functions)
- [Using RunnableConfig as Parameters](#using-runnableconfig-as-parameters)

### References

- [LangChain Python API Reference > langchain: 0.3.29 > runnables > RunnableLambda](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)
- [LangChain Python API Reference > langchain: 0.3.29 > runnables > RunnableConfig](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig)
- [LangChain Python API Reference > docs > concepts > runnables > Configurable Runnables](https://python.langchain.com/docs/concepts/runnables/#configurable-runnables)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The `langchain-opentutorial` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_core",
        "langchain_openai",
    ],
    verbose=False,
    upgrade=False,
)
```

Alternatively, you can set and load `OPENAI_API_KEY` from a `.env` file.

**[Note]** This is only necessary if you haven't already set `OPENAI_API_KEY` in previous steps.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "ConversationBufferWindowMemory",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## How to Execute Custom Functions

**Important Note**

A limitation of `RunnableLambda` is that **custom functions can only accept a single argument**. You can wrap custom functions with `RunnableLambda` to use them in your pipeline.

If you have a function that requires multiple parameters, create a wrapper function:
1. Accepts a single input (typically a dictionary).
2. Unpacks this input into multiple arguments inside the wrapper.
3. Passes these arguments to your original function.

For example:

```python
# Won't work with RunnableLambda
def original_function(arg1, arg2, arg3):
    pass

# Will work with RunnableLambda
def wrapper_function(input_dict):
    return original_function(
            input_dict['arg1'],
            input_dict['arg2'],
            input_dict['arg3']
)
```

```python
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# Function for returning the length of the text
def length_function(text):
    return len(text)


# Function for multiplying the length of two texts
def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


# Wrapper function for connecting the function that receives 2 arguments
def multiple_length_function(
    _dict,
):  # Function for multiplying the length of two texts
    return _multiple_length_function(_dict["text1"], _dict["text2"])


# Create a prompt template
prompt = ChatPromptTemplate.from_template("what is {a} + {b}?")
# Initialize the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Connect the prompt and model to create a chain
chain1 = prompt | model

# Chain configuration
chain = (
    {
        "a": itemgetter("input_1") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("input_1"), "text2": itemgetter("input_2")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

Execute the chain and verify the result.

```python
# Execute the chain with the given arguments.
chain.invoke({"input_1": "bar", "input_2": "gah"})
```




<pre class="custom">'3 + 9 equals 12.'</pre>



## Using RunnableConfig as Parameters

`RunnableLambda` can optionally accept a [RunnableConfig](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig) object.

This allows you to pass various configuration options to nested executions, including:
- Callbacks: For tracking and monitoring function execution.
- Tags: For labeling and organizing different runs.
- Other configuration: Additional settings to control function behavior.

For example, you can use `RunnableConfig` to:
- Track function performance.
- Add logging capabilities.
- Group related operations using tags.
- Configure error handling and retry logic.
- Set timeouts and other execution parameters.

This makes `RunnableLambda` highly configurable for complex workflows with fine-grained control over execution.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
import json


def parse_or_fix(text: str, config: RunnableConfig):
    # Create a prompt template for fixing the next text
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Fix the following text:\n\ntext\n{input}\n\nError: {error}"
            " Don't narrate, just respond with the fixed data."
        )
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    # Try up to 3 times
    for _ in range(3):
        try:
            # Parse the text as JSON
            return json.loads(text)
        except Exception as e:
            # If parsing fails, call the fixing chain to fix the text
            text = fixing_chain.invoke({"input": text, "error": e}, config)
            print(f"config: {config}")
    # If parsing fails, return "Failed to parse"
    return "Failed to parse"
```

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    # Call the parse_or_fix function using RunnableLambda
    output = RunnableLambda(parse_or_fix).invoke(
        input="{foo:: bar}",
        config={"tags": ["my-tag"], "callbacks": [cb]},  # Pass the config
    )
    # Print the modified result
    print(f"\n\nModified result:\n{output}")
```

<pre class="custom">config: {'tags': ['my-tag'], 'metadata': {}, 'callbacks': <langchain_core.callbacks.manager.CallbackManager object at 0x12d7f7250>, 'recursion_limit': 25, 'configurable': {}}
    
    
    Modified result:
    {'foo': 'bar'}
</pre>

```python
# Check the output
print(output)
```

<pre class="custom">{'foo': 'bar'}
</pre>

```python
# Check the callback
print(cb)
```

<pre class="custom">Tokens Used: 59
    	Prompt Tokens: 52
    		Prompt Tokens Cached: 0
    	Completion Tokens: 7
    		Reasoning Tokens: 0
    Successful Requests: 1
    Total Cost (USD): $1.1999999999999997e-05
</pre>
