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

# Binding

- Author: [Wonyoung Lee](https://github.com/BaBetterB)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BaBetterB/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/10-Binding.ipynb) 
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb)


## Overview

This tutorial covers a scenario where you need to pass constant arguments(not included in the output of the previous Runnable or user input) when calling a Runnable inside a Runnable sequence. In such cases, `Runnable.bind()` is a convenient way to pass these arguments


### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Runtime Arguments Binding](#runtime-arguments-binding)
- [Connecting OpenAI Functions](#connecting-openai-functions)
- [Connecting OpenAI Tools](#connecting-openai-tools)

### References

- [LangChain RunnablePassthrough API reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
- [LangChain ChatPromptTemplate API reference](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)

----

 


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [ `langchain-opentutorial` ](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

Load sample text and output the content.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

<pre class="custom">
    [notice] A new release of pip is available: 24.2 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

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

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Binding",  # title
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it.

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
# Configuration File for Managing API Keys as Environment Variables
from dotenv import load_dotenv

# Load API Key Information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Runtime Arguments Binding

This section explains how to use `Runnable.bind()` to pass constant arguments to a `Runnable` within a sequence, especially when those arguments aren't part of the previous Runnable's output or use input.

**Passing variables to prompts:**

1. Use `RunnablePassthrough` to pass the `{equation_statement}` variable to the prompt.
2. Use `StrOutputParser` to parse the model's output into a string, creating a `runnable` object.
3. Call the `runnable.invoke()` method to pass the equation statement (e.g., \"x raised to the third plus seven equals 12\") get the result.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # Write the following equation using algebraic symbols and then solve it.
            "Write out the following equation using algebraic symbols then solve it. "
            "Please avoid LaTeX-style formatting and use plain symbols."
            "Use the format:\n\nEQUATION:...\nSOLUTION:...\n",
        ),
        (
            "human",
            "{equation_statement}",  # Accepts the equation statement from the user as a variable.
        ),
    ]
)
# Initialize the ChatOpenAI model and set temperature to 0.
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Pass the equation statement to the prompt and parse the model's output as a string.
runnable = (
    {"equation_statement": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

# Input an example equation statement and print the result.
result = runnable.invoke("x raised to the third plus seven equals 12")
print(result)
```

<pre class="custom">EQUATION: x^3 + 7 = 12
    
    SOLUTION:
    1. Subtract 7 from both sides of the equation to isolate the x^3 term:
       x^3 + 7 - 7 = 12 - 7
       x^3 = 5
    
    2. Take the cube root of both sides to solve for x:
       x = 5^(1/3)
    
    Therefore, the solution is:
    x ≈ 1.71 (rounded to two decimal places)
</pre>

**Using bind() method with stop words**

For controlling the end of the model's output using a specific stop word, you can use `model.bind()` to instruct the model to halt its generation upon encountering the stop token like `SOLUTION`.

```python
runnable = (
    # Create a runnable passthrough object and assign it to the "equation_statement" key.
    {"equation_statement": RunnablePassthrough()}
    | prompt  # Add the prompt to the pipeline.
    | model.bind(
        stop="SOLUTION"
    )  # Bind the model and set it to stop generating at the "SOLUTION" token.
    | StrOutputParser()  # Add the string output parser to the pipeline.
)
# Execute the pipeline with the input "x raised to the third plus seven equals 12" and print the result.
print(runnable.invoke("x raised to the third plus seven equals 12"))
```

<pre class="custom">EQUATION: x^3 + 7 = 12
    
    
</pre>

## Connecting OpenAI Functions

`bind()` is particularly useful for connecting OpenAI Functions with compatible OpenAI models.

Let's define `openai_function` according to a schema.

```python
openai_function = {
    "name": "solver",  # Function name
    # Function description: Formulate and solve an equation.
    "description": "Formulates and solves an equation",
    "parameters": {  # Function parameters
        "type": "object",  # Parameter type: object
        "properties": {  # Parameter properties
            "equation": {  # Equation property
                "type": "string",  # Type: string
                "description": "The algebraic expression of the equation",  # Description
            },
            "solution": {  # Solution property
                "type": "string",  # Type: string
                "description": "The solution to the equation",  # Description
            },
        },
        "required": [
            "equation",
            "solution",
        ],  # Required parameters: equation and solution
    },
}
```

**Binding a solver function.**

We can then use the `bind()` method to associate a function call (like `solver`) with the language model.

```python
# Write the following equation using algebraic symbols and then solve it
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write out the following equation using algebraic symbols then solve it.",
        ),
        ("human", "{equation_statement}"),
    ]
)


model = ChatOpenAI(model="gpt-4o", temperature=0).bind(
    function_call={"name": "solver"},  # Bind the OpenAI function schema
    functions=[openai_function],
)


runnable = {"equation_statement": RunnablePassthrough()} | prompt | model


# Equation: x raised to the third plus seven equals 12


runnable.invoke("x raised to the third plus seven equals 12")
```




<pre class="custom">AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"equation":"x^3 + 7 = 12","solution":"x^3 = 12 - 7; x^3 = 5; x = 5^(1/3)"}', 'name': 'solver'}, 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 44, 'prompt_tokens': 95, 'total_tokens': 139, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4', 'finish_reason': 'stop', 'logprobs': None}, id='run-bb333525-2117-4a09-bf1c-c6bdca21b50c-0', usage_metadata={'input_tokens': 95, 'output_tokens': 44, 'total_tokens': 139, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



## Connecting OpenAI Tools

This section explains how to connect and use OpenAI tools within your LangChain applications.
The `tools` object simplifies using various OpenAI features.
For example, calling the `tool.run` method with a natural language query allows the model to utilize the spcified tool to generate a response.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",  # Function name to get current weather
            "description": "Fetches the current weather for a given location",  # Description
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g.: San Francisco, CA",  # Location description
                    },
                    # Temperature unit
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],  # Required parameter: location
            },
        },
    }
]
```

**Binding tools and invoking the model:**

1. Use `bind()` to associate `tools` with the language model.
2. Call the `invoke()` method on the bound model, providing a natural language question as input.


```python
# Initialize the ChatOpenAI model and bind the tools.
model = ChatOpenAI(model="gpt-4o").bind(tools=tools)
# Invoke the model to ask about the weather in San Francisco, New York, and Los Angeles.
model.invoke(
    "Can you tell me the current weather in San Francisco, New York, and Los Angeles?"
)
```




<pre class="custom">AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ixydz9CyFSyB0LHUATftfdXA', 'function': {'arguments': '{"location": "San Francisco, CA"}', 'name': 'get_current_weather'}, 'type': 'function'}, {'id': 'call_VFAGF4YanFQVg1lJQ9x1miR3', 'function': {'arguments': '{"location": "New York, NY"}', 'name': 'get_current_weather'}, 'type': 'function'}, {'id': 'call_RMYL8pWFlMarWkOPkigKQSug', 'function': {'arguments': '{"location": "Los Angeles, CA"}', 'name': 'get_current_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 71, 'prompt_tokens': 90, 'total_tokens': 161, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-50665b85-c055-413a-8a40-3c1070aa3c45-0', tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'San Francisco, CA'}, 'id': 'call_ixydz9CyFSyB0LHUATftfdXA', 'type': 'tool_call'}, {'name': 'get_current_weather', 'args': {'location': 'New York, NY'}, 'id': 'call_VFAGF4YanFQVg1lJQ9x1miR3', 'type': 'tool_call'}, {'name': 'get_current_weather', 'args': {'location': 'Los Angeles, CA'}, 'id': 'call_RMYL8pWFlMarWkOPkigKQSug', 'type': 'tool_call'}], usage_metadata={'input_tokens': 90, 'output_tokens': 71, 'total_tokens': 161, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>


