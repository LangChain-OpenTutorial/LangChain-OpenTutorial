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

# Runtime Arguments Binding

- Author: [Wonyoung Lee](https://github.com/BaBetterB)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BaBetterB/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/10-Binding.ipynb) 
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb)


## Overview

This tutorial covers a scenario where, when calling a Runnable inside a Runnable sequence, we need to pass constant arguments that are not included in the output of the previous Runnable or user input. 
In such cases, `Runnable.bind()` can be used to easily pass these arguments.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Connecting OpenAI Functions](#connecting-openai-functions)
- [Connecting OpenAI Tools](#connecting-openai-tools)

### References


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
        "LANGCHAIN_PROJECT": "Runtime Arguments Binding",  # title
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

Use `RunnablePassthrough` to pass the `{equation_statement}` variable to the prompt, and use `StrOutputParser` to parse the model's output into a string, creating a `runnable` object.

The `runnable.invoke()` method is called to pass the equation statement "x raised to the third plus seven equals 12" and output the result.

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
    SOLUTION: x^3 = 12 - 7
    x^3 = 5
    x = ∛5
</pre>

Using bind() Method with Stop Word.
You may want to call the model using a specific `stop` word. 
`model.bind()` can be used to call the language model and stop the generation at the "SOLUTION" token.

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

One particularly useful way to use bind() is to connect OpenAI Functions with compatible OpenAI models.

Below is the code that defines `OpenAI Functions` according to a schema.


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

Binding the solver Function.
We use the `bind()` method to bind the function call named `solver` to the model.

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




<pre class="custom">AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{\n"equation": "x^3 + 7 = 12",\n"solution": "x = ∛5"\n}', 'name': 'solver'}, 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 28, 'prompt_tokens': 95, 'total_tokens': 123, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-6fa11a89-41ec-4316-8a75-beab094d7803-0', usage_metadata={'input_tokens': 95, 'output_tokens': 28, 'total_tokens': 123, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



## Connecting OpenAI Tools

Here’s how you can connect and use OpenAI tools.

The tools object helps you use various OpenAI features easily.

For example, by calling the `tool.run` method with a natural language question, the model can generate an answer to that question.

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

Binding Tools and Invoking the Model
- Use `bind()` to bind `tools` to the model.
- Call the `invoke()` method with a question like "Tell me the current weather in San Francisco, New York, and Los Angeles?"

```python
# Initialize the ChatOpenAI model and bind the tools.
model = ChatOpenAI(model="gpt-4o").bind(tools=tools)
# Invoke the model to ask about the weather in San Francisco, New York, and Los Angeles.
model.invoke(
    "Can you tell me the current weather in San Francisco, New York, and Los Angeles?"
)
```




<pre class="custom">AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_q8WYveGpmjfckyAU3TENgNMi', 'function': {'arguments': '{\n  "location": "San Francisco, CA"\n}', 'name': 'get_current_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 92, 'total_tokens': 112, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-9c4e4e91-244c-49d1-a54b-b0922a3bc228-0', tool_calls=[{'name': 'get_current_weather', 'args': {'location': 'San Francisco, CA'}, 'id': 'call_q8WYveGpmjfckyAU3TENgNMi', 'type': 'tool_call'}], usage_metadata={'input_tokens': 92, 'output_tokens': 20, 'total_tokens': 112, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>


