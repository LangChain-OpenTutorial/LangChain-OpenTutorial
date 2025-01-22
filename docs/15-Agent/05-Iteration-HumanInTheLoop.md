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

# Iteration-human-in-the-loop

- Author: [Wonyoung Lee](https://github.com/BaBetterB)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BaBetterB/LangChain-OpenTutorial/blob/main/15-Agent/05-Iteration-HumanInTheLoop.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/04-SemanticChunker.ipynb)


## Overview

This tutorial covers the functionality of repeating the agent's execution process or receiving user input to decide whether to proceed during intermediate steps. 

The feature of asking the user whether to continue during the agent's execution process is called human-in-the-loop . 

The `iter()` method creates an iterator that allows you to step through the agent's execution process step-by-step.


### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [AgentExecutor](#agentexecutor)



### References


- [LangChain ChatOpenAI API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [LangChain AgentExecutor API reference](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html)
- [LangSmith API reference](https://docs.smith.langchain.com/)

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
        "langchain_community",
        "load_dotenv",
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
        "LANGCHAIN_PROJECT": "Iteration-human-in-the-loop",  # title
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



First, define the tool.

```python
from langchain.agents import tool


@tool
def add_function(a: float, b: float) -> float:
    """Adds two numbers together."""

    return a + b
```

Next, define an agent that performs addition calculations using `add_function`.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Define tools
tools = [add_function]

# Create LLM
gpt = ChatOpenAI(model="gpt-4o-mini")

# Create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant."
            "Please avoid LaTeX-style formatting and use plain symbols.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create Agent
gpt_agent = create_tool_calling_agent(gpt, tools, prompt)

# Create AgentExecutor
agent_executor = AgentExecutor(
    agent=gpt_agent,
    tools=tools,
    verbose=False,
    max_iterations=10,
    handle_parsing_errors=True,
)
```

## AgentExecutor

This method creates an iterator (`AgentExecutorIterator` ) that allows you to step through the agent's execution process.

**Function Description**
The `iter()` method returns an `AgentExecutorIterator` object that provides sequential access to each step the agent takes until reaching the final output.

**Key Features**
- **Step-by-step execution access** : Enables you to examine the agent's execution process step-by-step.


**Flow Overview**

To perform the addition calculation for `"114.5 + 121.2 + 34.2 + 110.1"`, the steps are executed as follows:

1. 114.5 + 121.2 = 235.7
2. 235.7 + 34.2 = 269.9
3. 269.9 + 110.1 = 380.0

You can observe each step in this calculation process.

During this process, the system displays the intermediate calculation results to the user and asks if they want to continue. (**Human-in-the-loop**)

If the user inputs anything other than 'y', the iteration stops.

In practice, while calculating 114.5 + 121.2 = 235.7, 34.2 + 110.1 = 144.3 is also calculated simultaneously.

Then, the result of 235.7 + 144.3 = 380.0 is calculated as the second step.

This process can be observed when `verbose=True` is set in the `AgentExecutor`.



```python
# Define the user input question
question = "What is the result of 114.5 + 121.2 + 34.2 + 110.1?"


# Flag to track if the calculation is stopped
calculation_stopped = False

# Use AgentExecutor's iter() method to run step-by-step execution
for step in agent_executor.iter({"input": question}):
    # Access each calculation step through intermediate_step
    if output := step.get("intermediate_step"):
        action, value = output[0]

        # Print the result of each calculation step
        if action.tool == "add_function":
            print(f"Tool Name: {action.tool}, Execution Result: {value}")

        # Ask the user whether to continue
        while True:
            _continue = input("Do you want to continue? (y/n):").strip().lower()
            if _continue in ["y", "n"]:
                if _continue == "n":
                    print(f"Calculation stopped. Last computed result: {value}")
                    calculation_stopped = True  # Set flag to indicate calculation stop
                    break  # Break from the loop to stop calculation
                break  # Break the inner while loop after valid input
            else:
                print("Invalid input! Please enter 'y' or 'n'.")

    # Exit the iteration if the calculation is stopped
    if calculation_stopped:
        break

# Print the final result
if "output" in step:
    print(f"Final result: {step['output']}")
else:
    print(f"Final result (from last computation): {value}")
```

<pre class="custom">Tool Name: add_function, Execution Result: 235.7
    Tool Name: add_function, Execution Result: 380.0
    Final result: The result of 114.5 + 121.2 + 34.2 + 110.1 is 380.0.
</pre>
