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

# Bind Tools

- Author: [Jaemin Hong](https://github.com/geminii01)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial introduces `bind_tools` , a powerful function in LangChain for integrating custom tools with LLMs.

It aims to demonstrate how to create, bind, and execute tools seamlessly, enabling enriched AI-driven workflows.

Through this guide, you'll learn to bind tools, parse and execute outputs, and integrate them into an `AgentExecutor` .

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Tool Creation](#tool-creation)
- [Tool Binding](#tool-binding)
- [bind_tools + Parser + Execution](#bind_tools-+-parser-+-execution)
- [bind_tools to Agent & AgentExecutor](#bind_tools-to-agent-&-agentexecutor)

### References

- [Conceptual guide - Tool calling](https://python.langchain.com/docs/concepts/tool_calling/)
- [tool_calls](https://python.langchain.com/docs/concepts/tool_calling/#tool-calling-1)
- [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor)
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
    [
        "langsmith",
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "02-Bind-Tools",
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



## Tool Creation

Define tools for experimentation:

- `get_word_length` : Returns the length of a word
- `add_function` : Adds two numbers
- `bbc_news_crawl` : Crawls BBC news and extracts main content

[Note]

- Use the `@tool` decorator for defining tools, and provide clear English docstrings.

```python
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# Define the tools
@tool
def get_word_length(word: str) -> int:
    """Return the length of the given text"""
    return len(word)


@tool
def add_function(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@tool
def bbc_news_crawl(news_url: str) -> str:
    """Crawl a news article from BBC"""
    response = requests.get(news_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the desired information from the article
        article = soup.find("article")
        if article:
            title = article.find("h1").get_text()  # Extract the title
            content_list = [
                tag.get_text()
                for tag in article.find_all(["h2", "p"])
                if (tag.name == "h2" and "sc-518485e5-0" in tag.get("class", []))
                or (tag.name == "p" and "sc-eb7bd5f6-0" in tag.get("class", []))
            ]  # Extract the content
            content = "\n\n".join(content_list)
    else:
        print(f"HTTP request failed. Response code: {response.status_code}")
    return f"{title}\n\n----------\n\n{content}"


tools = [get_word_length, add_function, bbc_news_crawl]
```

## Tool Binding

Use the `bind_tools` function to bind the tools to an LLM model.

```python
from langchain_openai import ChatOpenAI

# Create a model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Tool binding
llm_with_tools = llm.bind_tools(tools)
```

Let's check the result!

The results are stored in `tool_calls` . Therefore, let's print `tool_calls` .

[Note]

- `name` is the name of the tool.
- `args` are the arguments passed to the tool.

```python
# Execution result
llm_with_tools.invoke(
    "What is the length of the given text 'LangChain OpenTutorial'?"
).tool_calls
```




<pre class="custom">[{'name': 'get_word_length',
      'args': {'word': 'LangChain OpenTutorial'},
      'id': 'call_km7ieeNgjOvbPEfPt3bwO4cy',
      'type': 'tool_call'}]</pre>



Next, we connect `llm_with_tools` with `JsonOutputToolsParser` to parse `tool_calls` and review the results.

[Note]

- `type` is the name of the tool.
- `args` are the arguments passed to the tool.

```python
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

# Tool Binding + Tool Parser
chain = llm_with_tools | JsonOutputToolsParser(tools=tools)

# Execution Result
tool_call_results = chain.invoke(
    "What is the length of the given text 'LangChain OpenTutorial'?"
)
print(tool_call_results)
```

<pre class="custom">[{'args': {'word': 'LangChain OpenTutorial'}, 'type': 'get_word_length'}]
</pre>

```python
print(tool_call_results)
print("\n==========\n")

# First tool call result
single_result = tool_call_results[0]

print(single_result["type"])
print(single_result["args"])
```

<pre class="custom">[{'args': {'word': 'LangChain OpenTutorial'}, 'type': 'get_word_length'}]
    
    ==========
    
    get_word_length
    {'word': 'LangChain OpenTutorial'}
</pre>

Execute the tool matching the tool name.

```python
tool_call_results[0]["type"], tools[0].name
```




<pre class="custom">('get_word_length', 'get_word_length')</pre>



The `execute_tool_calls` function identifies the appropriate tool, passes the corresponding `args` , and then executes the tool.

```python
def execute_tool_calls(tool_call_results):
    """
    Function to execute the tool call results.

    :param tool_call_results: List of the tool call results
    :param tools: List of available tools
    """

    # Iterate over the list of the tool call results
    for tool_call_result in tool_call_results:
        # Tool name (function name)
        tool_name = tool_call_result["type"]
        # Tool arguments
        tool_args = tool_call_result["args"]

        # Find the tool that matches the name and execute it
        # Use the next() function to find the first matching tool
        matching_tool = next((tool for tool in tools if tool.name == tool_name), None)
        if matching_tool:
            # Execute the tool
            result = matching_tool.invoke(tool_args)
            print(
                f"[Executed Tool] {tool_name} [Args] {tool_args}\n[Execution Result] {result}"
            )
        else:
            print(f"Warning: Unable to find the tool corresponding to {tool_name}.")


# Execute the tool calls
execute_tool_calls(tool_call_results)
```

<pre class="custom">[Executed Tool] get_word_length [Args] {'word': 'LangChain OpenTutorial'}
    [Execution Result] 22
</pre>

## bind_tools + Parser + Execution

This time, the entire process will be executed in one step.

- `llm_with_tools` : The LLM model with bound tools
- `JsonOutputToolsParser` : The parser that processes the results of tool calls
- `execute_tool_calls` : The function that executes the results of tool calls

[Flow Summary]

1. Bind tools to the model
2. Parse the results of tool calls
3. Execute the results of tool calls

```python
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

# bind_tools + Parser + Execution
chain = llm_with_tools | JsonOutputToolsParser(tools=tools) | execute_tool_calls
```

```python
# Execution Result 1
chain.invoke("What is the length of the given text 'LangChain OpenTutorial'?")
```

<pre class="custom">[Executed Tool] get_word_length [Args] {'word': 'LangChain OpenTutorial'}
    [Execution Result] 22
</pre>

```python
# Execution Result 2
chain.invoke("114.5 + 121.2")

# Double check
print(114.5 + 121.2)
```

<pre class="custom">[Executed Tool] add_function [Args] {'a': 114.5, 'b': 121.2}
    [Execution Result] 235.7
    235.7
</pre>

```python
# Execution Result 3
chain.invoke("Crawl the news article: https://www.bbc.com/news/articles/cew52g8p2lko")
```

<pre class="custom">[Executed Tool] bbc_news_crawl [Args] {'news_url': 'https://www.bbc.com/news/articles/cew52g8p2lko'}
    [Execution Result] New AI hub 'to create 1,000 jobs' on Merseyside
    
    ----------
    
    A new Artificial Intelligence (AI) hub planned for Merseyside is set to create 1,000 jobs over the next three years, the government said.
    
    Prime Minister Sir Keir Starmer said he wanted to make the UK one of the world's AI "super powers" as a way of boosting economic growth and improving public services.
    
    Global IT company Kyndryl announced it was going to create the new tech hub in the Liverpool City Region.
    
    Metro Mayor Steve Rotheram welcomed the investment, saying it would be "hugely beneficial" to the area.
    
    'International investment'
    
    In a speech setting out the government's AI ambitions, Starmer spoke of its "vast potential" for rejuvenating public services.
    
    The government said its AI Opportunities Action Plan was backed by leading tech firms, some of which have committed £14bn towards various projects including growth zones, creating 13,250 jobs.
    
    Mr Rotheram told BBC Radio Merseyside: "I went over last year to speak to [Kyndryl] face-to-face in New York.
    
    "To have that come to fruition so quickly is hugely beneficial to the workforce in the Liverpool City Region." 
    
    He said attracting the world's largest IT infrastructure services provider was "testament to what we can achieve when local ambition is matched by national support to help attract international investment".
    
    The Labour mayor said the Liverpool City Region was "leading the way in the UK's AI revolution". 
    
    He added: "As a region with a proud history of innovation we're ready to seize the opportunities that AI and digital technology can bring; not just to boost our economy but to improve lives, develop skills, tackle inequality, and ensure no-one is left behind."
    
    The BBC has asked the Department for Science, Innovation and Technology for more details about Merseyside's AI hub plans.
    
    Listen to the best of BBC Radio Merseyside on Sounds and follow BBC Merseyside on Facebook, X, and Instagram and watch BBC North West Tonight on BBC iPlayer.
</pre>

## bind_tools to Agent & AgentExecutor

`bind_tools` provides schemas (tools) that can be used by the model.

`AgentExecutor` creates an execution loop for tasks such as invoking the LLM, routing to the appropriate tool, executing it, and re-invoking the model.

[Note]

- `Agent` and `AgentExecutor` will be covered in detail in the *next chapter* .

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Create an Agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create a model
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Use the tools defined previously
tools = [get_word_length, add_function, bbc_news_crawl]

# Create an Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)
```

Let's try calculating the length of the word.

```python
# Execute the Agent
result = agent_executor.invoke(
    {"input": "What is the length of the given text 'LangChain OpenTutorial'?"}
)

# Execution Result
print(result["output"])
```

<pre class="custom">
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `get_word_length` with `{'word': 'LangChain OpenTutorial'}`
    
    
    [0m[36;1m[1;3m22[0m[32;1m[1;3mThe length of the text "LangChain OpenTutorial" is 22 characters.[0m
    
    [1m> Finished chain.[0m
    The length of the text "LangChain OpenTutorial" is 22 characters.
</pre>

Let's try calculating the result of two numbers.

```python
# Execute the Agent
result = agent_executor.invoke({"input": "Calculate the result of 114.5 + 121.2"})

# Execution Result
print(result["output"])
print("\n==========\n")
print(114.5 + 121.2)
```

<pre class="custom">
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `add_function` with `{'a': 114.5, 'b': 121.2}`
    
    
    [0m[33;1m[1;3m235.7[0m[32;1m[1;3mThe result of 114.5 + 121.2 is 235.7.[0m
    
    [1m> Finished chain.[0m
    The result of 114.5 + 121.2 is 235.7.
    
    ==========
    
    235.7
</pre>

Let's try adding more than two numbers. 

In this process, you can observe that the agent verifies its own results and repeats the process if necessary.

```python
# Execute the Agent
result = agent_executor.invoke(
    {"input": "Calculate the result of 114.5 + 121.2 + 34.2 + 110.1"}
)

# Execution Result
print(result["output"])
print("\n==========\n")
print(114.5 + 121.2 + 34.2 + 110.1)
```

<pre class="custom">
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `add_function` with `{'a': 114.5, 'b': 121.2}`
    
    
    [0m[33;1m[1;3m235.7[0m[32;1m[1;3m
    Invoking: `add_function` with `{'a': 235.7, 'b': 34.2}`
    
    
    [0m[33;1m[1;3m269.9[0m[32;1m[1;3m
    Invoking: `add_function` with `{'a': 34.2, 'b': 110.1}`
    
    
    [0m[33;1m[1;3m144.3[0m[32;1m[1;3m
    Invoking: `add_function` with `{'a': 269.9, 'b': 110.1}`
    
    
    [0m[33;1m[1;3m380.0[0m[32;1m[1;3mThe result of adding 114.5, 121.2, 34.2, and 110.1 is 380.0.[0m
    
    [1m> Finished chain.[0m
    The result of adding 114.5, 121.2, 34.2, and 110.1 is 380.0.
    
    ==========
    
    380.0
</pre>

Let's try summarizing the news article.

```python
# Execute the Agent
result = agent_executor.invoke(
    {
        "input": "Summarize the news article: https://www.bbc.com/news/articles/cew52g8p2lko"
    }
)

# Execution Result
print(result["output"])
```

<pre class="custom">
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `bbc_news_crawl` with `{'news_url': 'https://www.bbc.com/news/articles/cew52g8p2lko'}`
    
    
    [0m[38;5;200m[1;3mNew AI hub 'to create 1,000 jobs' on Merseyside
    
    ----------
    
    A new Artificial Intelligence (AI) hub planned for Merseyside is set to create 1,000 jobs over the next three years, the government said.
    
    Prime Minister Sir Keir Starmer said he wanted to make the UK one of the world's AI "super powers" as a way of boosting economic growth and improving public services.
    
    Global IT company Kyndryl announced it was going to create the new tech hub in the Liverpool City Region.
    
    Metro Mayor Steve Rotheram welcomed the investment, saying it would be "hugely beneficial" to the area.
    
    'International investment'
    
    In a speech setting out the government's AI ambitions, Starmer spoke of its "vast potential" for rejuvenating public services.
    
    The government said its AI Opportunities Action Plan was backed by leading tech firms, some of which have committed £14bn towards various projects including growth zones, creating 13,250 jobs.
    
    Mr Rotheram told BBC Radio Merseyside: "I went over last year to speak to [Kyndryl] face-to-face in New York.
    
    "To have that come to fruition so quickly is hugely beneficial to the workforce in the Liverpool City Region." 
    
    He said attracting the world's largest IT infrastructure services provider was "testament to what we can achieve when local ambition is matched by national support to help attract international investment".
    
    The Labour mayor said the Liverpool City Region was "leading the way in the UK's AI revolution". 
    
    He added: "As a region with a proud history of innovation we're ready to seize the opportunities that AI and digital technology can bring; not just to boost our economy but to improve lives, develop skills, tackle inequality, and ensure no-one is left behind."
    
    The BBC has asked the Department for Science, Innovation and Technology for more details about Merseyside's AI hub plans.
    
    Listen to the best of BBC Radio Merseyside on Sounds and follow BBC Merseyside on Facebook, X, and Instagram and watch BBC North West Tonight on BBC iPlayer.[0m[32;1m[1;3mA new Artificial Intelligence (AI) hub is planned for Merseyside, expected to create 1,000 jobs over the next three years. Prime Minister Sir Keir Starmer aims to position the UK as a global AI "superpower" to boost economic growth and improve public services. The global IT company Kyndryl will establish the tech hub in the Liverpool City Region. Metro Mayor Steve Rotheram praised the investment, highlighting its benefits for the area. The government's AI Opportunities Action Plan, supported by leading tech firms, has secured £14 billion for various projects, including growth zones, creating 13,250 jobs. Rotheram emphasized the region's leadership in the UK's AI revolution and its readiness to leverage AI and digital technology for economic and social benefits.[0m
    
    [1m> Finished chain.[0m
    A new Artificial Intelligence (AI) hub is planned for Merseyside, expected to create 1,000 jobs over the next three years. Prime Minister Sir Keir Starmer aims to position the UK as a global AI "superpower" to boost economic growth and improve public services. The global IT company Kyndryl will establish the tech hub in the Liverpool City Region. Metro Mayor Steve Rotheram praised the investment, highlighting its benefits for the area. The government's AI Opportunities Action Plan, supported by leading tech firms, has secured £14 billion for various projects, including growth zones, creating 13,250 jobs. Rotheram emphasized the region's leadership in the UK's AI revolution and its readiness to leverage AI and digital technology for economic and social benefits.
</pre>
