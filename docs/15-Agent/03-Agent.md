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

# Tool Calling Agent

- Author: [Kenny Jung](https://www.linkedin.com/in/kwang-yong-jung)
- Design:
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/15-Agent/03-Agent.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/15-Agent/03-Agent.ipynb)


## Overview

Tool calling allows models to detect when one or more **tools** need to be **called and what inputs should be passed** to those tools.
 
In API calls, you can describe tools and intelligently choose to have the model output structured objects like JSON that contain arguments for calling these tools.
 
The goal of the tools API is to return valid and useful **tool calls** more reliably than what could be accomplished using plain text completion or chat APIs.
 
By combining this structured output with the ability to bind multiple tools to a tool-calling chat model and letting the model choose which tools to call, you can create agents that iteratively call tools and receive results until a query is resolved.
 
This is a more **generalized version** of the OpenAI tools agent that was designed specifically for OpenAI's particular tool-calling style.
 
This agent uses LangChain's ToolCall interface to support a wider range of provider implementations beyond OpenAI, including `Anthropic` , `Google Gemini` , and `Mistral` .


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Creating Tools](#creating-tools)
- [Creating Agent Prompt](#creating-agent-prompt)
- [Creating Agent](#creating-agent)
- [AgentExecutor](#agentexecutor)
- [Checking step-by-step results using Stream output](#checking-step-by-step-results-using-stream-output)
- [Customizing intermediate steps output using user-defined functions](#customizing-intermediate-steps-output-using-user-defined-functions)
- [Communicating Agent with previous conversation history](#communicating-agent-with-previous-conversation-history)

### References

- [LangChain Python API Reference > langchain: 0.3.14 > agents > create_tool_calling_agent](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html#create-tool-calling-agent)
- [LangChain Python API Reference > langchain: 0.3.14 > core > runnables > langchain_core.runnables.history > RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)

![](./assets/15-agent-agent-concept.png)

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
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "ToolCallingAgent",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it.

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Creating Tools

- Creating tools for searching news and executing python code
- `@tool` decorator is used to create a tool
- `TavilySearchResults` is a tool for searching news
- `PythonREPL` is a tool for executing python code


```python
from langchain.tools import tool
from typing import List, Dict, Annotated
from langchain_community.tools import TavilySearchResults
from langchain_experimental.utilities import PythonREPL


# Creating tool for searching news
@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search news by input keyword using Tavily Search API"""
    news_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        # search_depth="advanced",
        # include_domains = [],
        # exclude_domains = []
    )
    return news_tool.invoke(query, k=3)


# Creating tool for executing python code
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this tool to execute Python code. If you want to see the output of a value,
    you should print it using print(...). This output is visible to the user."""
    result = ""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        print(f"Failed to execute. Error: {repr(e)}")
    finally:
        return result


print(f"Tool name: {search_news.name}")
print(f"Tool description: {search_news.description}")
print(f"Tool name: {python_repl_tool.name}")
print(f"Tool description: {python_repl_tool.description}")
```

<pre class="custom">Tool name: search_news
    Tool description: Search news by input keyword using Tavily Search API
    Tool name: python_repl_tool
    Tool description: Use this tool to execute Python code. If you want to see the output of a value,
        you should print it using print(...). This output is visible to the user.
</pre>

```python
# Creating tools
tools = [search_news, python_repl_tool]
```

## Creating Agent Prompt

- `chat_history` : variable for storing previous conversation (if multi-turn is not supported, it can be omitted.)
- `agent_scratchpad` : variable for storing temporary variables
- `input` : user's input

```python
from langchain_core.prompts import ChatPromptTemplate

# Creating prompt
# Prompt is a text that describes the task the model should perform. (input the name and role of the tool)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `search_news` tool for searching keyword related news.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
```

## Creating Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent

# Creating LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Creating Agent
agent = create_tool_calling_agent(llm, tools, prompt)
```

## AgentExecutor

- AgentExecutor is a class for running an agent that uses tools.

**Key properties**
- `agent` : agent that creates plans and decides actions at each step of the execution loop
- `tools` : list of valid tools that the agent can use
- `return_intermediate_steps` : whether to return the intermediate steps of the agent with the final output
- `max_iterations` : maximum number of steps before terminating the execution loop
- `max_execution_time` : maximum time the execution loop can take
- `early_stopping_method` : method to use when the agent does not return `AgentFinish` . ("force" or "generate")
  - `"force"` : returns a string indicating that the execution loop was stopped due to time or iteration limit.
  - `"generate"` : calls the agent's LLM chain once to generate the final answer based on the previous steps.
- `handle_parsing_errors` : Method of handling parsing errors. (True, False, or error handling function)
- `trim_intermediate_steps` : Method of trimming intermediate steps. (-1 trim not, or trimming function)

**Key methods**
1. `invoke` : Run the agent
2. `stream` : Stream the steps needed to reach the final output

**Key features**
1. **Tool validation** : Check if the tool is compatible with the agent
2. **Execution control** : Set maximum number of iterations and execution time limit
3. **Error handling** : Various processing options for output parsing errors
4. **Intermediate step management** : Trimming intermediate steps and returning options
5. **Asynchronous support** : Asynchronous execution and streaming support

**Optimization tips**
- Set `max_iterations` and `max_execution_time` appropriately to manage execution time
- Use `trim_intermediate_steps` to optimize memory usage
- For complex tasks, use the `stream` method to monitor step-by-step results

```python
from langchain.agents import AgentExecutor

# Create AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    max_execution_time=10,
    handle_parsing_errors=True,
)

# Run AgentExecutor
result = agent_executor.invoke({"input": "Search news about AI Agent in 2025."})

print("Agent execution result:")
print(result["output"])
```

<pre class="custom">
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `search_news` with `{'query': 'AI Agent 2025'}`
    
    
    [0m[36;1m[1;3m[{'url': 'https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/', 'content': 'In a similar study, Deloitte forecasts that 25% of enterprises using GenAI will deploy AI Agents by 2025, growing to 50% by 2027. Meanwhile, Gartner predicts that by 2028, at least 15% of day-to-day work decisions will be made autonomously through agentic AI. It also states that by then, 33% of enterprise software applications will also include'}, {'url': 'https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents', 'content': 'Next year will be the year of AI agents | TechTarget This will make the AI agent more accurate in completing its task, Greene said. Other than the rise of single-task AI agents, 2025 may also be the year of building the infrastructure for AI agents, said Olivier Blanchard, an analyst with Futurum Group. "2025 isn\'t going to be the year when we see a fully developed agentic AI," he said. AI agents need an orchestration layer that works across different platforms and devices, Blanchard said. Because data is usually spread across different sources and processes, it might be challenging to give AI agents the data they need to perform the tasks they\'re being asked to do, Greene said.'}, {'url': 'https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks', 'content': 'According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents.'}][0m[32;1m[1;3mHere are some recent news articles discussing the future of AI agents in 2025:
    
    1. **AI Agent Trends**  
       - **Source**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/)  
       - **Summary**: A study by Deloitte forecasts that 25% of enterprises using Generative AI will deploy AI agents by 2025, with this number expected to grow to 50% by 2027. Gartner predicts that by 2028, at least 15% of day-to-day work decisions will be made autonomously through agentic AI.
    
    2. **Next Year Will Be the Year of AI Agents**  
       - **Source**: [TechTarget](https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents)  
       - **Summary**: Analysts suggest that 2025 may focus on building the infrastructure necessary for AI agents. While single-task AI agents will rise, the development of an orchestration layer that works across various platforms and devices will be crucial for their effectiveness.
    
    3. **Predictions for AI in 2025**  
       - **Source**: [Stanford Institute for Human-Centered AI](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)  
       - **Summary**: Experts predict a shift towards collaborative AI systems where multiple specialized agents work together, guided by humans. This includes the development of multimodal AI models in education and research, where AI agents collaborate on complex tasks with human oversight.
    
    These articles highlight the anticipated growth and evolution of AI agents, emphasizing collaboration, infrastructure development, and the integration of AI into enterprise decision-making processes.[0m
    
    [1m> Finished chain.[0m
    Agent execution result:
    Here are some recent news articles discussing the future of AI agents in 2025:
    
    1. **AI Agent Trends**  
       - **Source**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/)  
       - **Summary**: A study by Deloitte forecasts that 25% of enterprises using Generative AI will deploy AI agents by 2025, with this number expected to grow to 50% by 2027. Gartner predicts that by 2028, at least 15% of day-to-day work decisions will be made autonomously through agentic AI.
    
    2. **Next Year Will Be the Year of AI Agents**  
       - **Source**: [TechTarget](https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents)  
       - **Summary**: Analysts suggest that 2025 may focus on building the infrastructure necessary for AI agents. While single-task AI agents will rise, the development of an orchestration layer that works across various platforms and devices will be crucial for their effectiveness.
    
    3. **Predictions for AI in 2025**  
       - **Source**: [Stanford Institute for Human-Centered AI](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)  
       - **Summary**: Experts predict a shift towards collaborative AI systems where multiple specialized agents work together, guided by humans. This includes the development of multimodal AI models in education and research, where AI agents collaborate on complex tasks with human oversight.
    
    These articles highlight the anticipated growth and evolution of AI agents, emphasizing collaboration, infrastructure development, and the integration of AI into enterprise decision-making processes.
</pre>

## Checking step-by-step results using Stream output

We will use the `stream()` method of AgentExecutor to stream the intermediate steps of the agent.

The output of `stream()` alternates between (Action, Observation) pairs, and finally ends with the agent's answer if the goal is achieved.

It will look like the following.

1. Action output
2. Observation output
3. Action output
4. Observation output

... (Continue until the goal is achieved) ...

Then, the agent will output the final answer if the goal is achieved.

The content of this output is summarized as follows.

| Output | Content |
|--------|----------|
| Action | `actions`: AgentAction or its subclass<br>`messages`: Chat messages corresponding to the action call |
| Observation | `steps`: Record of the agent's work including the current action and its observation<br>`messages`: Chat messages including the function call result (i.e., observation) |
| Final Answer | `output`: AgentFinish<br>`messages`: Chat messages including the final output |
```

```python
from langchain.agents import AgentExecutor

# Create AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
)
```

```python
# Run in streaming mode
result = agent_executor.stream({"input": "Search news about AI Agent in 2025."})

for step in result:
    # Print intermediate steps
    print(step)
    print("===" * 20)
```

<pre class="custom">{'actions': [ToolAgentAction(tool='search_news', tool_input={'query': 'AI Agent 2025'}, log="\nInvoking: `search_news` with `{'query': 'AI Agent 2025'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'function': {'arguments': '{"query":"AI Agent 2025"}', 'name': 'search_news'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c'}, id='run-fc21b730-a5ac-4f0a-b009-585acca6519c', tool_calls=[{'name': 'search_news', 'args': {'query': 'AI Agent 2025'}, 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'search_news', 'args': '{"query":"AI Agent 2025"}', 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='call_xxsGXV6JZgYb7JAb0JU9EQ7o')], 'messages': [AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'function': {'arguments': '{"query":"AI Agent 2025"}', 'name': 'search_news'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c'}, id='run-fc21b730-a5ac-4f0a-b009-585acca6519c', tool_calls=[{'name': 'search_news', 'args': {'query': 'AI Agent 2025'}, 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'search_news', 'args': '{"query":"AI Agent 2025"}', 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'index': 0, 'type': 'tool_call_chunk'}])]}
    ============================================================
    {'steps': [AgentStep(action=ToolAgentAction(tool='search_news', tool_input={'query': 'AI Agent 2025'}, log="\nInvoking: `search_news` with `{'query': 'AI Agent 2025'}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'function': {'arguments': '{"query":"AI Agent 2025"}', 'name': 'search_news'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c'}, id='run-fc21b730-a5ac-4f0a-b009-585acca6519c', tool_calls=[{'name': 'search_news', 'args': {'query': 'AI Agent 2025'}, 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'search_news', 'args': '{"query":"AI Agent 2025"}', 'id': 'call_xxsGXV6JZgYb7JAb0JU9EQ7o', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='call_xxsGXV6JZgYb7JAb0JU9EQ7o'), observation=[{'url': 'https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/', 'content': 'In a similar study, Deloitte forecasts that 25% of enterprises using GenAI will deploy AI Agents by 2025, growing to 50% by 2027. Meanwhile, Gartner predicts that by 2028, at least 15% of day-to-day work decisions will be made autonomously through agentic AI. It also states that by then, 33% of enterprise software applications will also include'}, {'url': 'https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents', 'content': 'Next year will be the year of AI agents | TechTarget This will make the AI agent more accurate in completing its task, Greene said. Other than the rise of single-task AI agents, 2025 may also be the year of building the infrastructure for AI agents, said Olivier Blanchard, an analyst with Futurum Group. "2025 isn\'t going to be the year when we see a fully developed agentic AI," he said. AI agents need an orchestration layer that works across different platforms and devices, Blanchard said. Because data is usually spread across different sources and processes, it might be challenging to give AI agents the data they need to perform the tasks they\'re being asked to do, Greene said.'}, {'url': 'https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks', 'content': 'According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents.'}])], 'messages': [FunctionMessage(content='[{"url": "https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/", "content": "In a similar study, Deloitte forecasts that 25% of enterprises using GenAI will deploy AI Agents by 2025, growing to 50% by 2027. Meanwhile, Gartner predicts that by 2028, at least 15% of day-to-day work decisions will be made autonomously through agentic AI. It also states that by then, 33% of enterprise software applications will also include"}, {"url": "https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents", "content": "Next year will be the year of AI agents | TechTarget This will make the AI agent more accurate in completing its task, Greene said. Other than the rise of single-task AI agents, 2025 may also be the year of building the infrastructure for AI agents, said Olivier Blanchard, an analyst with Futurum Group. \\"2025 isn\'t going to be the year when we see a fully developed agentic AI,\\" he said. AI agents need an orchestration layer that works across different platforms and devices, Blanchard said. Because data is usually spread across different sources and processes, it might be challenging to give AI agents the data they need to perform the tasks they\'re being asked to do, Greene said."}, {"url": "https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks", "content": "According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents."}]', additional_kwargs={}, response_metadata={}, name='search_news')]}
    ============================================================
    {'output': 'Here are some recent news articles discussing the future of AI agents in 2025:\n\n1. **AI Agent Trends**  \n   - **Source**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/)  \n   - **Summary**: A study by Deloitte forecasts that 25% of enterprises using Generative AI will deploy AI agents by 2025, with this number expected to grow to 50% by 2027. Gartner predicts that by 2028, at least 15% of daily work decisions will be made autonomously through agentic AI.\n\n2. **Next Year Will Be the Year of AI Agents**  \n   - **Source**: [TechTarget](https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents)  \n   - **Summary**: Analysts suggest that 2025 may focus on building the infrastructure necessary for AI agents. While single-task AI agents will rise, the development of a comprehensive orchestration layer across platforms will be crucial for their effectiveness.\n\n3. **Predictions for AI in 2025**  \n   - **Source**: [Stanford Institute for Human-Centered AI](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)  \n   - **Summary**: Experts predict a shift towards collaborative AI systems where multiple specialized agents work together, guided by humans. This includes the development of multimodal AI models in education and research, where AI agents collaborate on complex tasks.\n\nThese articles highlight the anticipated growth and evolution of AI agents, emphasizing their increasing integration into enterprise operations and collaborative environments.', 'messages': [AIMessage(content='Here are some recent news articles discussing the future of AI agents in 2025:\n\n1. **AI Agent Trends**  \n   - **Source**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/)  \n   - **Summary**: A study by Deloitte forecasts that 25% of enterprises using Generative AI will deploy AI agents by 2025, with this number expected to grow to 50% by 2027. Gartner predicts that by 2028, at least 15% of daily work decisions will be made autonomously through agentic AI.\n\n2. **Next Year Will Be the Year of AI Agents**  \n   - **Source**: [TechTarget](https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents)  \n   - **Summary**: Analysts suggest that 2025 may focus on building the infrastructure necessary for AI agents. While single-task AI agents will rise, the development of a comprehensive orchestration layer across platforms will be crucial for their effectiveness.\n\n3. **Predictions for AI in 2025**  \n   - **Source**: [Stanford Institute for Human-Centered AI](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)  \n   - **Summary**: Experts predict a shift towards collaborative AI systems where multiple specialized agents work together, guided by humans. This includes the development of multimodal AI models in education and research, where AI agents collaborate on complex tasks.\n\nThese articles highlight the anticipated growth and evolution of AI agents, emphasizing their increasing integration into enterprise operations and collaborative environments.', additional_kwargs={}, response_metadata={})]}
    ============================================================
</pre>

## Customizing intermediate steps output using user-defined functions

Define the following 3 functions to customize the intermediate steps output.

- `tool_callback`: Function to handle tool call output
- `observation_callback`: Function to handle observation (Observation) output
- `result_callback`: Function to handle final answer output

The following is a callback function used to output the intermediate steps of the Agent in a clean manner.

This callback function can be useful when outputting intermediate steps to users in Streamlit.

```python
from typing import Dict, Any


# Create AgentStreamParser class
class AgentStreamParser:
    def __init__(self):
        pass

    def tool_callback(self, tool: Dict[str, Any]) -> None:
        print("\n=== Tool Called ===")
        print(f"Tool: {tool.get('tool')}")
        print(f"Input: {tool.get('tool_input')}")
        print("==================\n")

    def observation_callback(self, step: Dict[str, Any]) -> None:
        print("\n=== Observation ===")
        observation_data = step["steps"][0].observation
        print(f"Observation: {observation_data}")
        print("===================\n")

    def result_callback(self, result: str) -> None:
        print("\n=== Final Answer ===")
        print(result)
        print("====================\n")

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        if "actions" in step:
            for action in step["actions"]:
                self.tool_callback(
                    {"tool": action.tool, "tool_input": action.tool_input}
                )
        elif "output" in step:
            self.result_callback(step["output"])
        else:
            self.observation_callback(step)


# Create AgentStreamParser instance
agent_stream_parser = AgentStreamParser()
```

Check the response process of Agent in streaming mode.

```python
# Run in streaming mode
result = agent_executor.stream({"input": "Generate a pie chart using matplotlib."})
# result = agent_executor.stream({"input": "Search news about AI Agent in 2025."})


for step in result:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">Python REPL can execute arbitrary code. Use with caution.
</pre>


    
![png](./img/output_23_1.png)
    


    
    === Tool Called ===
    Tool: python_repl_tool
    Input: {'code': "import matplotlib.pyplot as plt\n\n# Data to plot\nlabels = ['A', 'B', 'C', 'D']\nsizes = [15, 30, 45, 10]\ncolors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']\nexplode = (0.1, 0, 0, 0)  # explode 1st slice\n\n# Plot\nplt.figure(figsize=(8, 6))\nplt.pie(sizes, explode=explode, labels=labels, colors=colors,\n        autopct='%1.1f%%', shadow=True, startangle=140)\nplt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.\nplt.title('Pie Chart Example')\nplt.show()"}
    ==================
    
    
    === Observation ===
    Observation: 
    ===================
    
    
    === Final Answer ===
    The pie chart has been generated successfully. However, I cannot display the chart directly here. You can run the provided code in your local Python environment to see the pie chart. Here’s the code again for your convenience:
    
    ```python
    import matplotlib.pyplot as plt
    
    # Data to plot
    labels = ['A', 'B', 'C', 'D']
    sizes = [15, 30, 45, 10]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Pie Chart Example')
    plt.show()
    ```
    
    Feel free to modify the data and labels as needed!
    ====================
    
    

```python
# Run in streaming mode
result = agent_executor.stream({"input": "Search news about AI Agent in 2025."})


for step in result:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom">
    === Tool Called ===
    Tool: search_news
    Input: {'query': 'AI Agent 2025'}
    ==================
    
    
    === Observation ===
    Observation: [{'url': 'https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/', 'content': 'In a similar study, Deloitte forecasts that 25% of enterprises using GenAI will deploy AI Agents by 2025, growing to 50% by 2027. Meanwhile, Gartner predicts that by 2028, at least 15% of day-to-day work decisions will be made autonomously through agentic AI. It also states that by then, 33% of enterprise software applications will also include'}, {'url': 'https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents', 'content': 'Next year will be the year of AI agents | TechTarget This will make the AI agent more accurate in completing its task, Greene said. Other than the rise of single-task AI agents, 2025 may also be the year of building the infrastructure for AI agents, said Olivier Blanchard, an analyst with Futurum Group. "2025 isn\'t going to be the year when we see a fully developed agentic AI," he said. AI agents need an orchestration layer that works across different platforms and devices, Blanchard said. Because data is usually spread across different sources and processes, it might be challenging to give AI agents the data they need to perform the tasks they\'re being asked to do, Greene said.'}, {'url': 'https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks', 'content': 'According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents.'}]
    ===================
    
    
    === Final Answer ===
    Here are some recent news articles discussing the future of AI agents in 2025:
    
    1. **AI Agent Trends**  
       - **Source**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/)  
       - **Summary**: A study by Deloitte forecasts that 25% of enterprises using Generative AI will deploy AI agents by 2025, with this number expected to grow to 50% by 2027. Gartner predicts that by 2028, at least 15% of daily work decisions will be made autonomously through agentic AI.
    
    2. **Next Year Will Be the Year of AI Agents**  
       - **Source**: [TechTarget](https://www.techtarget.com/searchEnterpriseAI/feature/Next-year-will-be-the-year-of-AI-agents)  
       - **Summary**: Analysts suggest that 2025 may focus on building the infrastructure necessary for AI agents. While single-task AI agents will rise, the development of a comprehensive orchestration layer across platforms will be crucial for their effectiveness.
    
    3. **Predictions for AI in 2025**  
       - **Source**: [Stanford Institute for Human-Centered AI](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)  
       - **Summary**: Experts predict a shift towards collaborative AI systems where multiple specialized agents work together, guided by humans. This includes the development of multimodal AI models in education and research, where AI agents collaborate on complex tasks.
    
    These articles highlight the anticipated growth and evolution of AI agents, emphasizing collaboration, infrastructure development, and the integration of AI into enterprise decision-making processes.
    ====================
    
</pre>

Modify the callback function to use it.

```python
from typing import Dict, Any, Callable


# 1. Define AgentCallbacks class
class AgentCallbacks:
    def __init__(
        self,
        tool_callback: Callable,
        observation_callback: Callable,
        result_callback: Callable,
    ):
        self.tool_callback = tool_callback
        self.observation_callback = observation_callback
        self.result_callback = result_callback


# 2. Define AgentStreamParser class
class AgentStreamParser:
    def __init__(self, callbacks: AgentCallbacks):
        self.callbacks = callbacks

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        if "actions" in step:
            for action in step["actions"]:
                self.callbacks.tool_callback(
                    {"tool": action.tool, "tool_input": action.tool_input}
                )
        elif "output" in step:
            self.callbacks.result_callback(step["output"])
        else:
            self.callbacks.observation_callback(step)


# 3. Define callback functions
def tool_callback(tool) -> None:
    print("<<<<<<< Tool Called >>>>>>")
    print(f"Tool: {tool.get('tool')}")
    print(f"Input: {tool.get('tool_input')}")
    print("<<<<<<< Tool Called >>>>>>")


def observation_callback(step) -> None:
    print("<<<<<<< Observation >>>>>>")
    observation_data = step["steps"][0].observation
    print(f"Observation: {observation_data}")
    print("<<<<<<< Observation >>>>>>")


def result_callback(result: str) -> None:
    print("<<<<<<< Final Answer >>>>>>")
    print(result)
    print("<<<<<<< Final Answer >>>>>>")


# 4. Example usage
# Wrap callback functions into AgentCallbacks instance
agent_callbacks = AgentCallbacks(
    tool_callback=tool_callback,
    observation_callback=observation_callback,
    result_callback=result_callback,
)

# Create AgentStreamParser instance
agent_stream_parser = AgentStreamParser(agent_callbacks)
```

Check the output content. You can see that the output value of the intermediate content has been changed to the output value of the callback function I modified.

```python
# Request streaming output for the query
result = agent_executor.stream({"input": "Search news about AI Agent in 2025."})

for step in result:
    # Output intermediate steps using parser
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom"><<<<<<< Tool Called >>>>>>
    Tool: search_news
    Input: {'query': 'AI Agent 2025'}
    <<<<<<< Tool Called >>>>>>
    <<<<<<< Observation >>>>>>
    Observation: [{'url': 'https://www.analyticsvidhya.com/blog/2024/12/ai-agents-to-look-out-for/', 'content': "Q2. Why are these five AI agents considered game-changers for 2025? Ans. The selected agents—Oracle's Miracle Agent, Nvidia's Eureka Agent, Google's Project Jarvis, SAP's Joule Collaborative AI Agents, and Cisco's Webex AI Agent—stand out due to their innovative designs, broad applications, and industry impact."}, {'url': 'https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/', 'content': 'In a similar study, Deloitte forecasts that 25% of enterprises using GenAI will deploy AI Agents by 2025, growing to 50% by 2027. Meanwhile, Gartner predicts that by 2028, at least 15% of day-to-day work decisions will be made autonomously through agentic AI. It also states that by then, 33% of enterprise software applications will also include'}, {'url': 'https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks', 'content': 'According to leading experts from Stanford Institute for Human-Centered AI, one major trend is the rise of collaborative AI systems where multiple specialized agents work together, with humans providing high-level guidance. I expect to see more focus on multimodal AI models in education, including in processing speech and images. AI Agents Work Together In 2025, we will see a significant shift from relying on individual AI models to using systems where multiple AI agents of diverse expertise work together. As an example, we recently introduced the\xa0Virtual Lab, where a professor AI agent leads a team of AI scientist agents (e.g., AI chemist, AI biologist) to tackle challenging, open-ended research, with a human researcher providing high-level feedback. We will experience an emerging paradigm of research around how humans work together with AI agents.'}]
    <<<<<<< Observation >>>>>>
    <<<<<<< Final Answer >>>>>>
    Here are some recent news articles discussing AI Agents and their expected developments in 2025:
    
    1. **AI Agents to Look Out For**  
       - **Source**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/ai-agents-to-look-out-for/)  
       - **Summary**: This article highlights five AI agents considered game-changers for 2025, including Oracle's Miracle Agent, Nvidia's Eureka Agent, Google's Project Jarvis, SAP's Joule Collaborative AI Agents, and Cisco's Webex AI Agent. These agents are noted for their innovative designs and broad applications across various industries.
    
    2. **AI Agent Trends**  
       - **Source**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/ai-agent-trends/)  
       - **Summary**: A study by Deloitte forecasts that 25% of enterprises using Generative AI will deploy AI Agents by 2025, with this number expected to grow to 50% by 2027. Gartner predicts that by 2028, at least 15% of daily work decisions will be made autonomously through AI agents.
    
    3. **Predictions for AI in 2025**  
       - **Source**: [Stanford Institute for Human-Centered AI](https://hai.stanford.edu/news/predictions-ai-2025-collaborative-agents-ai-skepticism-and-new-risks)  
       - **Summary**: Experts predict a rise in collaborative AI systems where multiple specialized agents work together, guided by humans. The article discusses the shift from individual AI models to systems where diverse AI agents collaborate, exemplified by a Virtual Lab where an AI professor leads a team of AI scientist agents.
    
    These articles provide insights into the anticipated advancements and trends in AI agents by 2025.
    <<<<<<< Final Answer >>>>>>
</pre>

## Communicating Agent with previous conversation history

To remember previous conversation history, wrap `AgentExecutor` with `RunnableWithMessageHistory`.

For more details on `RunnableWithMessageHistory`, please refer to the link below.

**Reference**
- [LangChain Python API Reference > langchain: 0.3.14 > core > runnables > langchain_core.runnables.history > RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create a dictionary to store session_id
store = {}


# Function to get session history based on session_id
def get_session_history(session_ids):
    if session_ids not in store:  # If session_id is not in store
        # Create a new ChatMessageHistory object and store it in store
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # Return session history for the corresponding session_id


# Create an agent with chat message history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # Chat session_id
    get_session_history,
    # The key for the question input in the prompt: "input"
    input_messages_key="input",
    # The key for the message input in the prompt: "chat_history"
    history_messages_key="chat_history",
)
```

```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {"input": "Hello! My name is Teddy!"},
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom"><<<<<<< Final Answer >>>>>>
    Hello Teddy! How can I assist you today?
    <<<<<<< Final Answer >>>>>>
</pre>

```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {"input": "What is my name?"},
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom"><<<<<<< Final Answer >>>>>>
    Your name is Teddy!
    <<<<<<< Final Answer >>>>>>
</pre>

```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {
        "input": "My email address is teddy@teddynote.com. The company name is TeddyNote Co., Ltd."
    },
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom"><<<<<<< Final Answer >>>>>>
    Thank you for sharing that information, Teddy! How can I assist you with TeddyNote Co., Ltd. or anything else today?
    <<<<<<< Final Answer >>>>>>
</pre>

```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {
        "input": "Search the latest news and write it as the body of the email. "
        "The recipient is `Ms. Sally` and the sender is my personal information."
        "Write in a polite tone, and include appropriate greetings and closings at the beginning and end of the email."
    },
    # Set session_id
    config={"configurable": {"session_id": "abc123"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom"><<<<<<< Tool Called >>>>>>
    Tool: search_news
    Input: {'query': 'latest news'}
    <<<<<<< Tool Called >>>>>>
    <<<<<<< Observation >>>>>>
    Observation: [{'url': 'https://apnews.com/', 'content': '[![Image 3: Image](https://dims.apnews.com/dims4/default/dbbfca1/2147483647/strip/true/crop/2949x1964+0+368/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fe8%2F0a%2Fc377af5f961e0128e91ba2d36dc4%2F4f91966abf024d799157d474fdbd0aea)](https://apnews.com/article/trump-trolling-canada-jill-biden-trudeau-39ecae6554c7b4350e6a106354672eb4) [![Image 4: Image](https://dims.apnews.com/dims4/default/78686e1/2147483647/strip/true/crop/6000x3997+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F55%2F26%2F09fa12ce1edbded24a1bdefc4a51%2Fa2601074f03248cdab2d041f3a626cbc)](https://apnews.com/article/syria-israel-airstrike-assad-war-b90edb8dbe8268dacf90e59ca601e2e3) [![Image 5: Image](https://dims.apnews.com/dims4/default/e588d90/2147483647/strip/true/crop/4032x2686+0+169/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Ff9%2F96%2F15aa6e41881b7ae9693c3f7edb87%2Fa95369e50d864a3dbca1a33f28f6dbe1)](https://apnews.com/article/migrant-health-care-dreamers-lawsuit-504c8f05b3d93538cb7292318c267441) [![Image 6: Image](https://dims.apnews.com/dims4/default/f49d8f1/2147483647/strip/true/crop/5301x3531+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F4b%2Fd0%2Fd9c6f22eb322668f8259364deee8%2Fbffc615f4ead41809a2b3a78d29bd8eb)](https://apnews.com/article/monarch-butterflies-endangered-species-climate-habitat-f5d4844289ede7b3d76918cc6f98a5cc) [![Image 12: Image](https://dims.apnews.com/dims4/default/dec17c1/2147483647/strip/true/crop/3000x1998+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F94%2F6d%2F5f84a76d9f279a512cd56e998989%2F0ec3c08087484de1816a01e9c9c2ffb0)](https://apnews.com/article/wildfire-malibu-evacuation-pepperdine-university-ef9f6ea11815be64feaf2e5e5eb7588c) [![Image 13: Image](https://dims.apnews.com/dims4/default/b5cf736/2147483647/strip/true/crop/8640x5755+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F06%2Fc3%2Fd6f2099324fea0bd0493f0c92c37%2F4178cf4001c4494898732f464b5bbf27)](https://apnews.com/article/biden-trump-economy-stupid-jobs-inflation-cde78b58c6b4ccbcb7ce237f741e06a6) [![Image 14: Image](https://dims.apnews.com/dims4/default/259b976/2147483647/strip/true/crop/3000x2000+0+0/resize/567x378!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fa2%2Ff4%2F0ceaf87a1d3c3410b4cba722aaef%2Fe3ff477aa750430e8fea9fe137e61dc3)](https://apnews.com/article/christopher-nolan-interstellar-rerelease-interview-bd7f4de84525062fb0d0e89a7fe6ea92) [![Image 15: Image](https://dims.apnews.com/dims4/default/61ffd0e/2147483647/strip/true/crop/4740x3157+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F0b%2Fc7%2Fcd3aa4ad63e7850000a45462a828%2F2d899e6be47b450587a7f40fad9e2e1c)](https://apnews.com/article/south-africa-car-road-crash-accident-2f063819764ae42cd4c8ffd5c76e4c6d) [![Image 18: Image](https://dims.apnews.com/dims4/default/9fdd823/2147483647/strip/true/crop/2300x1532+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fc0%2F26%2F15202742a43c06c94c553c2008d9%2F9a08766502ac4a7ca356980292b406ba)](https://apnews.com/article/scrim-new-orleans-fugitive-dog-6fdddcb2cb694981f50f4c25ecfab655) [![Image 20: Image](https://dims.apnews.com/dims4/default/24d5169/2147483647/strip/true/crop/4667x3109+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fd0%2Ff2%2F38f75b20e51bb3f3840e1386eb93%2F568053ca409f4c4db5a00c4868fbf7e5)](https://apnews.com/associated-press-100-photos-of-2024-an-epic-catalog-of-humanity) [![Image 21: Image](https://dims.apnews.com/dims4/default/0f779f2/2147483647/strip/true/crop/5819x3876+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Ffb%2F33%2F4cf11414c7dc0669cad5790c54c9%2Fccf3d40bbfe64a268dae81b2f9b31731)](https://apnews.com/world-news/general-news-1637ffe66255f7048734759ec6bd99e9) [![Image 22: Image](https://dims.apnews.com/dims4/default/2233dae/2147483647/strip/true/crop/5561x3707+0+2/resize/567x378!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F9e%2F1e%2Ff8ed9ac2b8080a2bcbde719e6f26%2F43efeab456df437dab2aae52577dfc83)](https://apnews.com/article/france-paris-notre-dame-carpenter-fire-catholic-ea516112b8393a795cd97b6abe5ef4e3) [![Image 23: Image](https://dims.apnews.com/dims4/default/6a5602a/2147483647/strip/true/crop/3000x1998+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fb3%2Fef%2Ffadb8b20993381c75377bcfb2279%2F7ac65958f4e6456d87fabb1c5d3166dc)](https://apnews.com/article/buddhism-interfaith-new-jersey-spirituality-ecce3b0d5569e0497c47c92eb57f9abc) [![Image 25: Image](https://dims.apnews.com/dims4/default/103e6fa/2147483647/strip/true/crop/4381x2918+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F4f%2F1f%2Fc8879ab3242b4d8d39cdb6884eb0%2Fa550497041f84c2c80354e985404803b)](https://apnews.com/world-news/taylor-swift-general-news-domestic-news-add8d87cc892eac2838c51812ea032de) [![Image 26: Image](https://dims.apnews.com/dims4/default/db4c05c/2147483647/strip/true/crop/5500x3664+0+52/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fb1%2F7b%2F62f6a74c6e33b81ed1f34dea89d6%2Fb0be2be6db014cfcbd9babff7f661d92)](https://apnews.com/article/notre-dame-cathedral-reopening-fire-paris-28cf9686d7d48dce0c046ca9e46f3e99) [![Image 69: Image](https://dims.apnews.com/dims4/default/dd0f3ed/2147483647/strip/true/crop/3072x2046+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F72%2Fe0%2F21056d49f768745cf7c7c2a4a517%2Fa3d9ef07c1c9475c8834ac5ec8a55297)](https://apnews.com/article/romania-ukraine-draft-cat-rescue-mountain-war-8e57bc8b07a2ec80ea16ca9cd09450b3) [![Image 78: Image](https://dims.apnews.com/dims4/default/6a231c6/2147483647/strip/true/crop/5616x3741+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F2e%2F97%2Fa1b23b3f7b859b296f2fd76e90a5%2F464a3abd0577440794bdcb70e41f9274)](https://apnews.com/article/trump-paris-notre-dame-f97fde62ca2ce68c3874c395b305e26b) [![Image 80: Image](https://dims.apnews.com/dims4/default/83b446e/2147483647/strip/true/crop/6000x3997+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F8e%2F09%2F243684f1b3e6b5cde4701ceda6ff%2F65c76be267b04d0395f7ed1c35849a14)](https://apnews.com/article/syria-assad-ousted-rebel-offensive-timeline-8c54a8b97803d4b10cde53b97227128e) [![Image 81: Image](https://dims.apnews.com/dims4/default/8017fc1/2147483647/strip/true/crop/7185x4786+0+86/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Ff1%2F7e%2Fe1e5508484de218a863d8ecd9fad%2Fb34a7af614a24239b810eb78f2e478cf)](https://apnews.com/article/trump-ukraine-military-aid-russia-biden-efbcff8ed068e621055e8fa70b5905e1) [![Image 82: Image](https://dims.apnews.com/dims4/default/b1034b5/2147483647/strip/true/crop/8640x5755+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fd6%2Fdd%2F0d1721473c6d61befedc97c7a426%2F2cdc242c49684c10a61fe1d24cd69d87)](https://apnews.com/article/russia-oreshnik-hypersonic-missile-putin-ukraine-war-345588a399158b9eb0b56990b8149bd9) [![Image 83: Image](https://dims.apnews.com/dims4/default/90facc9/2147483647/strip/true/crop/5081x3385+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F10%2F1f%2Fe69ad0b2a8e0010f4861f097d29e%2F5ad8c5d8d16d40d7a191dc549183afeb)](https://apnews.com/article/russia-ukraine-war-zelenskyy-troops-nato-9bf883670879e2a398fe1a7ebfb71ddb) [![Image 84: Image](https://dims.apnews.com/dims4/default/05046b2/2147483647/strip/true/crop/2045x1362+0+44/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Faf%2Fcc%2F1e4d32c41e0865d25a1eecbad0ef%2F418d96657c1f4cfba849d77a7361730c)](https://apnews.com/article/russia-putin-syria-assad-ukraine-war-31fa9b933372b3704ed285c96863892b) [![Image 85: Image](https://dims.apnews.com/dims4/default/3ea7e0c/2147483647/strip/true/crop/6000x3997+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F00%2Fd8%2Fb87667ac757b9ca481770ab152e6%2Fb6057e099a8241db9ae4ab336a5dc2c2)](https://apnews.com/article/tulsi-gabbard-hegseth-kash-patel-trump-syria-44d0150e7d251946b60fc7f6799c4a74) [![Image 88: Image](https://dims.apnews.com/dims4/default/41f81a5/2147483647/strip/true/crop/1919x1278+1+0/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F1c%2F12%2F6070e8327115344c7edba999d799%2Fda417d733523484caa3f22d29858c0ec)](https://apnews.com/article/secret-santa-stress-holiday-gift-guide-d5a907bd8aa1b0b1ebf1e04bba9fe89c) [![Image 89: Image](https://dims.apnews.com/dims4/default/3c8b8e5/2147483647/strip/true/crop/5184x3453+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Faf%2Fc1%2F5ef68e3d35cc53e50da760ec39eb%2F44fd1bff50d44b6d891d67e9921dd91d)](https://apnews.com/article/november-project-exercise-social-fitness-winter-fc55e4aefbd003f999aafec15fe25a0e) [![Image 90: Image](https://dims.apnews.com/dims4/default/8c75a98/2147483647/strip/true/crop/8184x5451+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fd1%2F43%2Fa03aa410da12d72c007f4e04dbe1%2Fe02494da5ee04e49bea36e5877aed408)](https://apnews.com/article/personal-trainer-exercise-workout-c94f4f6625d2d6a5a77537946d1518b5) [![Image 91: Image](https://dims.apnews.com/dims4/default/3fc5125/2147483647/strip/true/crop/4479x2984+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F62%2Fcf%2F564ba47875ce1cfd0bbb3b9fe958%2F1fe0aa9456ed4952a6ef4843535f8bde)](https://apnews.com/article/travel-stress-holiday-tips-db5e5d819b6251bd415a9ad204baa03a) [![Image 92: Image](https://dims.apnews.com/dims4/default/1318d6e/2147483647/strip/true/crop/1620x1079+0+0/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F0c%2F15%2F420caf3cf8f2efbc44b839ea596d%2Fee43d974cece47e09f8f324cd708f6d1)](https://apnews.com/article/work-life-stress-reduction-breathing-techniques-8c0636a09d605ef0c56e529e8be0f2f9) [![Image 97: Image](https://dims.apnews.com/dims4/default/74e65c3/2147483647/strip/true/crop/5993x3992+3+0/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F33%2F73%2F083534678639c949767e0353e020%2F1e147b6dd9aa48648288cc20279edc52)](https://apnews.com/article/ap-top-asia-photos-of-2024-e071bc96deed1052511a53e0e1cc61f9) [![Image 98: Image](https://dims.apnews.com/dims4/default/d1a96d7/2147483647/strip/true/crop/2829x1884+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Ff2%2F0e%2F6a3787430fbc823f8900f7e6ab52%2Ff8bf616f86284cad9e448061cfc929c9)](https://apnews.com/article/argentina-milei-trump-musk-default-economy-inflation-libertarian-18efe55d81df459792a038ea9e321800) [![Image 99: Image](https://dims.apnews.com/dims4/default/0f779f2/2147483647/strip/true/crop/5819x3876+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Ffb%2F33%2F4cf11414c7dc0669cad5790c54c9%2Fccf3d40bbfe64a268dae81b2f9b31731)](https://apnews.com/article/ap-sports-photos-of-year-d07f4a6c90716b4e5e46df43cea695e8) [![Image 101: Image](https://dims.apnews.com/dims4/default/df0270d/2147483647/strip/true/crop/6000x4000+0+0/resize/567x378!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fc3%2Fe4%2F7da2e99ba0a57a0bff843a3c04f8%2Fa5584bf0667343f78c019f2f35b56868)](https://apnews.com/article/germany-syria-refugees-asylum-future-ab24be8f2ffef4e118ed12a4b0df110f) [![Image 102: Image](https://dims.apnews.com/dims4/default/6fe020a/2147483647/strip/true/crop/5025x3347+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F50%2F3d%2Ffd28cebf15c3665f091349a5f7e0%2F1650cbfb41ee4e2194210bb4e8bf8ba0)](https://apnews.com/article/israel-netanyahu-corruption-trial-gaza-478c957c7749986d5b8b2d039f670d54) [![Image 103: Image](https://dims.apnews.com/dims4/default/26a7204/2147483647/strip/true/crop/3900x2598+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F8a%2F3c%2Fa6ff0ab1b0276a935bf063ee888d%2Fa05f1c5fd3514f569aee18054cc5a6e9)](https://apnews.com/article/ukraine-war-poland-tusk-negotiations-russia-eu-23802e5b26f5c0c1d45a32e9cc80f362) [![Image 107: Image](https://dims.apnews.com/dims4/default/c8ba91c/2147483647/strip/true/crop/4032x2688+0+0/resize/567x378!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F30%2Fc2%2F821b7e6e01f117fea2b2b1e4be05%2Fa4e7f61cd4f846488e769fc42692a476)](https://apnews.com/article/congo-world-heritage-site-gold-mining-china-5e9499fd939c3c2d798a6165f3fc487b) [![Image 110: Image](https://dims.apnews.com/dims4/default/0ceb174/2147483647/strip/true/crop/4796x3195+0+1/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fac%2F7f%2F5466595a879ca2d404379f4b8310%2Fad89b32dd88b4dcd8f005b757f1c624e)](https://apnews.com/article/climate-change-november-hottest-year-record-d1f41a2c7341c006f051ad321915d89c) [![Image 111: Image](https://dims.apnews.com/dims4/default/041f239/2147483647/strip/true/crop/6871x4577+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F21%2F3e%2F10d5b4943d3c79f80cf6c410519c%2F715b29407d0a4dbca2e3b035f924e1ad)](https://apnews.com/article/deadly-heat-humidity-mexico-climate-change-2e8db903deabd015f608e45204a19bf0) [![Image 113: Image](https://dims.apnews.com/dims4/default/44c34bd/2147483647/strip/true/crop/3368x2243+0+139/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F49%2F08%2F669217d7f9d734a85f4dadd01274%2F3f91221b4d87434789bbf4e34bc32ac3)](https://apnews.com/article/trump-energy-permitting-reform-drilling-billion-investment-dd99706a325082cdf475e599ec3c0687) [![Image 114: Image](https://dims.apnews.com/dims4/default/bcd5cd4/2147483647/strip/true/crop/5572x3712+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F90%2F3a%2F54e9a82c765fa74d587bf2041f6a%2Fbd09b6cd0ee84c57bfe484c73799a8cb)](https://apnews.com/article/supreme-court-environment-regulation-recuse-gorsuch-utah-16cff708f7549ff430823213eeaa22f6) [![Image 118: Image](https://dims.apnews.com/dims4/default/6bb9c84/2147483647/strip/true/crop/5616x3741+0+2/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fbd%2Fd5%2Fa6a842258afe7f54ba96306c1aa5%2Fcec851964be444f89a2d8ff1424493c0)](https://apnews.com/article/fact-check-vermont-supreme-court-vaccination-ruling-0256fc7ad888230ee8dc51f6fe0d479a)'}, {'url': 'https://www.cnn.com/', 'content': 'View the latest news and breaking news today for U.S., world, weather, entertainment, politics and health at CNN.com.'}, {'url': 'https://www.foxnews.com/', 'content': "[![Image 12: Trump makes vow to anyone who invests $1 billion or more in the US - Fox News](https://static.foxnews.com/static/orion/img/clear-16x9.gif)](https://www.foxbusiness.com/politics/trump-makes-vow-anyone-who-invests-1-billion-more-us) [![Image 19: Democratic governors won't address age limit for future presidential nominee - Fox News](https://static.foxnews.com/static/orion/img/clear-16x9.gif)](https://www.foxnews.com/media/democratic-governors-refuse-say-how-old-too-old-2028-nominee) [![Image 28: Meet Trump's 'valuable resource' who lawmakers say is 'critical' to his operation - Fox News](https://static.foxnews.com/static/orion/img/clear-16x9.gif)](https://www.foxnews.com/politics/meet-natalie-harp-trumps-valuable-resource-who-lawmakers-say-critical-his-operation) [![Image 32: President-elect Trump shooting task force unveils bombshell final report - Fox News](https://static.foxnews.com/static/orion/img/clear-16x9.gif)](https://www.foxnews.com/politics/trump-shooting-task-force-says-dhs-secret-service-havent-produced-docs-golf-course-incident) [![Image 38: Will Cain discusses CNN comparing Daniel Penny to CEO murder suspect - Fox News](https://static.foxnews.com/static/orion/img/clear-16x9.gif)Video](https://www.foxnews.com/video/6365813443112) [![Image 64: Many want Trump to address the border crisis in his first 100 days in office - Fox News](https://static.foxnews.com/static/orion/img/clear-16x9.gif)](https://www.foxnews.com/media/border-crisis-top-issue-many-want-trump-address-his-first-100-days-office) [![Image 68: Biden says Trump inheriting ‘strongest economy in modern history,’ slams tariff plan as ‘major mistake’ - Fox News](https://static.foxnews.com/static/orion/img/clear-16x9.gif)](https://www.foxnews.com/politics/biden-says-trump-inheriting-strongest-economy-modern-history-slams-tariff-plan-major-mistake)"}]
    <<<<<<< Observation >>>>>>
    <<<<<<< Final Answer >>>>>>
    Here's a draft for your email to Ms. Sally, including the latest news:
    
    ---
    
    **Subject:** Latest News Update
    
    Dear Ms. Sally,
    
    I hope this message finds you well.
    
    I wanted to share some of the latest news highlights that may interest you:
    
    1. **Trump's Investment Vow**: Former President Trump has made a significant promise to anyone who invests $1 billion or more in the United States. [Read more here.](https://www.foxbusiness.com/politics/trump-makes-vow-anyone-who-invests-1-billion-more-us)
    
    2. **Democratic Governors on Age Limits**: Democratic governors have chosen not to address the age limit for future presidential nominees, raising questions about the party's direction. [Read more here.](https://www.foxnews.com/media/democratic-governors-refuse-say-how-old-too-old-2028-nominee)
    
    3. **Meet Trump's 'Valuable Resource'**: A profile on Natalie Harp, who is considered a critical resource for Trump's operations, has been released. [Read more here.](https://www.foxnews.com/politics/meet-natalie-harp-trumps-valuable-resource-who-lawmakers-say-critical-his-operation)
    
    4. **Biden's Comments on the Economy**: President Biden has stated that Trump is inheriting the "strongest economy in modern history," while criticizing his tariff plan as a major mistake. [Read more here.](https://www.foxnews.com/politics/biden-says-trump-inheriting-strongest-economy-modern-history-slams-tariff-plan-major-mistake)
    
    For more updates, you can visit [CNN](https://www.cnn.com/) or [AP News](https://apnews.com/).
    
    Thank you for your attention, and I look forward to hearing your thoughts.
    
    Best regards,
    
    Teddy  
    teddy@teddynote.com  
    TeddyNote Co., Ltd.
    
    --- 
    
    Feel free to modify any part of the email as you see fit!
    <<<<<<< Final Answer >>>>>>
</pre>

```python
# Request streaming output for the query
response = agent_with_chat_history.stream(
    {"input": "What is my name?"},
    # Set session_id
    config={"configurable": {"session_id": "def456"}},
)

# Check the output
for step in response:
    agent_stream_parser.process_agent_steps(step)
```

<pre class="custom"><<<<<<< Final Answer >>>>>>
    I don't have access to your personal information, so I don't know your name. If you'd like to share it, feel free!
    <<<<<<< Final Answer >>>>>>
</pre>
