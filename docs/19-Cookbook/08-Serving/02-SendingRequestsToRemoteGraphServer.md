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

# Sending Requests to Remote Graph Server

- Author: [Yoonji Oh](https://github.com/samdaseuss)
- Design:
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview
In this tutorial, we will learn how to launch an application server and send Python requests to Remote Graph.

In this process, we will:
1. Understand the differences between LangServe and LangGraph
2. Learn why LangGraph is recommended
3. Get hands-on experience through the Chat LangChain application

We will examine each step in detail, from environment setup to server launch and sending actual requests. Through this tutorial, you will be able to build a foundation for AI application development using LangGraph.

Let's get started!

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is LangServe and LangGraph](#what-is-langserve-and-langgraph)
- [LangGraph is now recommended over LangServe.](#langgraph-is-now-recommended-over-langserve)
- [Practice with Chat LangChain Application](#practice-with-chat-langchain-application)
- [Thread-level persistence](#thread-level-persistence)
- [Using as a Subgraph](#using-as-a-subgraph)
- [Summary](#summary)

### References
- [LangServe](https://python.langchain.com/docs/langserve/)
- [LangGraph](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/#overview)
- [LangChain: Query Construction](https://blog.langchain.dev/query-construction/)
- [How to use remote graph](https://langchain-ai.github.io/langgraph/how-tos/use-remote-graph/)

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
        "langchain-anthropic",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
        "langgraph",
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
        "LANGCHAIN_PROJECT": "Sending Requests to Remote Graph Server",  # Please set it the same as title
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

**[Note]** This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## What is LangServe and LangGraph
Before proceeding with this tutorial, there are concepts you need to understand. These are LangServe and LangGraph. Let's make sure to clarify the difference between these two.

### LangServe
LangServe helps developers deploy LangChain runnables and chains as a REST API. Through the built-in Runnable object, you can easily create data pipelines from various components. These pipelines are ultimately provided through APIs called invoke, batch, and stream.
This library is integrated with FastAPI and uses pydantic for data validation.
In addition, it provides a client that can be used to call into runnables deployed on a server.


### LangGraph
LangGraph Platform is a commercial solution for deploying agentic applications to production, built on the open-source LangGraph framework.
- **LangGraph Server** : The server defines an opinionated API and architecture that incorporates best practices for deploying agentic applications, allowing you to focus on building your agent logic rather than developing server infrastructure.
- **LangGraph Studio** : LangGraph Studio is a specialized IDE that can connect to a LangGraph Server to enable visualization, interaction, and debugging of the application locally.
- **LangGraph CLI** : LangGraph CLI is a command-line interface that helps to interact with a local LangGraph
- **Python/JS SDK** : The Python/JS SDK provides a programmatic way to interact with deployed LangGraph Applications.
- **Remote Graph** : A RemoteGraph allows you to interact with any deployed LangGraph application as though it were running locally.

## LangGraph is now recommended over LangServe.

langchain-ai recommend using LangGraph Platform rather than LangServe for new projects.
Langchain-ai will continue to accept bug fixes for LangServe from the community; however, Langchain-ai will not be accepting new feature contributions.

In contrast to LangServe, LangGraph Platform provides comprehensive, out-of-the-box support for persistence, memory, double-texting handling, human-in-the-loop workflows, cron job scheduling, webhooks, high-load management, advanced streaming, support for long-running tasks, background task processing, and much more.

### Why use LangGraph?
LangGraph has been designed so that developers can concentrate exclusively on developing AI agent features.
Here are the key features and advantages of the LangGraph Platform:

1. Real-time Processing Features
- Streaming Support: Ability to monitor complex task progress in real-time
- Double Text Processing: Reliably manage rapid consecutive user messages
- Burst Handling: Stable processing through queue system even with multiple simultaneous requests

2. Long-running Task Management
- Background Execution: Reliably handle tasks that take hours to complete
- Long-run Support: Prevent unexpected connection drops through heartbeat signals
- Status Monitoring: Track execution status through polling and webhooks

3. Data Management
- Checkpoint Functionality: Save and recover task states
- Memory Management: Maintain data like conversation history across sessions
- Ready-to-use Storage Solution: Available without custom configuration

4. User Intervention Support
- Human-in-the-loop: Allow user intervention in processes when needed
- Dedicated Endpoints: Special access points for manual supervision

These features allow developers to focus on agent development rather than infrastructure building.


## Practice with Chat LangChain Application
To simply test server communication, please download and run the code locally through the Chat LangChain tutorial provided by LangChain. We will try using a simple chatbot that uses graphs.
Please visit the project repository to prepare the code.
- [Chat-Langchain](https://github.com/langchain-ai/chat-langchain/tree/langserve)

There are essential environment variables that must be set before running the code.
* `OPENAI_API_KEY`: your_secret_key_here
* `LANGCHAIN_TRACING_V2`: "true"
* `LANGCHAIN_PROJECT`: langserve-launch-example
* `LANGCHAIN_API_KEY`: your_secret_key_here
* `FIREWORKS_API_KEY`: your_secret_here
* `WEAVIATE_API_KEY`: your_secret_key_here
* `WEAVIATE_URL`: https://your-weaviate-instance.com(or https://weaviate.io/developers/weaviate/connections/connect-cloud)
* `WEAVIATE_INDEX_NAME`: your_index_name
* `RECORD_MANAGER_DB_URL`: your_db_url (e.g.  postgresql://postgres:[YOUR_DB_PASSWORD]@db.daxpgrzsg.supabase.co:5432/postgres)


Run the following command to start the server locally.

```
pip install "langgraph-cli[inmem]"
```

```
langgraph dev
```

The server will be launched this way.


<div style="text-align: center;">
    <img src="./assets/02-sending-requests-to-remote-graph-server-01.png" alt="Image Description" width="500">
</div>

- API: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API Docs: http://127.0.0.1:2024/docs

API (http://127.0.0.1:2024) is the main API endpoint, serving as the base address for direct communication with the server. 

Studio UI (https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024) is LangChain's web-based development environment that provides an interface for visually inspecting graphs and debugging. 

<div style="text-align: center;">
    <img src="./assets/02-sending-requests-to-remote-graph-server-02.png" alt="Image Description" width="500">
</div>

<div style="text-align: center;">
    <img src="./assets/02-sending-requests-to-remote-graph-server-04.png" alt="Image Description" width="500">
</div>

API Docs (http://127.0.0.1:2024/docs) is an API documentation page that contains Swagger documentation where you can find all available endpoints and their usage instructions.

<div style="text-align: center;">
    <img src="./assets/02-sending-requests-to-remote-graph-server-03.png" alt="Image Description" width="500">
</div>

```python
from langgraph.pregel.remote import RemoteGraph

url = "http://127.0.0.1:2024/"
graph_name = "chat"
remote_graph = RemoteGraph(graph_name, url=url)
```

After running this code, let's check the server logs.
<div style="text-align: center;">
    <img src="./assets/02-sending-requests-to-remote-graph-server-05.png" alt="Image Description" width="500">
</div>

```python
# invoke the graph
result = await remote_graph.ainvoke({
    "messages": [{"role": "user", "content": "hi!"}]
}, config={
    "configurable": {
        "embedding_model": "openai/text-embedding-3-small",
        "query_model": "openai/gpt-4-turbo-preview",
        "response_model": "openai/gpt-4-turbo-preview",
        "router_system_prompt": "You are a helpful assistant...",
        "research_plan_system_prompt": "You are a research planner...",
        "response_system_prompt": "You are an expert..."
    }
})

result
```




<pre class="custom">{'messages': [{'content': 'hi!',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'f169f605-c6a2-4609-86d0-33908882c0f7',
       'example': False},
      {'content': 'Hello! How can I assist you today?',
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 10,
         'prompt_tokens': 18,
         'total_tokens': 28,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-ee75b556-111f-4d6f-b7cf-19db660dc8e8-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 18,
        'output_tokens': 10,
        'total_tokens': 28,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}}],
     'steps': [],
     'documents': [],
     'answer': 'Hello! How can I assist you today?',
     'query': 'hi!'}</pre>



```python
# stream outputs from the graph
async for chunk in remote_graph.astream({
    "messages": [{"role": "user", "content": "what's the weather in la"}]
}, config={
    "configurable": {
        "embedding_model": "openai/text-embedding-3-small",
        "query_model": "openai/gpt-4-turbo-preview",
        "response_model": "openai/gpt-4-turbo-preview",
        "router_system_prompt": "You are a helpful assistant that directs users to the right information.",
        "research_plan_system_prompt": "You are a research planner. Create a step by step plan.",
        "response_system_prompt": "You are a helpful assistant. Answer based on the context provided: {context}",
        "more_info_system_prompt": "You need more information to answer. {logic}",
        "general_system_prompt": "You are a helpful assistant. {logic}"
    }
}):
    print(chunk)
```

<pre class="custom">{'create_research_plan': {'steps': ['Identify the specific location in LA (Los Angeles) you are interested in.', 'Determine the current date and time to ensure the weather forecast is up to date.', 'Visit a reliable weather forecasting website or use a weather app such as The Weather Channel, AccuWeather, or the National Weather Service.', 'Enter the specific location into the search bar of the chosen website or app.', 'Review the current weather conditions displayed, including temperature, humidity, precipitation, and wind speed.', 'Check the hourly and daily forecasts to get an idea of how the weather might change throughout the day or week.', 'For a more detailed understanding, look at the radar and satellite images available on the website or app.', 'If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.'], 'documents': 'delete', 'query': "what's the weather in la"}}
    {'conduct_research': {'documents': [], 'steps': ['Determine the current date and time to ensure the weather forecast is up to date.', 'Visit a reliable weather forecasting website or use a weather app such as The Weather Channel, AccuWeather, or the National Weather Service.', 'Enter the specific location into the search bar of the chosen website or app.', 'Review the current weather conditions displayed, including temperature, humidity, precipitation, and wind speed.', 'Check the hourly and daily forecasts to get an idea of how the weather might change throughout the day or week.', 'For a more detailed understanding, look at the radar and satellite images available on the website or app.', 'If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['Visit a reliable weather forecasting website or use a weather app such as The Weather Channel, AccuWeather, or the National Weather Service.', 'Enter the specific location into the search bar of the chosen website or app.', 'Review the current weather conditions displayed, including temperature, humidity, precipitation, and wind speed.', 'Check the hourly and daily forecasts to get an idea of how the weather might change throughout the day or week.', 'For a more detailed understanding, look at the radar and satellite images available on the website or app.', 'If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['Enter the specific location into the search bar of the chosen website or app.', 'Review the current weather conditions displayed, including temperature, humidity, precipitation, and wind speed.', 'Check the hourly and daily forecasts to get an idea of how the weather might change throughout the day or week.', 'For a more detailed understanding, look at the radar and satellite images available on the website or app.', 'If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['Review the current weather conditions displayed, including temperature, humidity, precipitation, and wind speed.', 'Check the hourly and daily forecasts to get an idea of how the weather might change throughout the day or week.', 'For a more detailed understanding, look at the radar and satellite images available on the website or app.', 'If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['Check the hourly and daily forecasts to get an idea of how the weather might change throughout the day or week.', 'For a more detailed understanding, look at the radar and satellite images available on the website or app.', 'If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['For a more detailed understanding, look at the radar and satellite images available on the website or app.', 'If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['If planning outdoor activities, pay attention to any weather alerts or warnings that might affect your plans.', 'Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['Consider checking multiple sources to compare forecasts and get the most accurate information.', 'Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': ['Keep monitoring the weather if you have upcoming plans in LA, as forecasts can change.']}}
    {'conduct_research': {'documents': [], 'steps': []}}
    {'respond': {'messages': [{'content': "I can't provide real-time weather updates or forecasts. For the most current weather conditions in Los Angeles, please check a reliable weather website, app, or news outlet.", 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 35, 'prompt_tokens': 35, 'total_tokens': 70, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run-74afd936-f55e-482e-bb0a-155047635ed8-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 35, 'output_tokens': 35, 'total_tokens': 70, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}], 'answer': "I can't provide real-time weather updates or forecasts. For the most current weather conditions in Los Angeles, please check a reliable weather website, app, or news outlet."}}
</pre>

## Thread-level persistence
Thread-level persistence is a method of maintaining the "memory" of conversations or tasks. It allows a program to "remember" the content of previous conversations or operations.
It's similar to writing down important information in a notebook and being able to refer back to it later.
Simple graph executions are stateless. Being stateless means that checkpoints and the final state of the graph are not saved.

```python
from langgraph_sdk import get_sync_client

sync_client = get_sync_client(url=url)
```

If you want to persist the outputs of graph execution (for example, to enable human-in-the-loop features), you can create a thread and provide the thread ID via the config argument, just as you would with a compiled graph.

```python
# create a thread (or use an existing thread instead)
thread = sync_client.threads.create()
config_thread =  {
    "configurable": {
        "thread_id": thread["thread_id"],
        "embedding_model": "openai/text-embedding-3-small",
        "query_model": "openai/gpt-4-turbo-preview",
        "response_model": "openai/gpt-4-turbo-preview",
        "router_system_prompt": "You are a helpful assistant...",
        "research_plan_system_prompt": "You are a research planner...",
        "response_system_prompt": "You are an expert..."
}}
```

After setting the thread ID, we will proceed to ask about the weather in San Francisco.

```python
# invoke the graph with the thread configs
result = await remote_graph.ainvoke({
   "messages": [{"role": "user", "content": "what's the weather in sf?"}]
}, config_thread)

result
```




<pre class="custom">{'messages': [{'content': "what's the weather in sf?",
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'eee694bf-4a7f-48f0-95f1-ef9e6ca97adb',
       'example': False},
      {'content': "I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco or any other location, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or AccuWeather. These sources can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details.",
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 78,
         'prompt_tokens': 23,
         'total_tokens': 101,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-eb507a89-ec6e-4093-b90c-21ef2de63788-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 23,
        'output_tokens': 78,
        'total_tokens': 101,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}}],
     'steps': [],
     'documents': [],
     'answer': "I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco or any other location, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or AccuWeather. These sources can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details.",
     'query': "what's the weather in sf?"}</pre>



Let's run the two code examples below that include the following questions:

| Question | Expected Answer |
|---|---|
| Did I ask about the weather in SF earlier? | YES |
| Did I ask about the weather in LA earlier? | NO |

```python
result = await remote_graph.ainvoke({
   "messages": [{"role": "user", "content": "Did I ask about the weather in SF earlier?"}]
}, config_thread)

result
```




<pre class="custom">{'messages': [{'content': "what's the weather in sf?",
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'eee694bf-4a7f-48f0-95f1-ef9e6ca97adb',
       'example': False},
      {'content': "I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco or any other location, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or AccuWeather. These sources can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details.",
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 78,
         'prompt_tokens': 23,
         'total_tokens': 101,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-eb507a89-ec6e-4093-b90c-21ef2de63788-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 23,
        'output_tokens': 78,
        'total_tokens': 101,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}},
      {'content': 'Did I ask about the weather in SF earlier?',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'ae8be155-1fdc-4b21-87ea-b51c0de08a09',
       'example': False},
      {'content': 'Yes, your previous question was about the weather in San Francisco (SF).',
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 16,
         'prompt_tokens': 118,
         'total_tokens': 134,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-3389d5bf-7c43-4281-af29-534228eb4ae7-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 118,
        'output_tokens': 16,
        'total_tokens': 134,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}}],
     'steps': [],
     'documents': [],
     'answer': 'Yes, your previous question was about the weather in San Francisco (SF).',
     'query': 'Did I ask about the weather in SF earlier?'}</pre>



```
'answer': 'Yes, your previous question was about the weather in San Francisco (SF).'
```
The system remembers your previous questions and responds based on that context.

```python
result = await remote_graph.ainvoke({
   "messages": [{"role": "user", "content": "Did I ask about the weather in LA earlier?"}]
}, config_thread)

result
```




<pre class="custom">{'messages': [{'content': "what's the weather in sf?",
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'eee694bf-4a7f-48f0-95f1-ef9e6ca97adb',
       'example': False},
      {'content': "I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco or any other location, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or AccuWeather. These sources can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details.",
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 78,
         'prompt_tokens': 23,
         'total_tokens': 101,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-eb507a89-ec6e-4093-b90c-21ef2de63788-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 23,
        'output_tokens': 78,
        'total_tokens': 101,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}},
      {'content': 'Did I ask about the weather in SF earlier?',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'ae8be155-1fdc-4b21-87ea-b51c0de08a09',
       'example': False},
      {'content': 'Yes, your previous question was about the weather in San Francisco (SF).',
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 16,
         'prompt_tokens': 118,
         'total_tokens': 134,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-3389d5bf-7c43-4281-af29-534228eb4ae7-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 118,
        'output_tokens': 16,
        'total_tokens': 134,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}},
      {'content': 'Did I ask about the weather in LA earlier?',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': '355e267e-a535-4961-82d4-264770c0c7a0',
       'example': False},
      {'content': 'No, you did not ask about the weather in Los Angeles (LA) earlier. Your previous question was about the weather in San Francisco (SF).',
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 31,
         'prompt_tokens': 151,
         'total_tokens': 182,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-3469b55a-57d6-41db-9a89-7a670c7fdd6f-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 151,
        'output_tokens': 31,
        'total_tokens': 182,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}}],
     'steps': [],
     'documents': [],
     'answer': 'No, you did not ask about the weather in Los Angeles (LA) earlier. Your previous question was about the weather in San Francisco (SF).',
     'query': 'Did I ask about the weather in LA earlier?'}</pre>



```
'answer': 'No, you did not ask about the weather in Los Angeles (LA) earlier. Your previous question was about the weather in San Francisco (SF).'
```
Since there was no prior request about LA weather, the system would respond accordingly, as demonstrated above.

### Remove the Thread ID
Now let's remove the thread ID. Removing the thread ID means the LLM will no longer retain context from previous conversations.

```python
config = {
    "configurable": {
        "embedding_model": "openai/text-embedding-3-small",
        "query_model": "openai/gpt-4-turbo-preview",
        "response_model": "openai/gpt-4-turbo-preview",
        "router_system_prompt": "You are a helpful assistant...",
        "research_plan_system_prompt": "You are a research planner...",
        "response_system_prompt": "You are an expert..."
}}
```

First, I will ask about the weather in San Francisco.

```python
# invoke the graph without the thread configs
result = await remote_graph.ainvoke({
   "messages": [{"role": "user", "content": "what's the weather in sf?"}]
}, config)

result
```




<pre class="custom">{'messages': [{'content': "what's the weather in sf?",
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': 'c283b57c-8873-4b67-bf19-a88a4d09bd01',
       'example': False},
      {'content': "I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco or any other location, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or a local news station's weather service. These sources update their information frequently to give you the most accurate and up-to-date weather forecasts.",
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 77,
         'prompt_tokens': 23,
         'total_tokens': 100,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-b6dd2b94-b8ad-41b4-874e-cc4ebd91bc7c-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 23,
        'output_tokens': 77,
        'total_tokens': 100,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}}],
     'steps': [],
     'documents': [],
     'answer': "I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco or any other location, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or a local news station's weather service. These sources update their information frequently to give you the most accurate and up-to-date weather forecasts.",
     'query': "what's the weather in sf?"}</pre>



Will the LLM remember that I previously asked about San Francisco's weather?

```python
# invoke the graph without the thread configs
result = await remote_graph.ainvoke({
   "messages": [{"role": "user", "content": "Did I ask about the weather in SF earlier?"}]
}, config)

result
```




<pre class="custom">{'messages': [{'content': 'Did I ask about the weather in SF earlier?',
       'additional_kwargs': {},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': '81fcf979-1317-4c9b-8ce8-f4012045eadf',
       'example': False},
      {'content': "I'm sorry, but I can't recall past interactions or questions. How can I assist you with information about the weather in San Francisco or anything else today?",
       'additional_kwargs': {'refusal': None},
       'response_metadata': {'token_usage': {'completion_tokens': 33,
         'prompt_tokens': 26,
         'total_tokens': 59,
         'completion_tokens_details': {'accepted_prediction_tokens': 0,
          'audio_tokens': 0,
          'reasoning_tokens': 0,
          'rejected_prediction_tokens': 0},
         'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
        'model_name': 'gpt-4-0125-preview',
        'system_fingerprint': None,
        'finish_reason': 'stop',
        'logprobs': None},
       'type': 'ai',
       'name': None,
       'id': 'run-b54a73f1-8742-4522-9c29-2d92652a1122-0',
       'example': False,
       'tool_calls': [],
       'invalid_tool_calls': [],
       'usage_metadata': {'input_tokens': 26,
        'output_tokens': 33,
        'total_tokens': 59,
        'input_token_details': {'audio': 0, 'cache_read': 0},
        'output_token_details': {'audio': 0, 'reasoning': 0}}}],
     'steps': [],
     'documents': [],
     'answer': "I'm sorry, but I can't recall past interactions or questions. How can I assist you with information about the weather in San Francisco or anything else today?",
     'query': 'Did I ask about the weather in SF earlier?'}</pre>



```
'answer': "I'm sorry, but I can't recall past interactions or questions. How can I assist you with information about the weather in San Francisco or anything else today?"
```
No, without a thread ID, it cannot retain memory of prior conversations.

```python
# verify that the state was persisted to the thread
thread_state = await remote_graph.aget_state(config_thread)
thread_state
```




<pre class="custom">StateSnapshot(values={'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'eee694bf-4a7f-48f0-95f1-ef9e6ca97adb', 'example': False}, {'content': "I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco or any other location, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or AccuWeather. These sources can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details.", 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 78, 'prompt_tokens': 23, 'total_tokens': 101, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run-eb507a89-ec6e-4093-b90c-21ef2de63788-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 23, 'output_tokens': 78, 'total_tokens': 101, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}, {'content': 'Did I ask about the weather in SF earlier?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'ae8be155-1fdc-4b21-87ea-b51c0de08a09', 'example': False}, {'content': 'Yes, your previous question was about the weather in San Francisco (SF).', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 16, 'prompt_tokens': 118, 'total_tokens': 134, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run-3389d5bf-7c43-4281-af29-534228eb4ae7-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 118, 'output_tokens': 16, 'total_tokens': 134, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}, {'content': 'Did I ask about the weather in LA earlier?', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '355e267e-a535-4961-82d4-264770c0c7a0', 'example': False}, {'content': 'No, you did not ask about the weather in Los Angeles (LA) earlier. Your previous question was about the weather in San Francisco (SF).', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 31, 'prompt_tokens': 151, 'total_tokens': 182, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run-3469b55a-57d6-41db-9a89-7a670c7fdd6f-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 151, 'output_tokens': 31, 'total_tokens': 182, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}], 'steps': [], 'documents': [], 'answer': 'No, you did not ask about the weather in Los Angeles (LA) earlier. Your previous question was about the weather in San Francisco (SF).', 'query': 'Did I ask about the weather in LA earlier?'}, next=(), config={'configurable': {'thread_id': '3aaea4b6-9e7a-4102-a968-8dbd2fd66a3a', 'checkpoint_ns': '', 'checkpoint_id': '1efe9fda-ec24-6070-8015-86d9a95a8640', 'checkpoint_map': {}}}, metadata={'embedding_model': 'openai/text-embedding-3-small', 'query_model': 'openai/gpt-4-turbo-preview', 'response_model': 'openai/gpt-4-turbo-preview', 'router_system_prompt': 'You are a helpful assistant...', 'research_plan_system_prompt': 'You are a research planner...', 'response_system_prompt': 'You are an expert...', 'langgraph_auth_user': None, 'langgraph_auth_user_id': '', 'langgraph_auth_permissions': [], 'graph_id': 'chat', 'assistant_id': 'eb6db400-e3c8-5d06-a834-015cb89efe69', 'user_id': '', 'created_by': 'system', 'run_attempt': 1, 'langgraph_version': '0.2.70', 'langgraph_plan': 'developer', 'langgraph_host': 'self-hosted', 'thread_id': '3aaea4b6-9e7a-4102-a968-8dbd2fd66a3a', 'run_id': '1efe9fda-54b6-683c-a30c-372aa32a56e5', 'source': 'loop', 'writes': {'respond': {'messages': [{'content': 'No, you did not ask about the weather in Los Angeles (LA) earlier. Your previous question was about the weather in San Francisco (SF).', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 31, 'prompt_tokens': 151, 'total_tokens': 182, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run-3469b55a-57d6-41db-9a89-7a670c7fdd6f-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 151, 'output_tokens': 31, 'total_tokens': 182, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}], 'answer': 'No, you did not ask about the weather in Los Angeles (LA) earlier. Your previous question was about the weather in San Francisco (SF).'}}, 'step': 21, 'parents': {}}, created_at='2025-02-13T11:28:43.973827+00:00', parent_config={'configurable': {'thread_id': '3aaea4b6-9e7a-4102-a968-8dbd2fd66a3a', 'checkpoint_ns': '', 'checkpoint_id': '1efe9fda-d9d5-65fe-8014-58c23188e2d7', 'checkpoint_map': {}}}, tasks=())</pre>



## Using as a subgraph
This code explains how to use a remote graph (RemoteGraph) as a subgraph of another graph. Here's a summary of the main points:

1. Setting up the remote graph:
   - Import a graph deployed on a remote server as a `RemoteGraph`.

2. Creating the parent graph:
   - Use `StateGraph` to create a new graph.
   - This graph manages message states.

3. Adding the remote graph as a subnode:
   - Directly add the remote graph as a node in the parent graph.
   - Create a connection from the start node to this subgraph.

4. Executing the graph:
   - Run the completed graph to obtain results.
   - Results can also be received in a streaming manner.

This approach allows easy integration of complex remote graphs as part of a local graph. This helps increase modularity and reusability.

```python
from langgraph.graph import StateGraph, MessagesState, START

# define parent graph
builder = StateGraph(MessagesState)
# add remote graph directly as a node
builder.add_node("child", remote_graph)
builder.add_edge(START, "child")
graph = builder.compile()

# invoke the parent graph
result = await graph.ainvoke({
   "messages": [{"role": "user", "content": "what's the weather in sf"}]
}, config)
result
```




<pre class="custom">{'messages': [HumanMessage(content="what's the weather in sf", additional_kwargs={'additional_kwargs': {'additional_kwargs': {}, 'response_metadata': {}, 'example': False}, 'response_metadata': {}, 'example': False}, response_metadata={}, id='c785d7b5-d1a7-4e74-87dd-d930c54f83b9'),
      AIMessage(content="I'm sorry, but I can't provide real-time weather updates or forecasts. For the most current weather conditions in San Francisco, I recommend checking a reliable weather website or app like the National Weather Service, Weather.com, or AccuWeather. They can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details.", additional_kwargs={'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 73, 'prompt_tokens': 22, 'total_tokens': 95, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'example': False, 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 22, 'output_tokens': 73, 'total_tokens': 95, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}, response_metadata={}, id='run-eb18497c-3536-4975-9a38-951b80a8242c-0')]}</pre>



```python
# stream outputs from both the parent graph and subgraph
async for chunk in remote_graph.astream(
    {"messages": [{"role": "user", "content": "what's the weather in la"}]}, 
    config,
    subgraphs=True
):
    print(chunk)
```

<pre class="custom">((), {'create_research_plan': {'steps': ['Check the current weather in Los Angeles using a reliable weather forecasting service.', 'Note the temperature, humidity, wind speed, and any precipitation.', 'Check the forecast for the next few days to provide a short-term outlook.', 'Consider any weather advisories or warnings in effect for the area.', 'Summarize the findings to give a comprehensive overview of the weather in Los Angeles.'], 'documents': 'delete', 'query': "what's the weather in la"}})
    ((), {'conduct_research': {'documents': [], 'steps': ['Note the temperature, humidity, wind speed, and any precipitation.', 'Check the forecast for the next few days to provide a short-term outlook.', 'Consider any weather advisories or warnings in effect for the area.', 'Summarize the findings to give a comprehensive overview of the weather in Los Angeles.']}})
    ((), {'conduct_research': {'documents': [], 'steps': ['Check the forecast for the next few days to provide a short-term outlook.', 'Consider any weather advisories or warnings in effect for the area.', 'Summarize the findings to give a comprehensive overview of the weather in Los Angeles.']}})
    ((), {'conduct_research': {'documents': [], 'steps': ['Consider any weather advisories or warnings in effect for the area.', 'Summarize the findings to give a comprehensive overview of the weather in Los Angeles.']}})
    ((), {'conduct_research': {'documents': [], 'steps': ['Summarize the findings to give a comprehensive overview of the weather in Los Angeles.']}})
    ((), {'conduct_research': {'documents': [], 'steps': []}})
    ((), {'respond': {'messages': [{'content': "I'm sorry, but I can't provide real-time weather updates or forecasts. To get the current weather conditions in Los Angeles or any other location, I recommend checking a reliable weather website like the National Weather Service, Weather.com, or using a weather app on your smartphone. These sources can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details.", 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 80, 'prompt_tokens': 22, 'total_tokens': 102, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run-3d7be337-d1d0-40e3-a55b-bf691b81a0b1-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 22, 'output_tokens': 80, 'total_tokens': 102, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}], 'answer': "I'm sorry, but I can't provide real-time weather updates or forecasts. To get the current weather conditions in Los Angeles or any other location, I recommend checking a reliable weather website like the National Weather Service, Weather.com, or using a weather app on your smartphone. These sources can provide you with up-to-date information on temperature, precipitation, wind speed, and other weather-related details."}})
</pre>

## Summary
Unfortunately, Langchain is not currently recruiting for beta testing of Langchain deploy. However, you could try deploying through various hosting services locally.
