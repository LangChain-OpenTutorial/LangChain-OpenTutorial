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

# Deploy on LangGraph Cloud

- Author: [JoonHo Kim](https://github.com/jhboyo)
- Design: []()
- Peer Review :
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/04-CSVLoader.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/04-CSVLoader.ipynb)


## Overview
**LangGraph Cloud** is a cloud-based framework designed to simplify the development, deployment, and management of graph-based workflows for AI applications. It extends the functionality of **LangGraph** by providing a scalable, distributed, and user-friendly environment to build complex AI agents, workflows, and pipelines.

With **LangGraph Cloud**, you can:
- Handle large workloads with horizontally-scaling servers, task queues, and built-in persistence.
- Debug agent failure modes and quickly iterate in a visual playground-like studio.
- Deploy in one-click and get integrated tracing & monitoring in LangSmith.

This tutorial will guide you through the key features and components of LangGraph Cloud, including:
- Setting up **LangGraph Cloud**: How to create an account, configure your workspace, and deploy your first workflow.
- Deploying workflows: Deploying workflows using LangGraph Cloud.
- Using **LangGraph Studio**: How to connect to the Web UI **LangGraph Studio** and test the assistant.
- Testing the API: How to send messages to the assistant using the **Python SDK** and verify the message data using **Rest API**.

By the end of this tutorial, you will be equipped with the knowledge to effectively utilize **LangGraph Cloud** for building and managing AI workflows in a scalable and efficient manner.

Now, let's dive in and explore how to boost performance with **LangGraph Cloud**! 🚀



### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Prerequisites](#prerequisites)
- [Setting up a new repository on GitHub](#setting-up-a-new-repository-on-github)
- [Deployment to LangGraph Cloud](#deployment-to-langgraph-cloud)
- [Using LangGraph Studio on the web](#using-langgraph-studio-on-the-web)
- [Testing the API](#testing-the-api)

### References

- [Deploy on LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/quick_start/#deploy-to-langgraph-cloud)
- [React Agent](https://github.com/langchain-ai/react-agent)
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#cloud-studio)

----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install -qU langchain-opentutorial
```

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "LangGraph-Cloud",
        }
    )
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langgraph-sdk",
    ],
    verbose=False,
    upgrade=False,
)
```


## Prerequisites
Before we start, ensure we have the following:

- [GitHub Account](https://github.com/join)
- [LangSmith API key](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key/)
- [Anthropic API key](https://console.anthropic.com/settings/keys)
- [Tavily API key](https://app.tavily.com/home)

## Setting up a new repository on GitHub

To deploy a **LangGraph** application on **LangGraph Cloud**, your application's code must be stored in a **GitHub** repository. 
You can deploy any **LangGraph** applications to **LangGraph Cloud** with ease. 

For this guide, we'll use the pre-built **Python** [`ReAct Agent`](https://github.com/langchain-ai/react-agent) template. You can go to **GitHub** and fork the repository.
This [`ReAct Agent`](https://github.com/langchain-ai/react-agent) application requires API keys from **Anthropic** and **Tavily**.

## Deployment to LangGraph Cloud

1. After logging in **[LangSmith](https://smith.langchain.com)**, you can click the **LangGraph Platform** menu at the bottom of the left sidebar.


    <img src="./assets/12-langgraph-cloud-sidebar-menu.png" width="280">

2. Click **+ New Deployment** button at the bottom of the page and then you can follow the steps below for creating a new deployment.
    
    * Select **Github react-agent** repository from the drop-down menu.
    * Write the deployment name in **Name** field.
    * Select Git branch. **main** is default.
    * Langgraph config file is **langgraph.json** as default. You can also select another file.
    * Select **Development type**.
    * Write **Environment Variables**. In this tutorial, we will use **ANTHROPIC_API_KEY** and **TAVILY_API_KEY**.
    * Click **Submit** button at the upper right corner. It takes a few minutes to build the application.

    <img src="./assets/12-langgraph-cloud-create-deployment-1.png" width="1150">
        


3. Now you can see the deployment status on the **Overview** section.

    <img src="./assets/12-langgraph-cloud-deployment-status.png" width="500">


## Using LangGraph Studio on the web

Once your application is deployed, you can test it in **LangGraph Studio** on the web.

You can find the **LangGraph Studio** text and **Endpoint URL** at the bottom of the page. Let's click the **LangGraph Studio** text to copy the clipboard.

<img src="./assets/12-langgraph-cloud-langgraph-platform.png" width="1150">


Now you can test your LangGraph application in **LangGraph Studio** on the web.

<img src="./assets/12-langgraph-cloud-langgraph-studio.png" width="1150">





##  Testing the API

Now we will send messages to the assistant using the **Python SDK**. you can also use **JavaScript SDK** or **Rest API**.

Prior to this, we need to install the **langgraph-sdk** package.
```python
pip install langgraph-sdk or poetry add langgraph-sdk
```














```python
from langgraph_sdk import get_client

# Initialize the LangGraph client with the endpoint URL and API key
client = get_client(url="{your_endpoint_url}", api_key="{your_langsmith_key}")

# Stream the response from the LangGraph assistant
async for chunk in client.runs.stream(
    None,       # Run without a specific thread (threadless run)
    "agent",    # Name of assistant. Defined in langgraph.json.
    input={
        "messages": [{
            "role": "human",    # User's role in the conversation
            "content": "say hello in french",   # User's input message
        }],
    },
    stream_mode="updates",  # Stream mode to receive updates in real-time
):
    print(chunk.data)   # Print the streamed data (assistant's response)
```

<pre class="custom">{'run_id': '1efef2b1-a250-6226-a457-8760b437980b', 'attempt': 1}
    {'call_model': {'messages': [{'content': 'To say "hello" in French, you don\'t need any special tools or searches. The French word for "hello" is:\n\n"Bonjour"\n\nThis is the most common and formal way to say hello in French. It literally translates to "good day" and can be used at any time of day.\n\nThere are also other ways to greet someone in French, depending on the time of day or the level of formality:\n\n1. "Salut" - A more casual way to say "hi" or "hey"\n2. "Bonsoir" - Used in the evening, meaning "good evening"\n3. "Coucou" - Very informal, similar to "hey there" or "hi there"\n\nIs there anything else you\'d like to know about French greetings or language?', 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_01L2AGDYYLD8ADWyuAonxk55', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 430, 'output_tokens': 181}}, 'type': 'ai', 'name': None, 'id': 'run-885a1808-0a9a-4763-a761-91972062d8c6-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 430, 'output_tokens': 181, 'total_tokens': 611, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}
</pre>

Now, you can verify the message data.

If you append `/docs` to the end of the **Endpoint URL** and enter it in a web browser, you can check the web API. We can refer to this document and use API testing tools like Postman or Scalar to conduct tests.

ex) `GET https://{{endpoint_url}}threads/{{thread_id}}/history`

<img src="./assets/12-langgraph-cloud-web-api-1.png" width="650">



