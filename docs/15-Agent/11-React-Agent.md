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

# React Agent

- Author: [ranian963](https://github.com/ranian963)
- Peer Review:

- This is part of the [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial)

## Overview

In this tutorial, we explore the concept and implementation of a **ReAct Agent**.

**ReAct Agent** stands for Reasoning + Action, meaning that the LLM explicitly goes through a reasoning phase, utilizes tools (or actions), and then generates the final answer based on the obtained results.

Throughout this tutorial, we will implement a ReAct Agent by covering the following:

- **Tool Setup**: Utilizing various tools such as web search, file management, document search based on VectorStore, etc.
- **Agent Creation**: Practice how to use the ReAct Agent in LangChain.
- **Graph Execution**: Execute the agent to observe the answers to queries.

Please follow the sections below to go through the entire process.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Tools](#tools)
  - [Web Search](#web-search)
  - [File Management](#file-management)
  - [Retriever Tool](#retriever-tool)
- [Create and Visualize the Agent](#create-and-visualize-the-agent)
- [Execute Agent](#execute-agent)

### References
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al.)](https://arxiv.org/abs/2210.03629)
- [LangChain Official Documentation](https://langchain.readthedocs.io/en/latest/)
- [LangChain-OpenTutorial GitHub](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)
----


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides an easy-to-use environment setup, useful functions, and utilities for tutorials.
- For more details, check out [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi).


```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "langgraph",
        "faiss-cpu",
        "pymupdf",
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
        "LANGCHAIN_PROJECT": "11-React-Agent",
        "TAVILY_API_KEY": "",
    }
)
```

You can alternatively set `OPENAI_API_KEY` in a `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` previously.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```

**How to Set Up Tavily Search**

- **Get an API Key**:

  To use Tavily Search, you need an API key.

  - [Generate your Tavily Search API key](https://app.tavily.com/sign-in)


## Tools

A ReAct Agent uses various **tools** to solve problems. In this tutorial, we will set up and use the following tools.


### Web Search

We use the `TavilySearch` tool for searching the latest information from the web.
The code below creates a web search tool and tests it by retrieving some search results.


```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Create an instance of TavilySearchResults with k=5 for retrieving up to 5 search results
web_search = TavilySearchResults(k=5)
```

```python
# Test the web search tool.
result = web_search.invoke(
    "Please find information related to the major AI-related from CES 2025"
)
result
```

### File Management

Using `FileManagementToolkit`, you can create, delete, and modify files.


```python
from langchain_community.agent_toolkits import FileManagementToolkit

# Set 'tmp' as the working directory.
working_directory = "tmp"
file_management_tools = FileManagementToolkit(
    root_dir=str(working_directory)
).get_tools()
file_management_tools
```

### Retriever Tool

To search for information within documents such as PDFs, we create a Retriever. First, we load the PDF, split the text, then embed it into a VectorStore.

**Tesla's Revenue Forecast Based on Business Model and Financial Statement Analysis**  

**Author:** Chenhao Fang  
**Institution:** Intelligent Accounting Management Institute, Guangdong University of Finance and Economics  
**Link:** [Tesla's revenue forecast base on business model and financial statement analysis ](https://www.shs-conferences.org/articles/shsconf/pdf/2024/01/shsconf_icdeba2023_02022.pdf)  
**File Name:** shsconf_icdeba2023_02022.pdf

_Please copy the downloaded file to the data folder for practice._



```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

# Example PDF file path (adjust according to your environment)
pdf_file_path = "data/shsconf_icdeba2023_02022.pdf"

# Load the PDF using PyMuPDFLoader
loader = PyMuPDFLoader(pdf_file_path)

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = loader.load_and_split(text_splitter)

# Create FAISS VectorStore
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Create a retriever from the VectorStore
pdf_retriever = vector.as_retriever()
```

```python
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate

# Create a tool for PDF-based search
retriever_tool = create_retriever_tool(
    retriever=pdf_retriever,
    name="pdf_retriever",
    description="use this tool to search for information in Tesla PDF file",
    document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
    ),
)
```

Combine these tools into a single list.


```python
tools = [web_search, *file_management_tools, retriever_tool]
tools
```

## Create and Visualize the Agent

We will now create a ReAct Agent and visualize the agent graph.
In LangChain, a **ReAct** agent generates answers through step-by-step reasoning and tool usage.

The code below uses `create_react_agent` from **langgraph** to easily build a ReAct Agent and visualize its structure as a graph.


```python
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Memory and model configuration
memory = MemorySaver()
model = ChatOpenAI(model_name="gpt-4o-mini")

# Create ReAct Agent
agent = create_react_agent(model, tools=tools, checkpointer=memory)
```

The code below visualizes the agent's graph structure.


```python
from IPython.display import Image, display

display(
    Image(agent.get_graph().draw_mermaid_png(output_file_path="11-React-Agent.png"))
)
```

```python
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
```

## Execute Agent

Let's execute the created ReAct Agent. We can track the step-by-step process of generating an answer to a query.

The example code below uses `stream_graph` to stream the agent's execution process. We place the user's query in the `messages` key, and the agent's reasoning will be displayed.


```python
config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [("human", "Hello, my name is Teddy.")]}

print_stream(agent.stream(inputs, config=config, stream_mode="values"))
```

```python
# Another example - maintaining chat flow
config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [("human", "What was my name again?")]}

print_stream(agent.stream(inputs, config=config, stream_mode="values"))
```

```python
config = {"configurable": {"thread_id": "1"}}
inputs = {
    "messages": [("human", "Please summarize Tesla from shsconf_icdeba2023_02022.pdf.")]
}

print_stream(agent.stream(inputs, config=config, stream_mode="values"))
```

```python
# Example of using web search + file management
config = {"configurable": {"thread_id": "1"}}
inputs = {
    "messages": [
        (
            "human",
            "Search for news about American writer John Smith's Pulitzer Prize and draft a brief report based on it.",
        )
    ]
}

print_stream(agent.stream(inputs, config=config, stream_mode="values"))
```

```python
# A more concrete scenario example
instruction = """
Your task is to write a 'press release.'
----
Please process the following steps in order:
1. Search for news about American writer John Smith's Pulitzer Prize.
2. Based on the Pulitzer Prize news, write a press release/report.
3. Actively use markdown table format to summarize key points.
4. Save the output to a file named `agent_press_release.md`.
"""

config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [("human", instruction)]}
print_stream(agent.stream(inputs, config=config, stream_mode="values"))
```
