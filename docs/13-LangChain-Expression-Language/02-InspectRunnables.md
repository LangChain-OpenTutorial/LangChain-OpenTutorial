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

# Inspect Runnables

- Author: [ranian963](https://github.com/ranian963)
- Peer Review: []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/)

## Overview
In this tutorial, we introduce how to **inspect** and visualize various components (including the graph structure) of a `Runnable` chain. Understanding the underlying graph structure can help diagnose and optimize complex chain flows.

### Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Introduction to Inspecting Runnables](#introduction-to-inspecting-runnables)
  - [Graph Inspection](#graph-inspection)
  - [Graph Output](#graph-output)
  - [Prompt Retrieval](#prompt-retrieval)

  ### References
- [LangChain: Runnables](https://python.langchain.com/api_reference/core/runnables.html)
- [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)
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
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "faiss-cpu",
        "grandalf",
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
        "LANGCHAIN_PROJECT": "02-InspectRunnables",
    }
)
```

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```

## Introduction to Inspecting Runnables

LangChain `Runnable` objects can be composed into pipelines, commonly referred to as **chains** or **flows**. After setting up a `runnable`, you might want to **inspect its structure** to see what's happening under the hood.

By inspecting these, you can:
- Understand the sequence of transformations and data flows.
- Visualize the graph for debugging.
- Retrieve or modify prompts or sub-chains as needed.


### Graph Inspection

We'll create a runnable chain that includes a retriever from FAISS, a prompt template, and a ChatOpenAI model. Then we’ll **inspect the chain’s graph** to understand how data flows between these components.


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Create a FAISS vector store from simple text data
vectorstore = FAISS.from_texts(
    ["Teddy is an AI engineer who loves programming!"], embedding=OpenAIEmbeddings()
)

# Create a retriever based on the vector store
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"""
# Create a prompt template
prompt = ChatPromptTemplate.from_template(template)

# Initialize ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Construct the chain: (dictionary format) => prompt => model => output parser
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }  # Search context and question
    | prompt
    | model
    | StrOutputParser()
)
```

### Graph Output
We can **inspect** the chain’s internal graph of nodes (steps) and edges (data flows).

```python
# Get nodes from the chain's graph
chain.get_graph().nodes
```

```python
# Get edges from the chain's graph
chain.get_graph().edges
```

We can also print the graph in an ASCII-based diagram to visualize the chain flow.

```python
chain.get_graph().print_ascii()
```

### Prompt Retrieval
Finally, we can retrieve the **actual prompts** used in this chain. This is helpful to see exactly what LLM instructions are being sent.


```python
chain.get_prompts()
```
