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

# Embedding-based Evaluator(embedding_distance)

- Author: [Youngjun Cho](https://github.com/choincnp)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

The **Embedding-based Evaluator** (`embedding_distance`) part is designed to evaluate question-answering systems using various **embedding models** and **distance metrics** .

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Defining functions for rag performance testing](#defining-functions-for-rag-performance-testing)
- [Embedding distance based evaluator](#embedding-distance-based-evaluator)

### References

- [LangChain OpenAIEmbeddings](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- [LangChain StringEvaluator](https://python.langchain.com/api_reference/langchain/evaluation/langchain.evaluation.schema.StringEvaluator.html)

----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.


**[Note]**

The langchain-opentutorial is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
Check out the  [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_community",
        "langchain_openai",
        "langchain_upstage",
        "PyMuPDF"
    ],
    verbose=False,
    upgrade=False,
)
```

You can set API keys in a `.env` file or set them manually.

[Note] If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "UPSTAGE_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "06-LangSmith-Embedding-Distance-Evaluation",  # set the project name same as the title
        }
    )
```

## Defining Functions for RAG Performance Testing

We will create a RAG system for testing purposes.

```python
from myrag import PDFRAG
from langchain_openai import ChatOpenAI

# Create a PDFRAG object
rag = PDFRAG(
    "data/Newwhitepaper_Agents2.pdf",
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# Create a retriever
retriever = rag.create_retriever()

# Create a chain
chain = rag.create_chain(retriever)

# Generate an answer for a question
chain.invoke("How do agents differ from standalone language models?")
```




<pre class="custom">'Agents differ from standalone language models in that agents extend the capabilities of language models by leveraging tools to access real-time information, suggest real-world actions, and plan and execute complex tasks autonomously. While standalone language models are limited to the knowledge available in their training data, agents can enhance their knowledge through connections to external resources and tools, allowing them to perform more dynamic and complex functions.'</pre>



Create a function named `ask_question` to handle answering questions. The function takes a dictionary `inputs` as input and returns a dictionary `answer` as output.

```python
# Create a function to answer questions
def ask_question(inputs: dict):
    return {"answer": chain.invoke(inputs["question"])}
```

## Embedding Distance-based Evaluator

We will build a system for evaluating sentence similarity using various embedding models and distance metrics. 

The code below defines configurations for each model and metric using the `LangChainStringEvaluator`.

[ **Note** ]  
For LangChainStringEvaluator, `OpenAIEmbeddings` is set as the default, but it can be changed.

```python
from langsmith.evaluation import LangChainStringEvaluator
from langchain_upstage import UpstageEmbeddings
from langchain_openai import OpenAIEmbeddings

# Create an embedding model evaluator
openai_embedding_cosine_evaluator = LangChainStringEvaluator(
    "embedding_distance",
    config={
        # OpenAIEmbeddings is set as the default, but can be changed
        "embeddings": OpenAIEmbeddings(model="text-embedding-3-small"),
        "distance_metric": "cosine",  # "cosine", "euclidean", "chebyshev", "hamming", and "manhattan"
    },
)

upstage_embedding_evaluator = LangChainStringEvaluator(
    "embedding_distance",
    config={
        # OpenAIEmbeddings is set as the default, but can be changed
        "embeddings": UpstageEmbeddings(model="embedding-query"),
        "distance_metric": "euclidean",  # "cosine", "euclidean", "chebyshev", "hamming", and "manhattan"
    },
)

openai_embedding_evaluator = LangChainStringEvaluator(
    "embedding_distance",
    config={
        # OpenAIEmbeddings is set as the default, but can be changed
        "embeddings": OpenAIEmbeddings(model="text-embedding-3-small"),
        "distance_metric": "euclidean",  # "cosine", "euclidean", "chebyshev", "hamming", and "manhattan"
    },
)
```

When multiple embedding models are used for **one metric** , the results are averaged.

Example:
- `cosine` : OpenAI
- `euclidean` : OpenAI, Upstage

For `euclidean` , the average value across the models is calculated.

```python
from langsmith.evaluation import evaluate

dataset_name = "RAG_EVAL_DATASET"

# Run evaluation
experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=[
        openai_embedding_cosine_evaluator,
        upstage_embedding_evaluator,
        openai_embedding_evaluator,
    ],
    experiment_prefix="EMBEDDING-EVAL",
    # Specify experiment metadata
    metadata={
        "variant": "Evaluation using embedding_distance",
    },
)
```

<pre class="custom">View the evaluation results for experiment: 'EMBEDDING-EVAL-e7657248' at:
    https://smith.langchain.com/o/9089d1d3-e786-4000-8468-66153f05444b/datasets/9b4ca107-33fe-4c71-bb7f-488272d895a3/compare?selectedSessions=43f0123f-de7a-4434-ab59-b4ff06134982
    
    
</pre>


    0it [00:00, ?it/s]


![](./assets/06-langSmith-embedding-distance-evaluation-01.png)
