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

# Upstage

- Author: [Sun Hyoung Lee](https://github.com/LEE1026icarus)
- Peer Review : [Pupba](https://github.com/pupba), [DoWoung Kong](https://github.com/krkrong)
- Proofread : [Youngjun cho](https://github.com/choincnp)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/04-UpstageEmbeddings.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/04-UpstageEmbeddings.ipynb)
## Overview

'Upstage' is a Korean startup specializing in artificial intelligence (AI) technology, particularly in large language models (LLM) and document AI.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)


### References

- [Upstage API docs](https://console.upstage.ai/docs/getting-started/overview)
- [Upstage Embeddings](https://console.upstage.ai/docs/capabilities/embeddings)
---


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

 **[Note]** 
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

### API Key Configuration
To use ```UpstageEmbeddings``` , you need to [obtain a Upstage API key](https://console.upstage.ai/api-keys).

Once you have your API key, set it as the value for the variable ```UPSTAGE_API_KEY``` .

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    ["langchain_community"],
    verbose=False,
    upgrade=False,
)
```

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "UPSTAGE_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "CH08-Embeddings-UpstageEmebeddings",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```UPSTAGE_API_KEY``` in ```.env``` file and load it.

[Note] This is not necessary if you've already set ```UPSTAGE_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
texts = [
    "Hello, nice to meet you.",
    "LangChain simplifies the process of building applications with large language models",
    "The LangChain Korean tutorial is designed to help users utilize LangChain more easily and effectively based on LangChain's official documentation, cookbook, and various practical examples.",
    "LangChain simplifies the process of building applications with large-scale language models.",
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.",
]
```

**Check Supported Embedding Models**

- https://developers.upstage.ai/docs/apis/embeddings

**Model Information**

| Model                              | Release Date | Context Length | Description                                                                                         |
|------------------------------------|--------------|----------------|-----------------------------------------------------------------------------------------------------|
| embedding-query      | 2024-05-10   | 4000           | A Solar-base Query Embedding model with a 4k context limit. This model is optimized for embedding user queries in information retrieval tasks such as search and re-ranking. |
| embedding-passage    | 2024-05-10   | 4000           | A Solar-base Passage Embedding model with a 4k context limit. This model is optimized for embedding documents or texts for retrieval purposes. |

```python
from langchain_upstage import UpstageEmbeddings

# Query-Only Embedding Model
query_embeddings = UpstageEmbeddings(model="embedding-query")

# Sentence-Only Embedding Model
passage_embeddings = UpstageEmbeddings(model="embedding-passage")
```

Embed the ```query```

```python
# Query Embedding
embedded_query = query_embeddings.embed_query(
    " Please provide detailed information about LangChain. "
)
# Print embedding dimension
len(embedded_query)
```




<pre class="custom">4096</pre>



Embed the document

```python
# Document Embedding
embedded_documents = passage_embeddings.embed_documents(texts)
```

The similarity calculation results are displayed.

```python
import numpy as np

# Question (embedded_query): Tell me about LangChain.
similarity = np.array(embedded_query) @ np.array(embedded_documents).T

# Sort by similarity in descending order
sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]

# Display results
print("[Query] Tell me about LangChain.\n====================================")
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] Similarity: {similarity[idx]:.3f} | {texts[idx]}")
    print()
```

<pre class="custom">[Query] Tell me about LangChain.
    ====================================
    [0] Similarity: 0.535 | LangChain simplifies the process of building applications with large-scale language models.
    
    [1] Similarity: 0.519 | LangChain simplifies the process of building applications with large language models
    
    [2] Similarity: 0.509 | The LangChain Korean tutorial is designed to help users utilize LangChain more easily and effectively based on LangChain's official documentation, cookbook, and various practical examples.
    
    [3] Similarity: 0.230 | Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.
    
    [4] Similarity: 0.158 | Hello, nice to meet you.
    
</pre>
