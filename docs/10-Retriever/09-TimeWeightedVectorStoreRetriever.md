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

# TimeWeightedVectorStoreRetriever

- Author: [Youngjun Cho](https://github.com/choincnp)
- Peer Review : []()
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/09-TimeWeightedVectorStoreRetriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/09-TimeWeightedVectorStoreRetriever.ipynb)
## Overview

```TimeWeightedVectorStoreRetriever``` is a retriever that uses a combination of semantic similarity and a time decay. 

By doing so, it considers both the " **freshness** " and " **relevance** " of the documents or data in its results.

The algorithm for scoring them is:  

> $\text{semantic\_similarity} + (1.0 - \text{decay\_rate})^{hours\_passed}$

- ```semantic_similarity``` indicates the semantic similarity between documents or data.
- ```decay_rate``` represents the ratio at which the score decreases over time.
- ```hours_passed``` is the number of hours elapsed since the object was last accessed.

The key feature of this approach is that it evaluates the “ **freshness of information** ” based on the last time the object was accessed. 

In other words, **objects that are accessed frequently maintain a higher score** over time, increasing the likelihood that **frequently used or important information will appear near the top** of search results. This allows the retriever to provide dynamic results that account for both recency and relevance.

Importantly, in this context, ```decay_rate``` is determined by the **time since the object was last accessed** , not since it was created. 

Hence, any objects that are accessed frequently remain "fresh."

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Low decay_rate](#low-decay_rate)
- [High decay_rate](#high-decay_rate)
- [Summary of the decay_rate](#summary-of-the-decay_rate)
- [Testing with Virtual time](#testing-with-virtual-time)

### References

- [Time-weighted vector store retriever](https://python.langchain.com/docs/how_to/time_weighted_vectorstore/)
- [TimeWeightedVectorStoreRetriever](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.time_weighted_retriever.TimeWeightedVectorStoreRetriever.html)
- [mock_now](https://python.langchain.com/api_reference/core/utils/langchain_core.utils.utils.mock_now.html)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_openai",
        "faiss-cpu"
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
        "LANGCHAIN_PROJECT": "TimeWeightedVectorStoreRetriever",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```

## Low decay_rate

- A low ```decay_rate``` (In this example, we'll set it to an extreme value close to 0) means that **memories are retained for a longer period** .

- A ```decay_rate``` of **0 means that memories are never forgotten** , which makes this retriever equivalent to a vector lookup.

Let's first initialize the ```TimeWeightedVectorStoreRetriever``` with a very small ```decay_rate``` and ```k=1``` (where ```k``` is the number of vectors to retrieve).

```python
from datetime import datetime, timedelta

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Define the embedding model.
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store empty.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# Initialize the time-weighted vector store retriever. (Here, we'll apply a very small decay_rate)
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.0000000000000000000000001, k=1
)
```

Let's add a simple example data.

```python
# Calculate the date of yesterday
yesterday = datetime.now() - timedelta(days=1)

retriever.add_documents(
    # Add a document with yesterday's date in the metadata
    [
        Document(
            page_content="Please subscribe to LangChain Youtube.",
            metadata={"last_accessed_at": yesterday},
        )
    ]
)

# Add another document. No metadata is specified here.
retriever.add_documents(
    [Document(page_content="Will you subscribe to LangChain Youtube? Please!")]
)
```




<pre class="custom">['58449575-d54f-47dc-9a76-806eccb850f3']</pre>



```python
# Invoke the retriever to search
retriever.invoke("LangChain Youtube")
```




<pre class="custom">[Document(metadata={'last_accessed_at': datetime.datetime(2025, 1, 7, 10, 19, 14, 305565), 'created_at': datetime.datetime(2025, 1, 7, 10, 19, 2, 632517), 'buffer_idx': 0}, page_content='Please subscribe to Langchain Youtube.')]</pre>



- The document "Please subscribe to LangChain Youtube" appears first because it is the **most salient** .

- Since the ```decay_rate``` is close to 0, the document is still considered **recent** .

## High decay_rate

When a high ```decay_rate``` is used (e.g., 0.9999...), the **recency score** rapidly converges to 0.

If this value were set to 1, all objects would end up with a ```recency``` value of 0, resulting in the same outcome as a standard vector lookup. 

Initialize the retriever using ```TimeWeightedVectorStoreRetriever``` , setting the ```decay_rate``` to 0.999 to adjust the time-based weight decay rate.

```python
# Define the embedding model.
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store empty.
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# Initialize the time-weighted vector store retriever.
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.999, k=1
)
```

Add new documents again.

```python
# Calculate the date of yesterday
yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents(
    [
        Document(
            page_content="Please subscribe to LangChain Youtube.",
            metadata={"last_accessed_at": yesterday},
        )
    ]
)
retriever.add_documents(
    [Document(page_content="Will you subscribe to LangChain Youtube? Please!")]
)
```




<pre class="custom">['68d6e6ce-8ab7-4c40-aaf9-1d852eedcb49']</pre>



```python
# Invoke the retriever to search
retriever.invoke("LangChain Youtube")
```




<pre class="custom">[Document(metadata={'last_accessed_at': datetime.datetime(2025, 1, 7, 10, 29, 2, 687697), 'created_at': datetime.datetime(2025, 1, 7, 10, 28, 37, 213151), 'buffer_idx': 1}, page_content='Will you subscribe to Langchain Youtube? Please!')]</pre>



In this case, when you invoke the retriever, "Will you subscribe to LangChain Youtube? Please!" is returned first.
- Because ```decay_rate``` is high (close to 1), older documents (like the one from yesterday) are nearly forgotten.

## Summary of the decay_rate

- when the ```decay_rate``` is set to a very small value, such as 0.000001:
    - The decay rate (i.e., the rate at which information is forgotten) is extremely low, so information is hardly forgotten.
    - As a result, there is almost **no difference in time-based weights between more or less recently accessed information** . In this case, similarity scores are given higher priority.

- When the ```decay_rate``` is set close to 1, such as 0.999:
    - The decay rate is very high, so most of the recently unaccessed information is almost completely forgotten.
    - As a result, in such cases, higher scores are given to more recently accessed information.


## Testing with Virtual Time

```LangChain``` provides some utilities that allow you to test time-based components by mocking the current time.

- The ```mock_now``` function is a utility function provided by LangChain, used to mock the current time.

[**NOTE**]  
Inside the with statement, all ```datetime.now``` calls return the **mocked time** . Once you **exit** the with block, it reverts back to the **original time** .

```python
import datetime
from langchain_core.utils import mock_now

# Define a function that print current time
def print_current_time():
    now = datetime.datetime.now()
    print(f"now is: {now}\n")

# Print the current time
print("before mocking")
print_current_time()

# Set the current time to a specific point in time
with mock_now(datetime.datetime(2025, 1, 7, 00, 00)):
    print("with mocking")
    print_current_time()

# Print the new current time(without mock_now block)
print("without mock_now block")
print_current_time()
```

<pre class="custom">before mocking
    now is: 2025-01-07 14:06:37.961348
    
    with mocking
    now is: 2025-01-07 00:00:00
    
    without mock_now block
    now is: 2025-01-07 14:06:37.961571
    
</pre>

By using the ```mock_now``` function, you can shift the current time and see how the search results change.
- This helps you find an appropriate ```decay_rate``` for your use case.

**[Note]**  

If you set the time too far in the past, an error might occur during ```decay_rate``` calculations.

```python
# Example usage changing the current time for testing.
with mock_now(datetime.datetime(2025, 1, 7, 00, 00)):
    # Execute a search in this simulated timeline.
    print(retriever.invoke("Langchain Youtube"))
```

<pre class="custom">[Document(metadata={'last_accessed_at': MockDateTime(2025, 1, 7, 0, 0), 'created_at': datetime.datetime(2025, 1, 7, 10, 28, 37, 213151), 'buffer_idx': 1}, page_content='Will you subscribe to Langchain Youtube? Please!')]
</pre>
