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

# Long Context Reorder

- Author: [Minji](https://github.com/r14minji)
- Peer Review: 
- Proofread : [jishin86](https://github.com/jishin86)
- This is a part of [LangChain OpenTutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/04-LongContextReorder.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/04-LongContextReorder.ipynb)

## Overview

Regardless of the model's architecture, performance significantly degrades when including more than 10 retrieved documents.

Simply put, when the model needs to access relevant information in the middle of a long context, it tends to ignore the provided documents.

For more details, please refer to the following paper:

- https://arxiv.org/abs/2307.03172

To avoid this issue, you can prevent performance degradation by reordering documents after retrieval.

Create a retriever that can store and search text data using the Chroma vector store.
Use the retriever's invoke method to search for highly relevant documents for a given query.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Create an instance of the LongContextReorder class named reordering](#create-an-instance-of-the-longcontextreorder-class-named-reordering)
- [Creating Question-Answering Chain with Context Reordering](#creating-question-answering-chain-with-context-reordering)

---


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install langchain-opentutorial
```

```python
# Configuration file for managing API keys as environment variables
from dotenv import load_dotenv

# Load API key information
load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python

from langchain_opentutorial import package

package.install(
    [
       "langsmith",
        "langchain",
        "langchain_openai",
        "langchain_community",
        "langchain-chroma",
    ],
    verbose=False,
    upgrade=False,
)
```

```python
from langchain_opentutorial import set_env

set_env(
    {
        # "OPENAI_API_KEY": "",
        # "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "04-LongContextReorder",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## Create an instance of the LongContextReorder class named reordering.

Enter a query for the retriever to perform the search.

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Get embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [
    "This is just a random text I wrote.",
    "ChatGPT, an AI designed to converse with users, can answer various questions.",
    "iPhone, iPad, MacBook are representative products released by Apple.",
    "ChatGPT was developed by OpenAI and is continuously being improved.",
    "ChatGPT has learned from vast amounts of data to understand user questions and generate appropriate answers.",
    "Wearable devices like Apple Watch and AirPods are also part of Apple's popular product line.",
    "ChatGPT can be used to solve complex problems or suggest creative ideas.",
    "Bitcoin is also called digital gold and is gaining popularity as a store of value.",
    "ChatGPT's capabilities are continuously evolving through ongoing learning and updates.",
    "The FIFA World Cup is held every four years and is the biggest event in international football.",
]



# Create a retriever (Set K to 10)
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
```

```python
query = "What can you tell me about ChatGPT?"

# Retrieves relevant documents sorted by relevance score.
docs = retriever.invoke(query)
docs
```




<pre class="custom">[Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.')]</pre>



Create an instance of LongContextReorder class.

- Call reordering.transform_documents(docs) to reorder the document list.
- Less relevant documents are positioned in the middle of the list, while more relevant documents are positioned at the beginning and end.


```python
# Reorder the documents
# Less relevant documents are positioned in the middle, more relevant elements at start/end
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# Verify that 4 relevant documents are positioned at start and end
reordered_docs
```




<pre class="custom">[Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT, an AI designed to converse with users, can answer various questions.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.'),
     Document(metadata={}, page_content='ChatGPT was developed by OpenAI and is continuously being improved.')]</pre>



## Creating Question-Answering Chain with Context Reordering

A chain that enhances QA (Question-Answering) performance by reordering documents using LongContextReorder, which optimizes the arrangement of context for better comprehension and response accuracy.

```python
def format_docs(docs):
    return "\n".join([doc.page_content for i, doc in enumerate(docs)])
```

```python
print(format_docs(docs))
```

<pre class="custom">ChatGPT was developed by OpenAI and is continuously being improved.
    ChatGPT was developed by OpenAI and is continuously being improved.
    ChatGPT was developed by OpenAI and is continuously being improved.
    ChatGPT was developed by OpenAI and is continuously being improved.
    ChatGPT was developed by OpenAI and is continuously being improved.
    ChatGPT, an AI designed to converse with users, can answer various questions.
    ChatGPT, an AI designed to converse with users, can answer various questions.
    ChatGPT, an AI designed to converse with users, can answer various questions.
    ChatGPT, an AI designed to converse with users, can answer various questions.
    ChatGPT, an AI designed to converse with users, can answer various questions.
</pre>

```python
def format_docs(docs):
    return "\n".join(
        [
            f"[{i}] {doc.page_content} [source: teddylee777@gmail.com]"
            for i, doc in enumerate(docs)
        ]
    )


def reorder_documents(docs):
    # Reorder
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    combined = format_docs(reordered_docs)
    print(combined)
    return combined
```

Prints the reordered documents.

```python
# Define prompt template
_ = reorder_documents(docs)
```

<pre class="custom">[0] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [1] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [2] ChatGPT, an AI designed to converse with users, can answer various questions. [source: teddylee777@gmail.com]
    [3] ChatGPT, an AI designed to converse with users, can answer various questions. [source: teddylee777@gmail.com]
    [4] ChatGPT, an AI designed to converse with users, can answer various questions. [source: teddylee777@gmail.com]
    [5] ChatGPT, an AI designed to converse with users, can answer various questions. [source: teddylee777@gmail.com]
    [6] ChatGPT, an AI designed to converse with users, can answer various questions. [source: teddylee777@gmail.com]
    [7] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [8] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [9] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
</pre>

```python
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Define prompt template
template = """Given this text extracts:
{context}

-----
Please answer the following question:
{question}

Answer in the following languages: {language}
"""

# Define prompt
prompt = ChatPromptTemplate.from_template(template)

# Define Chain
chain = (
    {
        "context": itemgetter("question")
        | retriever
        | RunnableLambda(reorder_documents),  # Search context based on question
        "question": itemgetter("question"),  # Extract question
        "language": itemgetter("language"),  # Extract answer language
    }
    | prompt  # Pass values to prompt template
    | ChatOpenAI(model="gpt-4o-mini")  # Pass prompt to language model
    | StrOutputParser()  # Parse model output as string
)
```


Enter the query in question and language for response.

Check the search results of reordered documents.

```python
answer = chain.invoke(
    {"question": "What can you tell me about ChatGPT?", "language": "English"}
)
```

<pre class="custom">[0] ChatGPT's capabilities are continuously evolving through ongoing learning and updates. [source: teddylee777@gmail.com]
    [1] ChatGPT's capabilities are continuously evolving through ongoing learning and updates. [source: teddylee777@gmail.com]
    [2] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [3] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [4] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [5] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [6] ChatGPT was developed by OpenAI and is continuously being improved. [source: teddylee777@gmail.com]
    [7] ChatGPT's capabilities are continuously evolving through ongoing learning and updates. [source: teddylee777@gmail.com]
    [8] ChatGPT's capabilities are continuously evolving through ongoing learning and updates. [source: teddylee777@gmail.com]
    [9] ChatGPT's capabilities are continuously evolving through ongoing learning and updates. [source: teddylee777@gmail.com]
</pre>

Prints the response.

```python
print(answer)
```

<pre class="custom">ChatGPT is an AI language model developed by OpenAI. Its capabilities are continuously evolving through ongoing learning and updates, which means it is regularly improved to enhance its performance and functionality. The model is designed to understand and generate human-like text, making it useful for a variety of applications such as conversational agents, content creation, and more.
</pre>
