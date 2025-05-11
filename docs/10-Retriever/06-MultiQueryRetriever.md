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

# MultiQueryRetriever

- Author: [hong-seongmin](https://github.com/hong-seongmin)
- Peer Review:
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain OpenTutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/06-MultiQueryRetriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/06-MultiQueryRetriever.ipynb)
## Overview

```MultiQueryRetriever``` offers a thoughtful approach to improving distance-based vector database retrieval by generating diverse queries with the help of an LLM.

This method simplifies the retrieval process, minimizes the need for manual prompt adjustments, and aims to provide more nuanced and comprehensive results.

- **Understanding Distance-Based Vector Search**  
  Distance-based vector search is a technique that identifies documents with embeddings similar to a query embedding based on their 'distance' in a high-dimensional space. However, subtle variations in query details or embedding representations can occasionally make it challenging to fully capture the intended meaning, which might affect the search results.

- **Streamlined Prompt Tuning**  
  ```MultiQueryRetriever``` reduces the complexity of prompt tuning by utilizing an LLM to automatically generate multiple queries from different perspectives for a single input. This helps minimize the effort required for manual adjustments or prompt engineering.

- **Broader Document Retrieval**  
  Each generated query is used to perform a search, and the unique documents retrieved from all queries are combined. This approach helps uncover a wider range of potentially relevant documents, increasing the chances of retrieving valuable information.

- **Improved Search Robustness**  
  By exploring a question from multiple perspectives through diverse queries, ```MultiQueryRetriever``` addresses some of the limitations of distance-based searches. This approach can better account for nuanced differences and deeper meanings in the data, leading to more contextually relevant and well-rounded results.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Building a Vector Database](#building-a-vector-database)
- [Usage](#usage)
- [How to Use the LCEL Chain](#how-to-use-the-lcel-chain)

### References

- [LangChain Documentation: How to use the MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)

---


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
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "06-Multi-Query-Retriever",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

Alternatively, environment variables can also be set using a ```.env``` file.

**[Note]**

- This is not necessary if you've already set the environment variables in the previous step.


```python
# Configuration file to manage API keys as environment variables
from dotenv import load_dotenv

# Load API key information
load_dotenv()
```




<pre class="custom">True</pre>



## Building a Vector Database

Vector databases enable efficient retrieval of relevant documents by embedding text data into a high-dimensional vector space.

This example demonstrates creating a simple vector database using LangChain, which involves loading and splitting a document, generating embeddings with OpenAI, and performing a search query to retrieve contextually relevant information.


```python
# Build a sample vector DB
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load a blog post
loader = WebBaseLoader(
    "https://python.langchain.com/docs/introduction/", encoding="utf-8"
)

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = loader.load_and_split(text_splitter)

# Define embedding
openai_embedding = OpenAIEmbeddings()

# Create the vector DB
db = FAISS.from_documents(docs, openai_embedding)

# Create a retriever
retriever = db.as_retriever()

# Document search
query = "Please explain the key features and architecture of the LangChain framework."
relevant_docs = retriever.invoke(query)

# Print the number of retrieved documents
print(f"Number of retrieved documents: {len(relevant_docs)}")

# Print each document with its number
for idx, doc in enumerate(relevant_docs, start=1):
    print(f"Document #{idx}:\n{doc.page_content}\n{'-'*40}")

```

<pre class="custom">Number of retrieved documents: 4
    Document #1:
    noteThese docs focus on the Python LangChain library. Head here for docs on the JavaScript LangChain library.
    Architecture​
    The LangChain framework consists of multiple open-source libraries. Read more in the
    Architecture page.
    ----------------------------------------
    Document #2:
    LangChain is a framework for developing applications powered by large language models (LLMs).
    LangChain simplifies every stage of the LLM application lifecycle:
    ----------------------------------------
    Document #3:
    However, these guides will help you quickly accomplish common tasks using chat models,
    vector stores, and other common LangChain components.
    Check out LangGraph-specific how-tos here.
    Conceptual guide​
    Introductions to all the key parts of LangChain you’ll need to know! Here you'll find high level explanations of all LangChain concepts.
    For a deeper dive into LangGraph concepts, check out this page.
    Integrations​
    ----------------------------------------
    Document #4:
    langgraph: Orchestration framework for combining LangChain components into production-ready applications with persistence, streaming, and other key features. See LangGraph documentation.
    ----------------------------------------
</pre>

## Usage

Simply specify the LLM to be used in ```MultiQueryRetriever``` and pass the query, and the retriever will handle the rest.


```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI


# Initialize the ChatOpenAI language model with temperature set to 0.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

multiquery_retriever = MultiQueryRetriever.from_llm(  # Initialize the MultiQueryRetriever using the language model.
    # Pass the vector database retriever and the language model.
    retriever=db.as_retriever(),
    llm=llm,
)
```

Below is code that you can run to debug the intermediate process of generating multiple queries.

First, we retrieve the ```langchain.retrievers.multi_query``` logger.

This is done using the ```logging.getLogger``` method. Then, we set the logger's log level to ```INFO```, so that only log messages at the ```INFO``` level or above are printed.


```python
# Logging settings for the query
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```

This code uses the ```invoke``` method of the ```retriever_from_llm``` object to search for documents relevant to the given ```question```.

The retrieved documents are stored in the variable ```relevant_docs```, and checking the length of this variable lets you see how many relevant documents were found.

Through this process, you can effectively locate information related to the user's question and assess how much of it is available.


```python
# Define the question
question = "Please explain the key features and architecture of the LangChain framework."
# Document search
relevant_docs = multiquery_retriever.invoke(question)

# Return the number of unique documents retrieved.
print(
    f"===============\nNumber of retrieved documents: {len(relevant_docs)}",
    end="\n===============\n",
)

# Print the content of the retrieved documents.
print(relevant_docs[0].page_content)
```

<pre class="custom">INFO:langchain.retrievers.multi_query:Generated queries: ['What are the main components and architectural design of the LangChain framework?', 'Can you describe the essential characteristics and structure of the LangChain framework?', 'What are the significant features and the underlying architecture of the LangChain framework?']
</pre>

    ===============
    Number of retrieved documents: 6
    ===============
    noteThese docs focus on the Python LangChain library. Head here for docs on the JavaScript LangChain library.
    Architecture​
    The LangChain framework consists of multiple open-source libraries. Read more in the
    Architecture page.
    

## How to Use the LCEL Chain

- Define a custom prompt, then create a ```Chain``` with that prompt.
- When the ```Chain``` receives a user question (in the following example), it generates 5 questions, and returns the 5 generated questions separated by '\n'.


```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the prompt template (written to generate 5 questions)
prompt = PromptTemplate.from_template(
    """You are an AI language model assistant. 
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Your response should be a list of values separated by new lines, eg: `foo\nbar\nbaz\n`

#ORIGINAL QUESTION: 
{question}

#Answer in English:
"""
)

# Create an instance of the language model.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Create the LLMChain.
custom_multiquery_chain = (
    {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# Define the question.
question = "Please explain the key features and architecture of the LangChain framework."

# Execute the chain and check the generated multiple queries.
multi_queries = custom_multiquery_chain.invoke(question)
# Check the result (5 generated questions)
print(multi_queries)
```

<pre class="custom">What are the main components and structure of the LangChain framework?  
    Can you describe the architecture and essential characteristics of LangChain?  
    What are the significant features and design elements of the LangChain framework?  
    How is the LangChain framework structured, and what are its key functionalities?  
    Could you provide an overview of the LangChain framework's architecture and its primary features?  
</pre>

You can pass the previously created ```Chain``` to the ```MultiQueryRetriever``` to perform retrieval.


```python
multiquery_retriever = MultiQueryRetriever.from_llm(
    llm=custom_multiquery_chain, retriever=db.as_retriever()
)
```

Use the ```MultiQueryRetriever``` to search documents and check the results.


```python
# Result
relevant_docs = multiquery_retriever.invoke(question)

# Return the number of unique documents retrieved.
print(
    f"===============\nNumber of retrieved documents: {len(relevant_docs)}",
    end="\n===============\n",
)

# Print the content of the retrieved documents.
print(relevant_docs[0].page_content)
```

<pre class="custom">INFO:langchain.retrievers.multi_query:Generated queries: ['What are the main characteristics and structure of the LangChain framework?  ', 'Can you describe the essential features and design of the LangChain framework?  ', 'Could you provide an overview of the key components and architecture of the LangChain framework?  ', 'What are the fundamental aspects and architectural elements of the LangChain framework?  ', 'Please outline the primary features and framework architecture of LangChain.']
</pre>

    ===============
    Number of retrieved documents: 5
    ===============
    LangChain is a framework for developing applications powered by large language models (LLMs).
    LangChain simplifies every stage of the LLM application lifecycle:
    
