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

# Runnable-Pass-Through

- Author: [Suhyun Lee](https://github.com/suhyun0115)
- Design: 
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/01-RunnablePassThrough.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/01-RunnablePassThrough.ipynb)

## Overview

`RunnablePassthrough` is a utility that facilitates unmodified data flow through a pipeline. Its `invoke()` method returns input data in its original form without alterations.

This functionality allows seamless data transmission between pipeline stages.

It frequently works in tandem with `RunnableParallel` for concurrent task execution, enabling the addition of new key-value pairs to the data stream.

Common use cases for `RunnablePassthrough` include:

- Direct data forwarding without transformation
- Pipeline stage bypassing
- Pipeline flow validation during debugging

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Passing Data with RunnablePassthrough and RunnableParallel](#passing-data-with-runnablepassthrough-and-runnableparallel)
  - [Example of Using `RunnableParallel` and `RunnablePassthrough`](#example-of-using-runnableparallel-and-runnablepassthrough)
  - [Summary of Results](#summary-of-results)
- [Search Engine Integration](#search-engine-integration)
  - [Using GPT](#using-gpt)
  - [Using Ollama](#using-ollama)
    - [Ollama Installation Guide on Colab](#ollama-installation-guide-on-colab)

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
        "langchain_openai",
        "langchain_core",
        "langchain-ollama",
        "langchain_community",
        "faiss-cpu",
    ],
    verbose=False,
    upgrade=False,
)
```

If you want to get automated tracing of your model calls you can also set your LangSmith API key by uncommenting below code:

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "LangChain-Expression-Language",
    }
)
```

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Passing Data with RunnablePassthrough and RunnableParallel

`RunnablePassthrough` is a utility that **passes data through unchanged** or adds minimal information before forwarding.

It commonly integrates with `RunnableParallel` to map data under new keys.

- **Standalone Usage**
  
  When used independently, `RunnablePassthrough()` returns the input data unmodified.

- **Usage with `assign`**
  
  When implemented with `assign` as `RunnablePassthrough.assign(...)`, it augments the input data with additional fields before forwarding.

By leveraging `RunnablePassthrough`, you can maintain data integrity through pipeline stages while selectively adding required information.

Let me continue reviewing any additional content. I'm tracking all modifications to provide a comprehensive summary once the review is complete.

## Example of Using `RunnableParallel` and `RunnablePassthrough`

While `RunnablePassthrough` is effective independently, it becomes more powerful when combined with `RunnableParallel`.

This section demonstrates how to configure and run **parallel tasks** using the `RunnableParallel` class. The following steps provide a beginner-friendly implementation guide.

---

1. **Initialize `RunnableParallel`**
   
   Create a `RunnableParallel` instance to manage concurrent task execution.

2. **Configure `passed` Task**
   
   - Define a `passed` task utilizing `RunnablePassthrough`
   - This task **preserves input data without modification**

3. **Set Up `extra` Task**
   
   - Implement an `extra` task using `RunnablePassthrough.assign()`
   - This task computes triple the "num" value and stores it with key "mult"

4. **Implement `modified` Task**
   
   - Create a `modified` task using a basic function
   - This function increments the "num" value by 1

5. **Task Execution**
   
   - Invoke all tasks using `runnable.invoke()`
   - Example: Input `{"num": 1}` triggers concurrent execution of all defined tasks


```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    # Sets up a Runnable that returns the input as-is.
    passed=RunnablePassthrough(),
    # Sets up a Runnable that multiplies the "num" value in the input by 3 and returns the result.
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    # Sets up a Runnable that adds 1 to the "num" value in the input and returns the result.
    modified=lambda x: {"num": x["num"] + 1},
)

# Execute the Runnable with {"num": 1} as input.
result = runnable.invoke({"num": 1})

# Print the result.
print(result)
```




<pre class="custom">{'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}</pre>



```python
r = RunnablePassthrough.assign(mult=lambda x: x["num"] * 3)
r.invoke({"num": 1})
```




<pre class="custom">{'num': 1, 'mult': 3}</pre>



## Summary of Results

When provided with input `{"num": 1}`, each task produces the following output:

1. **`passed`:** Returns unmodified input data
   - Output: `{"num": 1}`

2. **`extra`:** Augments input with `"mult"` key containing triple the `"num"` value
   - Output: `{"num": 1, "mult": 3}`

3. **`modified`:** Increments the `"num"` value by 1
   - Output: `{"num": 2}`

## Search Engine Integration

The following example illustrates an implementation of `RunnablePassthrough`.

## Using GPT

```python
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Create a FAISS vector store from text data.
vectorstore = FAISS.from_texts(
    [
        "Cats are geniuses at claiming boxes as their own.",
        "Dogs have successfully trained humans to take them for walks.",
        "Cats aren't fond of water, but the water in a human's cup is an exception.",
        "Dogs follow cats around, eager to befriend them.",
        "Cats consider laser pointers their arch-nemesis.",
    ],
    embedding=OpenAIEmbeddings(),
)

# Use the vector store as a retriever.
retriever = vectorstore.as_retriever()

# Define a template.
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# Create a chat prompt from the template.
prompt = ChatPromptTemplate.from_template(template)
```

```python
# Initialize the ChatOpenAI model.
model = ChatOpenAI(model_name="gpt-4o-mini")


# Function to format retrieved documents.
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


# Construct the retrieval chain.
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

```python
# Query retrieval chain
retrieval_chain.invoke("What kind of objects do cats like?")
```




<pre class="custom">'Cats like boxes.'</pre>



```python
retrieval_chain.invoke("What do dogs like?")
```




<pre class="custom">'Dogs like to befriend cats.'</pre>



## Using Ollama

- Download the application from the [Ollama official website](https://ollama.com/)
- For comprehensive Ollama documentation, visit the [GitHub tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/10-Ollama.ipynb)
- Implementation utilizes the `llama3.2` 1b model for response generation and `mxbai-embed-large` for embedding operations

## Ollama Installation Guide on Colab

Google Colab requires the `colab-xterm` extension for terminal functionality. Follow these steps to install Ollama:

---

1. **Install and Initialize `colab-xterm`**
    ```python
    !pip install colab-xterm
    %load_ext colabxterm
    ```

2. **Launch Terminal**
    ```python
    %xterm
    ```

3. **Install Ollama**

    Execute the following command in the terminal:
    ```python
    curl -fsSL https://ollama.com/install.sh | sh
    ```

4. **Installation Verification**

    Verify installation by running:
    ```python
    ollama
    ```
    Successful installation displays the "Available Commands" menu.

Download and Prepare the Embedding Model for Ollama

```python
!ollama pull mxbai-embed-large
```

```python
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings

# Configure embeddings
ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize FAISS vector store with text data
vectorstore = FAISS.from_texts(
    [
        "Cats are geniuses at claiming boxes as their own.",
        "Dogs have successfully trained humans to take them for walks.",
        "Cats aren't fond of water, but the water in a human's cup is an exception.",
        "Dogs follow cats around, eager to befriend them.",
        "Cats consider laser pointers their arch-nemesis.",
    ],
    embedding=ollama_embeddings(),
)
# Convert vector store to retriever
retriever = vectorstore.as_retriever()

# Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# Initialize chat prompt from template
prompt = ChatPromptTemplate.from_template(template)
```

Download and Prepare the Model for Answer Generation

```python
!ollama pull llama3.2:1b
```

```python
from langchain_ollama import ChatOllama

# Initialize Ollama chat model
ollama_model = ChatOllama(model="llama3.2:1b")


# Format retrieved documents
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


# Build retrieval chain
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ollama_model  # Use Ollama model for inference
    | StrOutputParser()
)
```

```python
# Query retrieval chain
retrieval_chain.invoke("What kind of objects do cats like?")
```




<pre class="custom">'Based on this context, it seems that cats tend to enjoy and claim boxes as their own.'</pre>



```python
# Query retrieval chain
retrieval_chain.invoke("What do dogs like?")
```




<pre class="custom">'Based on the context, it seems that dogs enjoy being around cats and having them follow them. Additionally, dogs have successfully trained humans to take them for walks.'</pre>


