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

# Runnable Parallel

- Author: [Jaemin Hong](https://github.com/geminii01)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial covers `RunnableParallel` , a core component of the LangChain Expression Language(LCEL).

`RunnableParallel` is designed to execute multiple Runnable objects in parallel and return a mapping of their outputs.

This class delivers the same input to each Runnable, making it ideal for running independent tasks concurrently. Moreover, we can instantiate `RunnableParallel` directly or use a dictionary literal within a sequence.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Handling Input and Output](#handling-input-and-output)
- [Using itemgetter as a Shortcut](#using-itemgetter-as-a-shortcut)
- [Understanding Parallel Processing Step-by-Step](#understanding-parallel-processing-step-by-step)
- [Parallel Processing](#parallel-processing)

### References

- [RunalbleParallel](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableParallel.html)
- [itemgetter](https://docs.python.org/3/library/operator.html#operator.itemgetter)
- [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/#setup)
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
        "langchain_community",
        "langchain_core",
        "langchain_openai",
        "faiss-cpu",
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
        "LANGCHAIN_PROJECT": "05-RunnableParallel",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Handling Input and Output

`RunnableParallel` is useful for manipulating the output of one Runnable within a sequence to match the input format requirements of the next Runnable.

Let's suppose a prompt expects input as a map with keys ( `context` , `question` ).

The user input is simply the question, providing content. Therefore, you'll need to use a retriever to get the context and pass the user input under the `question` key.

```python
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Create a FAISS vector store from text
vectorstore = FAISS.from_texts(
    ["Teddy is an AI engineer who loves programming!"], embedding=OpenAIEmbeddings()
)

# Use the vector store as a retriever
retriever = vectorstore.as_retriever()

# Define the template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# Create a chat prompt from the template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Construct the retrieval chain
retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Execute the retrieval chain to obtain an answer to the question
retrieval_chain.invoke("What is Teddy's occupation?")
```




<pre class="custom">"Teddy's occupation is an AI engineer."</pre>



Note that type conversion is handled automatically when configuring `RunnableParallel` with other Runnables. We don't need to manually wrap the dictionary input provided to the `RunnableParallel` class.

The following three methods present different initialization approaches that produce the same result:

```python
# Automatically wrapped into a RunnableParallel
1. {"context": retriever, "question": RunnablePassthrough()}

2. RunnableParallel({"context": retriever, "question": RunnablePassthrough()})

3. RunnableParallel(context=retriever, question=RunnablePassthrough())
```

## Using itemgetter as a Shortcut

Python’s `itemgetter` function offers a shortcut for extracting specific data from a map when it is combined with `RunnableParallel` .

For example, `itemgetter` extracts specific keys from a map.

```python
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Create a FAISS vector store from text
vectorstore = FAISS.from_texts(
    ["Teddy is an AI engineer who loves programming!"], embedding=OpenAIEmbeddings()
)
# Use the vector store as a retriever
retriever = vectorstore.as_retriever()

# Define the template
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""

# Create a chat prompt from the template
prompt = ChatPromptTemplate.from_template(template)

# Construct the chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Invoke the chain to answer the question
chain.invoke({"question": "What is Teddy's occupation?", "language": "English"})
```




<pre class="custom">"Teddy's occupation is an AI engineer."</pre>



## Understanding Parallel Processing Step-by-Step

Using `RunnableParallel` can easily run multiple Runnables in parallel and return a map of their outputs.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the chain for asking about capitals
capital_chain = (
    ChatPromptTemplate.from_template("Where is the capital of the {country}?")
    | model
    | StrOutputParser()
)

# Define the chain for asking about areas
area_chain = (
    ChatPromptTemplate.from_template("What is the area of the {country}?")
    | model
    | StrOutputParser()
)

# Create a RunnableParallel object to execute capital_chain and area_chain in parallel
map_chain = RunnableParallel(capital=capital_chain, area=area_chain)

# Invoke map_chain to ask about both the capital and area
map_chain.invoke({"country": "United States"})
```




<pre class="custom">{'capital': 'The capital of the United States is Washington, D.C.',
     'area': 'The total area of the United States is approximately 3.8 million square miles (about 9.8 million square kilometers). This includes all 50 states and the District of Columbia. If you need more specific details or comparisons, feel free to ask!'}</pre>



The following example explains how to execute chains that have different input template variables.

```python
# Define the chain for asking about capitals
capital_chain2 = (
    ChatPromptTemplate.from_template("Where is the capital of the {country1}?")
    | model
    | StrOutputParser()
)

# Define the chain for asking about areas
area_chain2 = (
    ChatPromptTemplate.from_template("What is the area of the {country2}?")
    | model
    | StrOutputParser()
)

# Create a RunnableParallel object to execute capital_chain2 and area_chain2 in parallel
map_chain2 = RunnableParallel(capital=capital_chain2, area=area_chain2)

# Invoke map_chain with specific values for each key
map_chain2.invoke({"country1": "Republic of Korea", "country2": "United States"})
```




<pre class="custom">{'capital': 'The capital of the Republic of Korea (South Korea) is Seoul.',
     'area': 'The total area of the United States is approximately 3.8 million square miles (about 9.8 million square kilometers). This includes all 50 states and the District of Columbia.'}</pre>



## Parallel Processing

`RunnableParallel` is particularly useful for running independent processes in parallel because each Runnable in the map is executed concurrently.

For example, you can see that `area_chain`, `capital_chain`, and `map_chain` take the almost same execution time, even though `map_chain` runs the other two chains in parallel.

```python
%%timeit

# Invoke the chain for area and measure execution time
area_chain.invoke({"country": "United States"})
```

<pre class="custom">1.49 s ± 208 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
</pre>

```python
%%timeit

# Invoke the chain for capital and measure execution time
capital_chain.invoke({"country": "United States"})
```

<pre class="custom">860 ms ± 195 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
</pre>

```python
%%timeit

# Invoke the chain constructed in parallel and measure execution time
map_chain.invoke({"country": "United States"})
```

<pre class="custom">1.65 s ± 379 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
</pre>
