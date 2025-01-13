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

# LangChain-Expression-Language

- Author: [Suhyun Lee](https://github.com/suhyun0115)
- Design: 
- Peer Review:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

`RunnablePassthrough` is a tool that **passes data through unchanged** or adds minimal information to it before forwarding. The `invoke()` method of this class **returns the input data without any modifications**.

This enables data to flow to the next stage without being altered.

It is commonly used in conjunction with `RunnableParallel`, which handles multiple tasks simultaneously, and it helps attach new **labels (keys)** to the data.

`RunnablePassthrough` is useful in scenarios such as:

- When there’s no need to transform or modify the data.
- To skip specific stages in a pipeline.
- For debugging or testing, to verify smooth data flow.

In this tutorial, we will implement this using the GPT-4o-mini model and Ollama, based on the LLaMA 3.2 1B model.

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
!pip install langchain-opentutorial
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

`RunnablePassthrough` is a tool that **passes data through unchanged** or adds minimal information to it before forwarding.

It is often used with `RunnableParallel` to store data under a new name.

- **Using it alone**
  
  When used on its own, `RunnablePassthrough()` returns the input data as is.

- **Using with `assign`**
  
  When used with `assign` like `RunnablePassthrough.assign(...)`, it adds additional information to the input data before passing it on.

By using `RunnablePassthrough`, you can pass data to the next stage unchanged while adding only the necessary information.

### Example of Using `RunnableParallel` and `RunnablePassthrough`

While `RunnablePassthrough` is useful on its own, it becomes even more powerful when used in combination with `RunnableParallel`.

In this section, we’ll learn how to define and execute **multiple tasks simultaneously** using the `RunnableParallel` class. The step-by-step guide ensures that even beginners can follow along easily.

---

1. **Create a `RunnableParallel` Instance**
   
   First, create an object using the `RunnableParallel` class to execute multiple tasks simultaneously.

2. **Add a `passed` Task**
   
   - Add a task named `passed` that uses `RunnablePassthrough`.
   - This task **returns the input data unchanged**.

3. **Add an `extra` Task**
   
   - Add a task named `extra` that uses `RunnablePassthrough.assign()`.
   - This task multiplies the "num" value in the input data by 3 and stores it under a new key named "mult".

4. **Add a `modified` Task**
   
   - Add a task named `modified` that uses a simple function.
   - This function adds 1 to the "num" value in the input data.

5. **Execute the Tasks**
   
   - After setting up all the tasks, call `runnable.invoke()`.
   - For example, if you input `{"num": 1}`, all the tasks you defined will execute simultaneously.

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



### Summary of Results

When the input data is set to `{"num": 1}`, the results of each task are as follows:

1. **`passed`:** Returns the input data unchanged.
   - Result: `{"num": 1}`

2. **`extra`:** Adds a `"mult"` key to the input data, with its value being the `"num"` value multiplied by 3.
   - Result: `{"num": 1, "mult": 3}`

3. **`modified`:** Adds 1 to the `"num"` value.
   - Result: `{"num": 2}`

## Search Engine Integration

The example below demonstrates a use case where `RunnablePassthrough` is utilized.

### Using GPT

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
# Execute the retrieval chain to get an answer to a question.
retrieval_chain.invoke("What kind of objects do cats like?")
```




<pre class="custom">'Cats like boxes.'</pre>



```python
# Execute the retrieval chain to get an answer to a question.
retrieval_chain.invoke("What do dogs like?")
```




<pre class="custom">'Dogs like to befriend cats.'</pre>



### Using Ollama

- Install the program from the [Ollama official website](https://ollama.com/).
- For detailed information about Ollama, refer to the [GitHub tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/10-Ollama.ipynb).
- The `llama3.2` 1b model is used for generating responses, while `mxbai-embed-large` is used for embedding tasks.


### Ollama Installation Guide on Colab

Google Colab does not natively support terminal access, but you can enable it using the `colab-xterm` extension. Below is a step-by-step guide for installing Ollama on Colab.

---

1. **Install and Load `colab-xterm`**
    ```python
   !pip install colab-xterm
   %load_ext colabxterm

2. **Open the Terminal**
    %xterm

3. **Install Ollama**

    In the terminal window that opens, run the following command to install Ollama:
    ```python
    curl -fsSL https://ollama.com/install.sh | sh

4. **Verify Installation** 


   After installation, type ollama in the terminal to check the installation status. If installed correctly, you should see the "Available Commands" list.
    ```python
    ollama

Download and Prepare the Embedding Model for Ollama

```python
!ollama pull mxbai-embed-large
```

<pre class="custom">[?25lpulling manifest ⠋ [?25h[?25l[2K[1Gpulling manifest ⠙ [?25h[?25l[2K[1Gpulling manifest ⠹ [?25h[?25l[2K[1Gpulling manifest ⠸ [?25h[?25l[2K[1Gpulling manifest ⠼ [?25h[?25l[2K[1Gpulling manifest ⠴ [?25h[?25l[2K[1Gpulling manifest ⠦ [?25h[?25l[2K[1Gpulling manifest ⠧ [?25h[?25l[2K[1Gpulling manifest ⠇ [?25h[?25l[2K[1Gpulling manifest ⠏ [?25h[?25l[2K[1Gpulling manifest ⠋ [?25h[?25l[2K[1Gpulling manifest ⠙ [?25h[?25l[2K[1Gpulling manifest ⠹ [?25h[?25l[2K[1Gpulling manifest ⠸ [?25h[?25l[2K[1Gpulling manifest ⠼ [?25h[?25l[2K[1Gpulling manifest ⠴ [?25h[?25l[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   0% ▕                ▏    0 B/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   0% ▕                ▏    0 B/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   0% ▕                ▏    0 B/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   0% ▕                ▏    0 B/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   0% ▕                ▏    0 B/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   1% ▕                ▏ 4.0 MB/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   1% ▕                ▏ 7.2 MB/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   2% ▕                ▏  13 MB/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   3% ▕                ▏  19 MB/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   3% ▕                ▏  22 MB/669 MB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   4% ▕                ▏  28 MB/669 MB   28 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   5% ▕                ▏  33 MB/669 MB   28 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   5% ▕                ▏  36 MB/669 MB   28 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   6% ▕                ▏  41 MB/669 MB   28 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   7% ▕█               ▏  47 MB/669 MB   28 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   8% ▕█               ▏  50 MB/669 MB   28 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   8% ▕█               ▏  56 MB/669 MB   28 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...   9% ▕█               ▏  62 MB/669 MB   28 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  10% ▕█               ▏  66 MB/669 MB   28 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  11% ▕█               ▏  72 MB/669 MB   28 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  12% ▕█               ▏  79 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  12% ▕█               ▏  82 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  13% ▕██              ▏  88 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  14% ▕██              ▏  95 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  15% ▕██              ▏  98 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  16% ▕██              ▏ 104 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  17% ▕██              ▏ 111 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  17% ▕██              ▏ 115 MB/669 MB   39 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  18% ▕██              ▏ 121 MB/669 MB   39 MB/s     13s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  19% ▕███             ▏ 128 MB/669 MB   39 MB/s     13s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  20% ▕███             ▏ 132 MB/669 MB   39 MB/s     13s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  21% ▕███             ▏ 139 MB/669 MB   45 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  22% ▕███             ▏ 146 MB/669 MB   45 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  22% ▕███             ▏ 148 MB/669 MB   45 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  23% ▕███             ▏ 154 MB/669 MB   45 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  24% ▕███             ▏ 160 MB/669 MB   45 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  24% ▕███             ▏ 163 MB/669 MB   45 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  25% ▕████            ▏ 169 MB/669 MB   45 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  26% ▕████            ▏ 175 MB/669 MB   45 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  27% ▕████            ▏ 178 MB/669 MB   45 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  28% ▕████            ▏ 184 MB/669 MB   45 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  28% ▕████            ▏ 190 MB/669 MB   46 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  29% ▕████            ▏ 192 MB/669 MB   46 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  30% ▕████            ▏ 199 MB/669 MB   46 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  31% ▕████            ▏ 205 MB/669 MB   46 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  31% ▕████            ▏ 208 MB/669 MB   46 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  32% ▕█████           ▏ 214 MB/669 MB   46 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  33% ▕█████           ▏ 220 MB/669 MB   46 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  33% ▕█████           ▏ 223 MB/669 MB   46 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  34% ▕█████           ▏ 229 MB/669 MB   46 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  35% ▕█████           ▏ 235 MB/669 MB   46 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  36% ▕█████           ▏ 238 MB/669 MB   47 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  37% ▕█████           ▏ 244 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  37% ▕█████           ▏ 250 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  38% ▕██████          ▏ 253 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  39% ▕██████          ▏ 259 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  40% ▕██████          ▏ 265 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  40% ▕██████          ▏ 268 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  41% ▕██████          ▏ 274 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  42% ▕██████          ▏ 280 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  42% ▕██████          ▏ 283 MB/669 MB   47 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  43% ▕██████          ▏ 289 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  44% ▕███████         ▏ 295 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  45% ▕███████         ▏ 298 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  45% ▕███████         ▏ 304 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  46% ▕███████         ▏ 309 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  47% ▕███████         ▏ 313 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  48% ▕███████         ▏ 318 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  48% ▕███████         ▏ 324 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  49% ▕███████         ▏ 327 MB/669 MB   48 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  50% ▕███████         ▏ 333 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  51% ▕████████        ▏ 339 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  51% ▕████████        ▏ 342 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  52% ▕████████        ▏ 348 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  53% ▕████████        ▏ 355 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  53% ▕████████        ▏ 357 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  54% ▕████████        ▏ 363 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  55% ▕████████        ▏ 369 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  56% ▕████████        ▏ 372 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  56% ▕█████████       ▏ 377 MB/669 MB   48 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  57% ▕█████████       ▏ 384 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  58% ▕█████████       ▏ 387 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  59% ▕█████████       ▏ 393 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  60% ▕█████████       ▏ 399 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  60% ▕█████████       ▏ 402 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  61% ▕█████████       ▏ 408 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  62% ▕█████████       ▏ 414 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  62% ▕█████████       ▏ 417 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  63% ▕██████████      ▏ 423 MB/669 MB   48 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  64% ▕██████████      ▏ 429 MB/669 MB   48 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  65% ▕██████████      ▏ 432 MB/669 MB   48 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  65% ▕██████████      ▏ 438 MB/669 MB   48 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  66% ▕██████████      ▏ 444 MB/669 MB   49 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  67% ▕██████████      ▏ 447 MB/669 MB   49 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  68% ▕██████████      ▏ 453 MB/669 MB   49 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  69% ▕██████████      ▏ 459 MB/669 MB   49 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  69% ▕███████████     ▏ 462 MB/669 MB   49 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  70% ▕███████████     ▏ 468 MB/669 MB   49 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  71% ▕███████████     ▏ 474 MB/669 MB   49 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  71% ▕███████████     ▏ 476 MB/669 MB   49 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  72% ▕███████████     ▏ 483 MB/669 MB   49 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  73% ▕███████████     ▏ 489 MB/669 MB   49 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  74% ▕███████████     ▏ 492 MB/669 MB   51 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  74% ▕███████████     ▏ 494 MB/669 MB   51 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  74% ▕███████████     ▏ 494 MB/669 MB   51 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  74% ▕███████████     ▏ 497 MB/669 MB   51 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  75% ▕████████████    ▏ 504 MB/669 MB   51 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  76% ▕████████████    ▏ 511 MB/669 MB   51 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  77% ▕████████████    ▏ 515 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  78% ▕████████████    ▏ 522 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  79% ▕████████████    ▏ 529 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  80% ▕████████████    ▏ 533 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  81% ▕████████████    ▏ 539 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  82% ▕█████████████   ▏ 546 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  82% ▕█████████████   ▏ 550 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  83% ▕█████████████   ▏ 557 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  84% ▕█████████████   ▏ 563 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  85% ▕█████████████   ▏ 566 MB/669 MB   51 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  86% ▕█████████████   ▏ 572 MB/669 MB   51 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  86% ▕█████████████   ▏ 578 MB/669 MB   51 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  87% ▕█████████████   ▏ 581 MB/669 MB   51 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  88% ▕██████████████  ▏ 587 MB/669 MB   51 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  89% ▕██████████████  ▏ 593 MB/669 MB   50 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  89% ▕██████████████  ▏ 596 MB/669 MB   50 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  90% ▕██████████████  ▏ 602 MB/669 MB   50 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  91% ▕██████████████  ▏ 608 MB/669 MB   50 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  91% ▕██████████████  ▏ 610 MB/669 MB   50 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  92% ▕██████████████  ▏ 617 MB/669 MB   50 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  93% ▕██████████████  ▏ 623 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  94% ▕██████████████  ▏ 626 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  94% ▕███████████████ ▏ 632 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  95% ▕███████████████ ▏ 638 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  96% ▕███████████████ ▏ 641 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  97% ▕███████████████ ▏ 647 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  97% ▕███████████████ ▏ 650 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  98% ▕███████████████ ▏ 653 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6...  99% ▕███████████████ ▏ 660 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕███████████████ ▏ 667 MB/669 MB   50 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917...   0% ▕                ▏    0 B/ 11 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917...   0% ▕                ▏    0 B/ 11 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917...   0% ▕                ▏    0 B/ 11 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917...   0% ▕                ▏    0 B/ 11 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917...   0% ▕                ▏    0 B/ 11 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917...   0% ▕                ▏    0 B/ 11 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855...   0% ▕                ▏    0 B/  16 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855...   0% ▕                ▏    0 B/  16 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855...   0% ▕                ▏    0 B/  16 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855...   0% ▕                ▏    0 B/  16 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91...   0% ▕                ▏    0 B/ 408 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91...   0% ▕                ▏    0 B/ 408 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91...   0% ▕                ▏    0 B/ 408 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91...   0% ▕                ▏    0 B/ 408 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠋ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠙ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠹ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠸ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠼ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠴ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠦ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠧ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠇ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠏ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest ⠋ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 819c2adf5ce6... 100% ▕████████████████▏ 669 MB                         
    pulling c71d239df917... 100% ▕████████████████▏  11 KB                         
    pulling b837481ff855... 100% ▕████████████████▏   16 B                         
    pulling 38badd946f91... 100% ▕████████████████▏  408 B                         
    verifying sha256 digest 
    writing manifest 
    success [?25h
</pre>

```python
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings

# Embedding Configuration
ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Create a FAISS vector store from text data.
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

Download and Prepare the Model for Answer Generation

```python
!ollama pull llama3.2:1b
```

<pre class="custom">[?25lpulling manifest ⠋ [?25h[?25l[2K[1Gpulling manifest ⠙ [?25h[?25l[2K[1Gpulling manifest ⠹ [?25h[?25l[2K[1Gpulling manifest ⠸ [?25h[?25l[2K[1Gpulling manifest ⠼ [?25h[?25l[2K[1Gpulling manifest ⠴ [?25h[?25l[2K[1Gpulling manifest ⠦ [?25h[?25l[2K[1Gpulling manifest ⠧ [?25h[?25l[2K[1Gpulling manifest ⠇ [?25h[?25l[2K[1Gpulling manifest ⠏ [?25h[?25l[2K[1Gpulling manifest ⠋ [?25h[?25l[2K[1Gpulling manifest ⠙ [?25h[?25l[2K[1Gpulling manifest ⠹ [?25h[?25l[2K[1Gpulling manifest ⠸ [?25h[?25l[2K[1Gpulling manifest ⠼ [?25h[?25l[2K[1Gpulling manifest ⠴ [?25h[?25l[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏    0 B/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏    0 B/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏    0 B/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏    0 B/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏    0 B/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏    0 B/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏    0 B/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   0% ▕                ▏ 222 KB/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   1% ▕                ▏ 6.8 MB/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   1% ▕                ▏  12 MB/1.3 GB                  [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   1% ▕                ▏  16 MB/1.3 GB   16 MB/s   1m20s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   2% ▕                ▏  22 MB/1.3 GB   16 MB/s   1m20s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   2% ▕                ▏  28 MB/1.3 GB   16 MB/s   1m20s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   2% ▕                ▏  30 MB/1.3 GB   16 MB/s   1m20s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   3% ▕                ▏  37 MB/1.3 GB   16 MB/s   1m19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   3% ▕                ▏  44 MB/1.3 GB   16 MB/s   1m19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   4% ▕                ▏  47 MB/1.3 GB   16 MB/s   1m18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   4% ▕                ▏  54 MB/1.3 GB   16 MB/s   1m18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   5% ▕                ▏  60 MB/1.3 GB   16 MB/s   1m18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   5% ▕                ▏  63 MB/1.3 GB   16 MB/s   1m18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   5% ▕                ▏  71 MB/1.3 GB   35 MB/s     35s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   6% ▕                ▏  77 MB/1.3 GB   35 MB/s     34s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   6% ▕                ▏  80 MB/1.3 GB   35 MB/s     34s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   7% ▕█               ▏  87 MB/1.3 GB   35 MB/s     34s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   7% ▕█               ▏  94 MB/1.3 GB   35 MB/s     34s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   7% ▕█               ▏  96 MB/1.3 GB   35 MB/s     34s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   8% ▕█               ▏ 104 MB/1.3 GB   35 MB/s     34s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   8% ▕█               ▏ 110 MB/1.3 GB   35 MB/s     34s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   9% ▕█               ▏ 112 MB/1.3 GB   35 MB/s     33s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   9% ▕█               ▏ 118 MB/1.3 GB   35 MB/s     33s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...   9% ▕█               ▏ 124 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  10% ▕█               ▏ 127 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  10% ▕█               ▏ 132 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  10% ▕█               ▏ 138 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  11% ▕█               ▏ 142 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  11% ▕█               ▏ 147 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  11% ▕█               ▏ 151 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  12% ▕█               ▏ 154 MB/1.3 GB   41 MB/s     28s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  12% ▕█               ▏ 160 MB/1.3 GB   41 MB/s     27s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  13% ▕██              ▏ 166 MB/1.3 GB   41 MB/s     27s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  13% ▕██              ▏ 169 MB/1.3 GB   41 MB/s     27s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  13% ▕██              ▏ 173 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  14% ▕██              ▏ 178 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  14% ▕██              ▏ 182 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  14% ▕██              ▏ 187 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  15% ▕██              ▏ 192 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  15% ▕██              ▏ 195 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  15% ▕██              ▏ 199 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  15% ▕██              ▏ 203 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  16% ▕██              ▏ 205 MB/1.3 GB   42 MB/s     26s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  16% ▕██              ▏ 209 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  16% ▕██              ▏ 215 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  16% ▕██              ▏ 217 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  17% ▕██              ▏ 222 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  17% ▕██              ▏ 227 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  18% ▕██              ▏ 231 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  18% ▕██              ▏ 237 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  18% ▕██              ▏ 244 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  19% ▕██              ▏ 247 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  19% ▕███             ▏ 253 MB/1.3 GB   42 MB/s     25s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  20% ▕███             ▏ 259 MB/1.3 GB   42 MB/s     24s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  20% ▕███             ▏ 263 MB/1.3 GB   43 MB/s     24s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  20% ▕███             ▏ 269 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  21% ▕███             ▏ 273 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  21% ▕███             ▏ 276 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  21% ▕███             ▏ 280 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  21% ▕███             ▏ 283 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  22% ▕███             ▏ 286 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  22% ▕███             ▏ 290 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  22% ▕███             ▏ 296 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  23% ▕███             ▏ 300 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  23% ▕███             ▏ 303 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  23% ▕███             ▏ 307 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  24% ▕███             ▏ 310 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  24% ▕███             ▏ 314 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  24% ▕███             ▏ 320 MB/1.3 GB   43 MB/s     23s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  24% ▕███             ▏ 323 MB/1.3 GB   43 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  25% ▕███             ▏ 326 MB/1.3 GB   43 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  25% ▕████            ▏ 330 MB/1.3 GB   43 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  25% ▕████            ▏ 333 MB/1.3 GB   43 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  26% ▕████            ▏ 337 MB/1.3 GB   43 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  26% ▕████            ▏ 343 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  26% ▕████            ▏ 346 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  26% ▕████            ▏ 349 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  27% ▕████            ▏ 354 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  27% ▕████            ▏ 356 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  27% ▕████            ▏ 360 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  28% ▕████            ▏ 366 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  28% ▕████            ▏ 370 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  28% ▕████            ▏ 374 MB/1.3 GB   42 MB/s     22s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  29% ▕████            ▏ 380 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  29% ▕████            ▏ 380 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  29% ▕████            ▏ 386 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  30% ▕████            ▏ 391 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  30% ▕████            ▏ 395 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  30% ▕████            ▏ 399 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  31% ▕████            ▏ 405 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  31% ▕████            ▏ 406 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  31% ▕████            ▏ 412 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  32% ▕█████           ▏ 416 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  32% ▕█████           ▏ 420 MB/1.3 GB   42 MB/s     21s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  32% ▕█████           ▏ 425 MB/1.3 GB   42 MB/s     20s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  33% ▕█████           ▏ 429 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  33% ▕█████           ▏ 430 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  33% ▕█████           ▏ 432 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  33% ▕█████           ▏ 435 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  33% ▕█████           ▏ 438 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  34% ▕█████           ▏ 443 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  34% ▕█████           ▏ 448 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  34% ▕█████           ▏ 450 MB/1.3 GB   45 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  34% ▕█████           ▏ 454 MB/1.3 GB   45 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  35% ▕█████           ▏ 457 MB/1.3 GB   45 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  35% ▕█████           ▏ 460 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  35% ▕█████           ▏ 466 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  36% ▕█████           ▏ 470 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  36% ▕█████           ▏ 472 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  36% ▕█████           ▏ 477 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  36% ▕█████           ▏ 480 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  37% ▕█████           ▏ 483 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  37% ▕█████           ▏ 490 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  37% ▕█████           ▏ 493 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  38% ▕██████          ▏ 496 MB/1.3 GB   43 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  38% ▕██████          ▏ 500 MB/1.3 GB   41 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  38% ▕██████          ▏ 504 MB/1.3 GB   41 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  38% ▕██████          ▏ 507 MB/1.3 GB   41 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  39% ▕██████          ▏ 513 MB/1.3 GB   41 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  39% ▕██████          ▏ 517 MB/1.3 GB   41 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  39% ▕██████          ▏ 520 MB/1.3 GB   41 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  40% ▕██████          ▏ 523 MB/1.3 GB   41 MB/s     19s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  40% ▕██████          ▏ 529 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  40% ▕██████          ▏ 532 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  41% ▕██████          ▏ 539 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  41% ▕██████          ▏ 546 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  42% ▕██████          ▏ 549 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  42% ▕██████          ▏ 555 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  43% ▕██████          ▏ 562 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  43% ▕██████          ▏ 565 MB/1.3 GB   41 MB/s     18s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  43% ▕██████          ▏ 572 MB/1.3 GB   41 MB/s     17s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  44% ▕███████         ▏ 580 MB/1.3 GB   41 MB/s     17s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  44% ▕███████         ▏ 583 MB/1.3 GB   41 MB/s     17s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  45% ▕███████         ▏ 590 MB/1.3 GB   41 MB/s     17s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  45% ▕███████         ▏ 597 MB/1.3 GB   41 MB/s     17s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  45% ▕███████         ▏ 601 MB/1.3 GB   41 MB/s     17s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  46% ▕███████         ▏ 606 MB/1.3 GB   43 MB/s     16s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  46% ▕███████         ▏ 612 MB/1.3 GB   43 MB/s     16s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  47% ▕███████         ▏ 615 MB/1.3 GB   43 MB/s     16s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  47% ▕███████         ▏ 622 MB/1.3 GB   43 MB/s     16s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  48% ▕███████         ▏ 629 MB/1.3 GB   43 MB/s     15s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  48% ▕███████         ▏ 632 MB/1.3 GB   43 MB/s     15s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  48% ▕███████         ▏ 639 MB/1.3 GB   43 MB/s     15s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  49% ▕███████         ▏ 646 MB/1.3 GB   43 MB/s     15s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  49% ▕███████         ▏ 649 MB/1.3 GB   43 MB/s     15s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  50% ▕███████         ▏ 657 MB/1.3 GB   43 MB/s     15s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  50% ▕████████        ▏ 664 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  51% ▕████████        ▏ 667 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  51% ▕████████        ▏ 674 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  52% ▕████████        ▏ 680 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  52% ▕████████        ▏ 684 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  52% ▕████████        ▏ 691 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  53% ▕████████        ▏ 698 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  53% ▕████████        ▏ 701 MB/1.3 GB   44 MB/s     14s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  54% ▕████████        ▏ 708 MB/1.3 GB   44 MB/s     13s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  54% ▕████████        ▏ 715 MB/1.3 GB   44 MB/s     13s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  54% ▕████████        ▏ 718 MB/1.3 GB   46 MB/s     13s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  55% ▕████████        ▏ 726 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  56% ▕████████        ▏ 733 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  56% ▕████████        ▏ 736 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  56% ▕█████████       ▏ 743 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  57% ▕█████████       ▏ 750 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  57% ▕█████████       ▏ 753 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  57% ▕█████████       ▏ 757 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  58% ▕█████████       ▏ 762 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  58% ▕█████████       ▏ 764 MB/1.3 GB   46 MB/s     12s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  58% ▕█████████       ▏ 769 MB/1.3 GB   47 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  59% ▕█████████       ▏ 774 MB/1.3 GB   47 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  59% ▕█████████       ▏ 777 MB/1.3 GB   47 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  59% ▕█████████       ▏ 783 MB/1.3 GB   47 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  60% ▕█████████       ▏ 790 MB/1.3 GB   47 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  60% ▕█████████       ▏ 793 MB/1.3 GB   47 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  61% ▕█████████       ▏ 799 MB/1.3 GB   47 MB/s     11s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  61% ▕█████████       ▏ 805 MB/1.3 GB   47 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  61% ▕█████████       ▏ 809 MB/1.3 GB   47 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  62% ▕█████████       ▏ 816 MB/1.3 GB   47 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  62% ▕█████████       ▏ 823 MB/1.3 GB   48 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  63% ▕██████████      ▏ 826 MB/1.3 GB   48 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  63% ▕██████████      ▏ 833 MB/1.3 GB   48 MB/s     10s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  64% ▕██████████      ▏ 840 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  64% ▕██████████      ▏ 844 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  64% ▕██████████      ▏ 851 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  65% ▕██████████      ▏ 858 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  65% ▕██████████      ▏ 861 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  66% ▕██████████      ▏ 867 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  66% ▕██████████      ▏ 874 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  66% ▕██████████      ▏ 878 MB/1.3 GB   48 MB/s      9s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  67% ▕██████████      ▏ 884 MB/1.3 GB   50 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  67% ▕██████████      ▏ 891 MB/1.3 GB   50 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  68% ▕██████████      ▏ 895 MB/1.3 GB   50 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  68% ▕██████████      ▏ 901 MB/1.3 GB   50 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  69% ▕███████████     ▏ 908 MB/1.3 GB   50 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  69% ▕███████████     ▏ 911 MB/1.3 GB   50 MB/s      8s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  70% ▕███████████     ▏ 918 MB/1.3 GB   50 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  70% ▕███████████     ▏ 925 MB/1.3 GB   50 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  70% ▕███████████     ▏ 928 MB/1.3 GB   50 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  71% ▕███████████     ▏ 934 MB/1.3 GB   50 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  71% ▕███████████     ▏ 939 MB/1.3 GB   52 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  71% ▕███████████     ▏ 941 MB/1.3 GB   52 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  71% ▕███████████     ▏ 944 MB/1.3 GB   52 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  72% ▕███████████     ▏ 950 MB/1.3 GB   52 MB/s      7s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  72% ▕███████████     ▏ 953 MB/1.3 GB   52 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  73% ▕███████████     ▏ 959 MB/1.3 GB   52 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  73% ▕███████████     ▏ 966 MB/1.3 GB   52 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  73% ▕███████████     ▏ 969 MB/1.3 GB   52 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  74% ▕███████████     ▏ 976 MB/1.3 GB   52 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  74% ▕███████████     ▏ 982 MB/1.3 GB   52 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  75% ▕███████████     ▏ 985 MB/1.3 GB   53 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  75% ▕████████████    ▏ 991 MB/1.3 GB   53 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  75% ▕████████████    ▏ 996 MB/1.3 GB   53 MB/s      6s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  76% ▕████████████    ▏ 1.0 GB/1.3 GB   53 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  76% ▕████████████    ▏ 1.0 GB/1.3 GB   53 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  77% ▕████████████    ▏ 1.0 GB/1.3 GB   53 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  77% ▕████████████    ▏ 1.0 GB/1.3 GB   53 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  77% ▕████████████    ▏ 1.0 GB/1.3 GB   53 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  78% ▕████████████    ▏ 1.0 GB/1.3 GB   53 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  78% ▕████████████    ▏ 1.0 GB/1.3 GB   53 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  79% ▕████████████    ▏ 1.0 GB/1.3 GB   54 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  79% ▕████████████    ▏ 1.0 GB/1.3 GB   54 MB/s      5s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  79% ▕████████████    ▏ 1.0 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  80% ▕████████████    ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  80% ▕████████████    ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  80% ▕████████████    ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  81% ▕████████████    ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  81% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  82% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  82% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  82% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  83% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  83% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  84% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      4s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  84% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  84% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  85% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  85% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  85% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  86% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  86% ▕█████████████   ▏ 1.1 GB/1.3 GB   54 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  87% ▕█████████████   ▏ 1.1 GB/1.3 GB   53 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  87% ▕█████████████   ▏ 1.1 GB/1.3 GB   53 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  87% ▕█████████████   ▏ 1.2 GB/1.3 GB   53 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  88% ▕██████████████  ▏ 1.2 GB/1.3 GB   53 MB/s      3s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  88% ▕██████████████  ▏ 1.2 GB/1.3 GB   53 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  88% ▕██████████████  ▏ 1.2 GB/1.3 GB   53 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  89% ▕██████████████  ▏ 1.2 GB/1.3 GB   53 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  89% ▕██████████████  ▏ 1.2 GB/1.3 GB   53 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  89% ▕██████████████  ▏ 1.2 GB/1.3 GB   53 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  90% ▕██████████████  ▏ 1.2 GB/1.3 GB   53 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  90% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  91% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  91% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  92% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  92% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      2s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  92% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  93% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  93% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  93% ▕██████████████  ▏ 1.2 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  94% ▕███████████████ ▏ 1.2 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  94% ▕███████████████ ▏ 1.2 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  94% ▕███████████████ ▏ 1.2 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  95% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  95% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  96% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      1s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  96% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  96% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  97% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  97% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  97% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  98% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  98% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   52 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6...  99% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   47 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   41 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   37 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   37 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕███████████████ ▏ 1.3 GB/1.3 GB   37 MB/s      0s[?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         [?25h[?25l[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6...   0% ▕                ▏    0 B/1.4 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6...   0% ▕                ▏    0 B/1.4 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6...   0% ▕                ▏    0 B/1.4 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6...   0% ▕                ▏    0 B/1.4 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6...   0% ▕                ▏    0 B/1.4 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da...   0% ▕                ▏    0 B/7.7 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da...   0% ▕                ▏    0 B/7.7 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da...   0% ▕                ▏    0 B/7.7 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da...   0% ▕                ▏    0 B/7.7 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9...   0% ▕                ▏    0 B/6.0 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9...   0% ▕                ▏    0 B/6.0 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9...   0% ▕                ▏    0 B/6.0 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9...   0% ▕                ▏    0 B/6.0 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9...   0% ▕                ▏    0 B/6.0 KB                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7...   0% ▕                ▏    0 B/ 485 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7...   0% ▕                ▏    0 B/ 485 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7...   0% ▕                ▏    0 B/ 485 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7...   0% ▕                ▏    0 B/ 485 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7...   0% ▕                ▏    0 B/ 485 B                  [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠋ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠙ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠹ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠸ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠼ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠴ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠦ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠧ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠇ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠏ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠋ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠙ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠹ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠸ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠼ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠴ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠦ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠧ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠇ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠏ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠋ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠙ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest ⠹ [?25h[?25l[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1G[A[2K[1Gpulling manifest 
    pulling 74701a8c35f6... 100% ▕████████████████▏ 1.3 GB                         
    pulling 966de95ca8a6... 100% ▕████████████████▏ 1.4 KB                         
    pulling fcc5a6bec9da... 100% ▕████████████████▏ 7.7 KB                         
    pulling a70ff7e570d9... 100% ▕████████████████▏ 6.0 KB                         
    pulling 4f659a1e86d7... 100% ▕████████████████▏  485 B                         
    verifying sha256 digest 
    writing manifest 
    success [?25h
</pre>

```python
from langchain_ollama import ChatOllama

ollama_model = ChatOllama(model="llama3.2:1b")


# Function to format retrieved documents.
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


# Construct the retrieval chain.
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ollama_model  # Switch to the Ollama model
    | StrOutputParser()
)
```

```python
# Execute the retrieval chain to get an answer to a question.
retrieval_chain.invoke("What kind of objects do cats like?")
```




<pre class="custom">'Based on this context, it seems that cats tend to enjoy and claim boxes as their own.'</pre>



```python
# Execute the retrieval chain to get an answer to a question.
retrieval_chain.invoke("What do dogs like?")
```




<pre class="custom">'Based on the context, it seems that dogs enjoy being around cats and having them follow them. Additionally, dogs have successfully trained humans to take them for walks.'</pre>


