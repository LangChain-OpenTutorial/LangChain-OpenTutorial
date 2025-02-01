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

# Generator

- Author: [Junseong Kim](https://www.linkedin.com/in/%EC%A4%80%EC%84%B1-%EA%B9%80-591b351b2/)
- Design: [Junseong Kim](https://www.linkedin.com/in/%EC%A4%80%EC%84%B1-%EA%B9%80-591b351b2/)
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/02-CommaSeparatedListOutputParser.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/02-CommaSeparatedListOutputParser.ipynb)

## Overview

This tutorial demonstrates how to use a **user-defined generator** (or asynchronous generator) within a LangChain pipeline to process text outputs in a streaming manner. Specifically, we’ll show how to parse a comma-separated string output into a Python list, leveraging the benefits of streaming from a language model. We will also cover asynchronous usage, showing how to adopt the same approach with async generators.

By the end of this tutorial, you’ll be able to:
- Implement a custom generator function that can handle streaming outputs.
- Parse comma-separated text chunks into a list in real time.
- Use both synchronous and asynchronous approaches for streaming data.
- Integrate these parsers into a LangChain chain.
- Optionally, explore how `RunnableGenerator` can be used to implement custom generator transformations within a streaming context

### Table of Contents

- [Overview](#overview)  
- [Environment Setup](#environment-setup)  
- [Implementing a Comma-Separated List Parser with a Custom Generator](#implementing-a-comma-separated-list-parser-with-a-custom-generator)  
  - [Synchronous Parsing](#synchronous-parsing)  
  - [Asynchronous Parsing](#asynchronous-parsing)  
- [Using RunnableGenerator with Our Comma-Separated List Parser](#using-runnablegenerator-with-our-comma-separated-list-parser)  

### References

- [LangChain ChatOpenAI API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [LangChain custom functions](https://python.langchain.com/docs/how_to/functions/)
- [LangChain RunnableGenerator](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableGenerator.html)
- [Python Generators Documentation](https://docs.python.org/3/tutorial/classes.html#generators)
- [Python Async IO Documentation](https://docs.python.org/3/library/asyncio.html)
---

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The `langchain-opentutorial` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_openai",
        "langchain_core",
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "09-Generator",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

Alternatively, you can set and load `OPENAI_API_KEY` from a `.env` file.

**[Note]** This is only necessary if you haven't already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Implementing a Comma-Separated List Parser with a Custom Generator

When working with language models, you might receive outputs as plain text, such as comma-separated strings. To parse these into a structured format (e.g., a list) as they are generated, you can implement a custom generator function. This retains the streaming benefits — observing partial outputs in real time — while transforming the data into a more usable format.

### Synchronous Parsing

In this section, we define a custom generator function called `split_into_list()`. For each incoming chunk of tokens (strings), it builds up a string by aggregating characters until a comma is encountered within that chunk. At each comma, it yields the current text (stripped and split) as a list item.

```python
from typing import Iterator, List


# A user-defined parser that splits a stream of tokens into a comma-separated list
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    buffer = ""
    for chunk in input:
        # Accumulate tokens in the buffer
        buffer += chunk
        # Whenever we find a comma, split and yield the segment
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [buffer[:comma_index].strip()]
            buffer = buffer[comma_index + 1 :]
    # Finally, yield whatever remains in the buffer
    yield [buffer.strip()]
```

We then construct a LangChain pipeline that:

- Defines a prompt template for comma-separated outputs.
- Uses `ChatOpenAI` with `temperature=0.0` for deterministic responses. 
- Converts the raw output to a string using `StrOutputParser`.
- Pipes ( **|** ) the string output into `split_into_list()` for parsing.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Write a comma-separated list of 5 companies similar to: {company}"
)

# Initialize the model with temperature=0.0 for deterministic output
model = ChatOpenAI(temperature=0.0, model="gpt-4o")

# Chain 1: Convert to a string
str_chain = prompt | model | StrOutputParser()

# Chain 2: Parse the comma-separated string into a list using our generator
list_chain = str_chain | split_into_list
```

By streaming the output through `list_chain`, you can observe the partial results in real time. Each list item appears as soon as the parser encounters a comma in the stream.

```python
# Stream the parsed data
for chunk in list_chain.stream({"company": "Google"}):
    print(chunk, flush=True)
```

<pre class="custom">['Microsoft']
    ['Apple']
    ['Amazon']
    ['Facebook']
    ['IBM']
</pre>

If you need the entire parsed list at once (after the entire generation process is completed), you can use the `.invoke()` method instead of streaming.

```python
output = list_chain.invoke({"company": "Google"})
print(output)
```

<pre class="custom">['Microsoft', 'Apple', 'Amazon', 'Facebook', 'IBM']
</pre>

### Asynchronous Parsing

The method described above works for synchronous iteration. However, some applications may require **asynchronous** operations to prevent blocking the main thread. The following section shows how to achieve the same comma-separated parsing using an **async generator**.


The `asplit_into_list()` works similarly to its synchronous counterpart, aggregating tokens until a comma is encountered. However, it uses the `async for` construct to handle asynchronous data streams.

```python
from typing import AsyncIterator


async def asplit_into_list(input: AsyncIterator[str]) -> AsyncIterator[List[str]]:
    buffer = ""
    async for chunk in input:
        buffer += chunk
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [buffer[:comma_index].strip()]
            buffer = buffer[comma_index + 1 :]
    yield [buffer.strip()]
```

Then, you can **pipe** the asynchronous parser into a chain like the synchronous version.

```python
alist_chain = str_chain | asplit_into_list
```

When you call `astream()`, you can process each incoming data chunk as it becomes available within an asynchronous context.

```python
async for chunk in alist_chain.astream({"company": "Google"}):
    print(chunk, flush=True)
```

<pre class="custom">['Microsoft']
    ['Apple']
    ['Amazon']
    ['Facebook']
    ['IBM']
</pre>

Similarly, you can get the entire parsed list, using the asynchronous `ainvoke()` method.

```python
result = await alist_chain.ainvoke({"company": "Google"})
print(result)
```

<pre class="custom">['Microsoft', 'Apple', 'Amazon', 'Facebook', 'IBM']
</pre>

## Using RunnableGenerator with Our Comma-Separated List Parser

In addition to implementing your own generator functions directly, LangChain offers the `RunnableGenerator` class for more advanced or modular streaming behavior. This approach wraps your generator logic in a Runnable, easily pluggin it into a chain while preserving partial output streaming. Below, we modify our **comma-separated list parser** to demonstrate how `RunnableGenerator` can be applied.

### Advantages of RunnableGenerator
- Modularity: Easily encapsulate your parsing logic as a Runnable component.
- Consistency: The `RunnableGenerator` interface (`invoke`, `stream`, `ainvoke`, `astream`) is consistent with other LangChain Runnables.
- Extendability: Combine multiple Runnables (e.g., `RunnableLambda`, `RunnableGenerator`) in sequence for more complex transformations.  

### Transforming the Same Parser Logic

Previously, we defined `split_into_list()` as a standalone Python generator function. Now, let’s create an equivalent **transform** function, specifically designed for use with `RunnableGenerator`. Our goal remains the same: we want to parse a streaming sequence of tokens into a **list** of individual items upon encountering a comma.

```python
from langchain_core.runnables import RunnableGenerator
from typing import Iterator, List


def comma_parser_runnable(input_iter: Iterator[str]) -> Iterator[List[str]]:
    """
    This function accumulates tokens from input_iter and yields
    each chunk split by commas as a list.
    """
    buffer = ""
    for chunk in input_iter:
        buffer += chunk
        # Whenever we find a comma, split and yield
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [buffer[:comma_index].strip()]
            buffer = buffer[comma_index + 1 :]
    # Finally, yield whatever remains
    yield [buffer.strip()]


# Wrap it in a RunnableGenerator
parser_runnable = RunnableGenerator(comma_parser_runnable)
```

We can now integrate `parser_runnable` into the **same** prompt-and-model pipeline we used before.

```python
list_chain_via_runnable = str_chain | parser_runnable
```

When run, partial outputs will appear as single-element lists, like our original custom generator approach. 

The difference is that we’re now using `RunnableGenerator` to encapsulate the logic in a more modular, LangChain-native way.

```python
# Stream partial results
for parsed_chunk in list_chain_via_runnable.stream({"company": "Google"}):
    print(parsed_chunk)
```

<pre class="custom">['Microsoft']
    ['Apple']
    ['Amazon']
    ['Facebook']
    ['IBM']
</pre>
