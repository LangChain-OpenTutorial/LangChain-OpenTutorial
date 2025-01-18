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

This tutorial demonstrates how to use a **user-defined generator** (or async generator) in a `LangChain` pipeline to process text outputs in a streaming fashion. Specifically, we’ll show how to parse a comma-separated string output into a Python list, all while maintaining the benefits of streaming from a Language Model.

We will also cover asynchronous usage, showing how to adopt the same approach with async generators. By the end of this tutorial, you’ll be able to:

Implement a custom generator function that can handle streaming outputs
Parse comma-separated text chunks into a list in real time
Use both synchronous and asynchronous approaches for streaming
Integrate these parsers in a `LangChain` chain
Optionally, explore how `RunnableGenerator` can help implement custom generator transformations in a streaming context

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

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Implementing a Comma-Separated List Parser with a Custom Generator

When working with Language Models, you may often receive outputs in plain text form, such as comma-separated strings. If you want to parse those outputs into a structured format (e.g., a list) as they are generated, you can implement a custom generator function. This retains the streaming benefits—observing partial outputs in real time—while converting the data into a more usable format.

### Synchronous Parsing

In this section, we define a custom generator function `split_into_list()`. It accepts an iterator of tokens (strings) and continuously accumulates them until it encounters a comma. At each comma, it yields the current accumulated text (stripped and split) as a list item.


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

Here, we create a LangChain pipeline that does the following:

- Defines a prompt template to generate comma-separated outputs.
- Uses `ChatOpenAI` to get deterministic responses by setting `temperature=0.0`.
- Converts the raw output into a string using `StrOutputParser`.
- Pipes (|) that string output into our `split_into_list` function for parsing.

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

By streaming the output through `list_chain`, you can see the partial results in real time. Each chunk appears as soon as the parser encounters a comma:

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

If you prefer to get the entire parsed result at once (after the entire generation is completed), use the .`invoke()` method:

```python
output = list_chain.invoke({"company": "Google"})
print(output)
```

<pre class="custom">['Microsoft', 'Apple', 'Amazon', 'Facebook', 'IBM']
</pre>

### Asynchronous Parsing

The above approach works for synchronous iteration. However, some applications may require **async** iteration to avoid blocking. The following shows how to handle the same comma-separated parsing with an **async generator**.


Here, `asplit_into_list()` accumulates tokens in the same way but uses async for to handle asynchronous data streams.

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

Next, you can **pipe** the asynchronous parser into a chain just like the synchronous version:

```python
alist_chain = str_chain | asplit_into_list
```

When you call `astream()`, you can handle each chunk as it arrives, in an async context:


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

Similarly, you can get the entire parsed list using the asynchronous `ainvoke()` method:

```python
result = await alist_chain.ainvoke({"company": "Google"})
print(result)
```

<pre class="custom">['Microsoft', 'Apple', 'Amazon', 'Facebook', 'IBM']
</pre>

## Using `RunnableGenerator` with Our Comma-Separated List Parser
In addition to writing your own generator functions, you can leverage `RunnableGenerator` for more advanced or modular streaming behavior. This approach wraps your generator logic in a Runnable, making it easy to plug into a chain and still preserve partial output streaming. Below, we modify our **comma-separated list parser** to demonstrate how `RunnableGenerator` can be used.

### Why Use `RunnableGenerator`?
- Modularity: Easily encapsulate your parsing logic as a “runnable” component.
- Consistency: The `RunnableGenerator` interface ( `invoke` , `stream` , `ainvoke` , `astream` ) is consistent with other LangChain runnables.
- Extendability: Combine multiple runnables (e.g., `RunnableLambda` , `RunnableGenerator` ) in sequence for more complex transformations.  

### Transforming the Same Parser Logic

Previously, we defined `split_into_list()` as a standalone Python generator function. Let’s do something similar, but as a **transform** function for `RunnableGenerator`. We want to parse a streaming sequence of tokens into a **list** of individual items whenever we see a comma.

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

We can now integrate 'parser_runnable' into the **same** prompt-and-model pipeline we used before. 

```python
list_chain_via_runnable = str_chain | parser_runnable
```

When run, partial outputs will appear as single-element lists, just like our original custom generator approach. 

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
