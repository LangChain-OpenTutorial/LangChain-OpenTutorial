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

# Comma Separated List Output Parser

- Author: [Junseong Kim](https://www.linkedin.com/in/%EC%A4%80%EC%84%B1-%EA%B9%80-591b351b2/)
- Design: []()
- Peer Review: []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

## Overview

The `CommaSeparatedListOutputParser` is a specialized output parser in LangChain designed for generating structured outputs in the form of comma-separated lists.

It simplifies the process of extracting and presenting data in a clear and concise list format, making it particularly useful for organizing information such as data points, names, items, or other structured values. By leveraging this parser, users can enhance data clarity, ensure consistent formatting, and improve workflow efficiency, especially in applications where structured outputs are essential.

This tutorial demonstrates how to use the `CommaSeparatedListOutputParser` to:

  1. Set up and initialize the parser for generating comma-separated lists
  2. Integrate it with a prompt template and language model
  3. Process structured outputs iteratively using streaming mechanisms


### Table of Contents

- [Comma Separated List Output Parser](#comma-separated-list-output-parser)
  - [Overview](#overview)
    - [Table of Contents](#table-of-contents)
    - [References](#references)
  - [Environment Setup](#environment-setup)
  - [Implementing the Comma-Separated List Output Parser](#implementing-the-comma-separated-list-output-parser)
    - [Importing Required Modules](#importing-required-modules)
    - [Creating the Prompt Template](#creating-the-prompt-template)
    - [Integrating with ChatOpenAI and Running the Chain](#integrating-with-chatopenai-and-running-the-chain)
    - [Accessing Data with Python Indexing](#accessing-data-with-python-indexing)
  - [Using Streamed Outputs](#using-streamed-outputs)

### References

- [LangChain ChatOpenAI API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [LangChain Core Output Parsers](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.list.CommaSeparatedListOutputParser.html#)
- [Python List Tutorial](https://docs.python.org/3.13/tutorial/datastructures.html)
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
        "LANGCHAIN_PROJECT": "02-CommaSeparatedListOutputParser",
    }
)
```
<pre class="custom">Environment variables have been set successfully.
</pre>You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```
<pre class="custom">True</pre>

## Implementing the Comma-Separated List Output Parser

If you need to generate outputs in the form of a comma-separated list, the `CommaSeparatedListOutputParser` from LangChain simplifies the process. 

Below is a step-by-step implementation:

### Importing Required Modules

Start by importing the necessary modules and initializing the `CommaSeparatedListOutputParser`. Retrieve the formatting instructions from the parser to guide the output structure.


```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# Initialize the output parser
output_parser = CommaSeparatedListOutputParser()

# Retrieve format instructions for the output parser
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
```
<pre class="custom">Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`
</pre>

### Creating the Prompt Template

Define a `PromptTemplate` that dynamically generates a list of items. The placeholder subject will be replaced with the desired topic during execution.

```python
from langchain_core.prompts import PromptTemplate

# Define the prompt template
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 'subject' will be dynamically replaced
    partial_variables={
        "format_instructions": format_instructions
    },  # Use parser's format instructions
)
print(prompt)
```
<pre class="custom">input_variables=['subject'] input_types={} partial_variables={'format_instructions': 'Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`'} template='List five {subject}.\n{format_instructions}'
</pre>

### Integrating with ChatOpenAI and Running the Chain

Combine the `PromptTemplate`, `ChatOpenAI` model, and `CommaSeparatedListOutputParser` into a chain. Finally, run the chain with a specific `subject` to produce results.

```python
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model
model = ChatOpenAI(temperature=0)

# Combine the prompt, model, and output parser into a chain
chain = prompt | model | output_parser

# Run the chain with a specific subject
result = chain.invoke({"subject": "famous landmarks in South Korea"})
print(result)
```
<pre class="custom">['Gyeongbokgung Palace', 'N Seoul Tower', 'Bukchon Hanok Village', 'Seongsan Ilchulbong Peak', 'Haeundae Beach']
</pre>

### Accessing Data with Python Indexing

Since the `CommaSeparatedListOutputParser` automatically formats the output as a Python list, you can easily access individual elements using indexing.

```python
# Accessing specific elements using Python indexing
print("First Landmark:", result[0])
print("Second Landmark:", result[1])
print("Last Landmark:", result[-1])
```
<pre class="custom">First Landmark: Gyeongbokgung Palace
Second Landmark: N Seoul Tower
Last Landmark: Haeundae Beach
</pre>

## Using Streamed Outputs

For larger outputs or real-time feedback, you can process the results using the `stream` method. This allows you to handle data piece by piece as it is generated.

```python
# Iterate through the streamed output for a subject
for output in chain.stream({"subject": "famous landmarks in South Korea"}):
    print(output)
```
<pre class="custom">['Gyeongbokgung Palace']
['N Seoul Tower']
['Bukchon Hanok Village']
['Seongsan Ilchulbong Peak']
['Haeundae Beach']
</pre>



