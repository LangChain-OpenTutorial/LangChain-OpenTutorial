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

# DatetimeOutputParser

- Author: [Donghak Lee](https://github.com/stsr1284)
- Peer Review : [JaeHo Kim](https://github.com/Jae-hoya), [ranian963](https://github.com/ranian963)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/06-DatetimeOutputParser.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/06-DatetimeOutputParser.ipynb)
## Overview

The ```DatetimeOutputParser``` is an output parser that generates structured outputs in the form of ```datetime``` objects.

By converting the outputs of LLMs into ```datetime``` objects, it enables more systematic and consistent processing of date and time data, making it useful for data processing and analysis.

This tutorial demonstrates how to use the ```DatetimeOutputParser``` to:
1. Set up and initialize the parser for ```datetime``` generation
2. Convert a ```datetime``` object to a string

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Using the DatetimeOutputParser](#using-the-datetimeoutputparser)
- [Using DatetimeOutputParser in astream](#using-datetimeoutputparser-in-astream)


### References

- [LangChain DatetimeOutputParser](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.datetime.DatetimeOutputParser.html)
- [LangChain ChatOpenAI API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
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
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "06-DatetimeOutputParser",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it.

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">False</pre>



## Using the ```DatetimeOutputParser```
If you need to generate output in the form of a date or time, the ```DatetimeOutputParser``` from LangChain simplifies the process.

The **format of the ```DatetimeOutputParser```** can be specified by referring to the table below.
| Format Code | Description           | Example              |
|--------------|-----------------------|----------------------|
| %Y           | 4-digit year          | 2024                 |
| %y           | 2-digit year          | 24                   |
| %m           | 2-digit month         | 07                   |
| %d           | 2-digit day           | 04                   |
| %H           | 24-hour format hour   | 14                   |
| %I           | 12-hour format hour   | 02                   |
| %p           | AM or PM              | PM                   |
| %M           | 2-digit minute        | 45                   |
| %S           | 2-digit second        | 08                   |
| %f           | Microsecond (6 digits)| 000123               |
| %z           | UTC offset            | +0900                |
| %Z           | Timezone name         | KST                  |
| %a           | Abbreviated weekday   | Thu                  |
| %A           | Full weekday name     | Thursday             |
| %b           | Abbreviated month     | Jul                  |
| %B           | Full month name       | July                 |
| %c           | Full date and time    | Thu Jul 4 14:45:08 2024 |
| %x           | Full date             | 07/04/24             |
| %X           | Full time             | 14:45:08             |


```python
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate

# Initialize the output parser
output_parser = DatetimeOutputParser()

# Specify date format
date_format = "%Y-%m-%d"
output_parser.format = date_format

# Get format instructions
format_instructions = output_parser.get_format_instructions()

# Create answer template for user questions
template = """Answer the users question:\n\n#Format Instructions: \n{format_instructions}\n\n#Question: \n{question}\n\n#Answer:"""

# Create a prompt from the template
prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        "format_instructions": format_instructions,
    },  # Use parser's format instructions
)

print(format_instructions)
print("-----------------------------------------------\n")
print(prompt)
```

<pre class="custom">Write a datetime string that matches the following pattern: '%Y-%m-%d'.
    
    Examples: 0594-05-12, 0088-08-25, 0371-10-02
    
    Return ONLY this string, no other words!
    -----------------------------------------------
    
    input_variables=['question'] input_types={} partial_variables={'format_instructions': "Write a datetime string that matches the following pattern: '%Y-%m-%d'.\n\nExamples: 0594-05-12, 0088-08-25, 0371-10-02\n\nReturn ONLY this string, no other words!"} template='Answer the users question:\n\n#Format Instructions: \n{format_instructions}\n\n#Question: \n{question}\n\n#Answer:'
</pre>

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# Combine the prompt, chat model, and output parser into a chain
chain = prompt | model | output_parser

# Call the chain to get an answer to the question
output = chain.invoke({"question": "The year Google was founded"})

print(output)
print(type(output))
```

<pre class="custom">1998-09-04 00:00:00
    <class 'datetime.datetime'>
</pre>

```python
# Convert the result to a string
output.strftime(date_format)
```




<pre class="custom">'1998-09-04'</pre>



## Using ```DatetimeOutputParser``` in ```astream```
Refer to the [user-defined generator](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/09-Generator.ipynb) to create a generator function.

Let's create a simple example that converts ```astream``` output to ```datetime``` objects using a generator function.



```python
from langchain_core.output_parsers.string import StrOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
import datetime
from typing import AsyncIterator, List

# Initialize the output parser
output_parser = DatetimeOutputParser()

# Specify date format
date_format = "%Y-%m-%d"
output_parser.format = date_format

# Get format instructions
format_instructions = output_parser.get_format_instructions()

# Create answer template for user questions
template = (
    "Answer the users question:\n\n"
    "#Format Instructions: \n{format_instructions}\n"
    "Write a comma-separated list of 5 founding years of companies similar to: {company}"
)

# Create a prompt from the template
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": format_instructions},
)

# Initialize the ChatOpenAI model with temperature set to 0.0
model = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")

# Create a chain combining the prompt, model, and string output parser
str_chain = prompt | model | StrOutputParser()


# Define an asynchronous function to convert strings to datetime objects
async def convert_strings_to_datetime(
    input: AsyncIterator[str],
) -> AsyncIterator[List[datetime.datetime]]:
    buffer = ""
    async for chunk in input:
        buffer += chunk
        while "," in buffer:
            comma_index = buffer.index(",")
            date_str = buffer[:comma_index].strip()
            date_obj = output_parser.parse(date_str)  # Convert to datetime object
            yield [date_obj]
            buffer = buffer[comma_index + 1 :]
    date_str = buffer.strip()
    if date_str:
        date_obj = output_parser.parse(
            date_str
        )  # Convert remaining buffer to datetime object
        yield [date_obj]


# Connect the str_chain and convert_strings_to_datetime in a pipeline
alist_chain = str_chain | convert_strings_to_datetime
```

```python
# Use async for loop to stream data.
async for chunk in alist_chain.astream({"company": "Google"}):
    # Print each chunk and flush the buffer.
    print(chunk, flush=True)
```

<pre class="custom">[datetime.datetime(1998, 9, 4, 0, 0)]
    [datetime.datetime(2004, 2, 4, 0, 0)]
    [datetime.datetime(2003, 2, 4, 0, 0)]
    [datetime.datetime(2001, 3, 1, 0, 0)]
    [datetime.datetime(1994, 3, 1, 0, 0)]
</pre>
