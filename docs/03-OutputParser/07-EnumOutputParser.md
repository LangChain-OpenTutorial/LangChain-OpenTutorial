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

# EnumOutputParser

- Author: [ranian963](https://github.com/ranian963)
- Peer Review : [JaeHo Kim](https://github.com/Jae-hoya)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/07-EnumOutputParser.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/07-EnumOutputParser.ipynb)
## Overview
In this tutorial, we introduce how to use ```EnumOutputParser``` to **extract valid Enum values** from the output of an LLM.

```EnumOutputParser``` is a tool that parses the output of a language model into **one of the predefined enumeration (Enum) values** , offering the following features:

- Enumeration Parsing: Converts the string output into a predefined ```Enum``` value.
- Type Safety: Ensures that the parsed result is always one of the defined ```Enum``` values.
- Flexibility: Automatically handles spaces and line breaks.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Introduction to EnumOutputParser](#introduction-to-enumoutputparser)
- [Example: Colors Enum Parser](#example-colors-enum-parser)

### References

- [LangChain: Output Parsers](https://python.langchain.com/docs/concepts/output_parsers/)
- [LangChain: EnumOutputParser](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.enum.EnumOutputParser.html#langchain.output_parsers.enum.EnumOutputParser)
- [Enum in Python](https://docs.python.org/3/library/enum.html)
- [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)
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
# Installing required libraries
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_openai"
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
        "LANGCHAIN_PROJECT": "07-EnumOutputParser",
    }
)
```

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```

## Introduction to EnumOutputParser

```EnumOutputParser``` is a tool that strictly parses an LLM's output into a defined enumeration (Enum).
This ensures that the model output is always one of the enumerated values.

**Use cases**
- When you only want one valid choice from a set of possibilities.
- When you want to avoid typos and variations by using a clear Enum value.

In the following example, we define an ```Colors``` Enum and make the LLM return one of ```red/green/blue``` by parsing the output.

## Example: Colors Enum Parser

The code below shows how to define the ```Colors(Enum)``` class and wrap it with ```EnumOutputParser```, then integrate it into a prompt chain.
Once the chain is executed, the LLM response is **strictly** parsed into one of the values in ```Colors```.

```python
# Import EnumOutputParser
from langchain.output_parsers.enum import EnumOutputParser
```

Define the ```Colors``` enumeration using the ```Enum``` class from Python's built-in ```enum``` module.

```python
from enum import Enum

class Colors(Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
```

Now we create an ```EnumOutputParser``` object for parsing strings into the ```Colors``` enumeration.

```python
# Instantiate EnumOutputParser
parser = EnumOutputParser(enum=Colors)

# You can view the format instructions that the parser expects.
print(parser.get_format_instructions())
```

Below is an example that constructs a simple chain using ```PromptTemplate``` and ```ChatOpenAI```.
If the LLM responds about which color the object "sky" is, the parser will convert that string into a ```Colors``` value.

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Prompt template: the parser's format instructions are added at the end.
prompt = (
    PromptTemplate.from_template(
        """Which color is this object?

Object: {object}

Instructions: {instructions}"""
    ).partial(instructions=parser.get_format_instructions())
)

# Entire chain: (prompt) -> (LLM) -> (Enum Parser)
chain = prompt | ChatOpenAI(temperature=0) | parser
```

Now let's run the chain.

```python
response = chain.invoke({"object": "sky"})
print("Parsed Enum:", response)
print("Raw Enum Value:", response.value)
```
