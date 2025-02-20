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

# Understanding Common Python Syntax Used in LangGraph

- Author: [JeongHo Shin](https://github.com/ThePurpleCollar)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

LangGraph is a powerful framework that allows you to design complex workflows for language models using a graph-based structure. It enhances modularity, scalability, and efficiency in building AI-driven applications.

This tutorial explains key Python concepts frequently used in LangGraph, including `TypedDict` , `Annotated` , and the `add_messages` function. We will also compare these concepts with standard Python features to highlight their advantages and typical use cases.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Typedict](#typeddict)
- [Annotated](#annotated)
- [add_messages](#add_messages)

### References

- [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.


**[Note]**

The langchain-opentutorial is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
Check out the  [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_core",
        "langchain-anthropic",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
    ],
    verbose=False,
    upgrade=False,
)
```

You can set API keys in a `.env` file or set them manually.

[Note] If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "",  # set the project name same as the title
        }
    )
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## TypedDict

`TypedDict` is a feature introduced in Python's `typing` module that enables developers to define dictionaries with a fixed structure and explicit key-value types. It ensures type safety and improves code readability.

### Key Differences Between `dict` and `TypedDict`

1. **Type Checking**

- `dict` : Does not provide type checking at runtime or during development.

- `TypedDict` : Supports static type checking using tools like `mypy` or IDEs with type checking enabled.

2. **Key and Value Specification**

- `dict`: Specifies generic key-value types (e.g., `Dict[str, str]` ).

- `TypedDict` : Specifies the exact keys and their respective types.

3. **Flexibility**

- `dict` : Allows runtime addition or removal of keys without restriction.

- `TypedDict` : Enforces a predefined structure, disallowing extra keys unless explicitly marked.

### Why Use `TypedDict` ?

- **Type Safety** : Helps catch errors during development.

- **Readability** : Provides a clear schema for dictionaries.

- **IDE Support** : Enhances autocompletion and documentation.

- **Documentation** : Serves as self-documenting code.

### Example

```python
from typing import Dict, TypedDict

# Standard Python dictionary usage
sample_dict: Dict[str, str] = {
    "name": "Teddy",
    "age": "30",  # Stored as a string (allowed in dict)
    "job": "Developer",
}

# Using TypedDict
class Person(TypedDict):
    name: str
    age: int  # Defined as an integer
    job: str

typed_dict: Person = {"name": "Shirley", "age": 25, "job": "Designer"}

# Behavior with a standard dictionary
sample_dict["age"] = 35  # Type inconsistency is allowed
sample_dict["new_field"] = "Additional Info"  # Adding new keys is allowed

# Behavior with TypedDict
typed_dict["age"] = 35  # Correct usage
typed_dict["age"] = "35"  # Error: Type mismatch detected by type checker
typed_dict["new_field"] = "Additional Info"  # Error: Key not defined in TypedDict
```

`TypedDict` truly shines when paired with static type checkers like `mypy` or when using IDEs such as PyCharm or VS Code with type-checking enabled. These tools detect type mismatches and undefined keys at development time, providing valuable feedback to prevent runtime errors.

## Annotated

`Annotated` is a feature in Python's `typing` module that allows metadata to be added to type hints. This feature provides enhanced functionality by including additional context, improving code readability and usability for developers and tools alike. For example, metadata can act as additional documentation for readers or as actionable information for tools.

### Why Use `Annotated` ?

- **Additional Context** : Adds metadata to enrich type hints, improving clarity for developers and tools.

- **Enhanced Documentation** : Serves as self-contained documentation that can clarify the purpose and constraints of variables.

- **Validation** : Can be combined with libraries like Pydantic to enforce data validation using the annotated metadata.

- **Framework-Specific Behavior** : Enables advanced features in frameworks such as LangGraph by defining specialized operations.

### Syntax

- Type: Defines the variable's data type (e.g., `int`, `str`, `List[str]`, etc.).
- Metadata: Adds descriptive information about the variable (e.g., `"unit: cm"`, `"range: 0-100"`).

### Usage Example

```python
from typing import Annotated\

# Basic usage of Annotated with metadata for descriptive purposes
name: Annotated[str, "User's name"]
age: Annotated[int, "User's age (0-150)"]
```

### Example with `Pydantic` 

```python
from typing import Annotated, List
from pydantic import Field, BaseModel, ValidationError

class Employee(BaseModel):
    id: Annotated[int, Field(..., description="Employee ID")]
    name: Annotated[str, Field(..., min_length=3, max_length=50, description="Name")]
    age: Annotated[int, Field(gt=18, lt=65, description="Age (19-64)")]
    salary: Annotated[float, Field(gt=0, lt=10000, description="Salary (in units of 10,000, up to 10 billion)")]
    skills: Annotated[List[str], Field(min_items=1, max_items=10, description="Skills (1-10 items)")]

# Example of valid data
try:
    valid_employee = Employee(
        id=1, name="Teddynote", age=30, salary=1000, skills=["Python", "LangChain"]
    )
    print("Valid employee data:", valid_employee)
except ValidationError as e:
    print("Validation error:", e)

# Example of invalid data
try:
    invalid_employee = Employee(
        id=1, 
        name="Ted",  # Name is too short
        age=17,  # Age is out of range
        salary=20000,  # Salary exceeds the maximum
        skills="Python"  # Skills is not a list
    )
except ValidationError as e:
    print("Validation errors:")
    for error in e.errors():
        print(f"- {error['loc'][0]}: {error['msg']}")
```

<pre class="custom">Valid employee data: id=1 name='Teddynote' age=30 salary=1000.0 skills=['Python', 'LangChain']
    Validation errors:
    - age: Input should be greater than 18
    - salary: Input should be less than 10000
    - skills: Input should be a valid list
</pre>

## add_messages

The `add_messages` reducer function, referenced by the `messages` key, instructs LangGraph to append new messages to an existing list.

For state keys without annotations, each update overwrites the value, storing only the most recent data.

The `add_messages` function operates by merging two inputs (`left` and `right` ) into a unified list of messages.

### Key Features

- **Merges Two Message Lists** : Combines two separate message lists into one.

- **Maintains Append-Only State** : Ensures that new messages are added while retaining existing ones.

- **Replaces Messages with Matching IDs** : If a message in `right` has the same ID as one in `left`, it replaces the existing message.

### How It Works:
- Messages in `right` with IDs that match those in `left` replace the corresponding messages in `left` .

- All other messages in `right` are appended to `left` .

### Parameters:
- `left` (Messages): The base list of messages.

- `right` (Messages): A list of new messages to merge or a single message to add.

### Return Value:
- `Messages` : A new list of messages that merges `right` into `left` .

### Example

```python
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages


# Example 1: Merging two message lists
# `msgs1` and `msgs2` are combined into a single list without overlapping IDs.
msgs1 = [HumanMessage(content="Hello?", id="1")]
msgs2 = [AIMessage(content="Nice to meet you!", id="2")]

result1 = add_messages(msgs1, msgs2)
print(result1)

# Example 2: Replacing messages with the same ID
# If `msgs2` contains a message with the same ID as one in `msgs1`, 
# the message in `msgs2` replaces the corresponding message in `msgs1`.
msgs1 = [HumanMessage(content="Hello?", id="1")]
msgs2 = [HumanMessage(content="Nice to meet you!", id="1")]

result2 = add_messages(msgs1, msgs2)
print(result2)

```

<pre class="custom">[HumanMessage(content='Hello?', additional_kwargs={}, response_metadata={}, id='1'), AIMessage(content='Nice to meet you!', additional_kwargs={}, response_metadata={}, id='2')]
    [HumanMessage(content='Nice to meet you!', additional_kwargs={}, response_metadata={}, id='1')]
</pre>
