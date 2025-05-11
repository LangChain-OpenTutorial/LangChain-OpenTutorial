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

# JsonOutputParser

- Author: [Jaehun Choi](https://github.com/ash-hun)
- Peer Review : [Jeongeun Lim](https://www.linkedin.com/in/jeongeun-lim-808978188/), [frimer](https://github.com/brian604)
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/04-JsonOutputParser.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/04-JsonOutputParser.ipynb)
## Overview

This tutorial covers the implementation of the ```JsonOutputParser```.
```JsonOutputParser``` is a tool that allows users to specify the desired JSON schema. It is designed to enable an LLM(Large Language Model) to query data and return results in JSON format that adheres to the specified schema.
To ensure that the LLM processes data accurately and efficiently, generating JSON in the desired format, the model must have sufficient capacity (e.g., intelligence). For instance, the llama-70B model has a larger capacity compared to the llama-8B model, making it more suitable for handling complex data.

**[Note]**

**JSON (JavaScript Object Notation)** is a lightweight data interchange format used for storing and structuring data. It plays a crucial role in web development and is widely used for communication between servers and clients. JSON is based on text that is easy to read and simple for machines to parse and generate.

JSON data consists of key-value pairs. Here, the "key" is a string, and the "value" can be various data types. JSON has two primary structures:

- Object: A collection of key-value pairs enclosed in curly braces { }. Each key is associated with its value using a colon ( : ), and multiple key-value pairs are separated by commas ( , ).  
- Array: An ordered list of values enclosed in square brackets [ ]. Values within an array are separated by commas ( , ).

```json
{
  "name": "John Doe",
  "age": 30,
  "is_student": false,
  "skills": ["Java", "Python", "JavaScript"],
  "address": {
    "street": "123 Main St",
    "city": "Anytown"
  }
}
```

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Using JsonOutputParser with Pydantic](#using-jsonoutputparser-with-pydantic)
- [Using JsonOutputParser without Pydantic](#using-jsonoutputparser-without-pydantic)

### References

- [JsonOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.json.JsonOutputParser.html)
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
        "langsmith",
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
        "LANGCHAIN_PROJECT": "04-JsonOutputParser",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it.  
**[Note]** This is not necessary if your've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Using ```JsonOutputParser``` with ```Pydantic```

If you need to generate output in JSON format, you can easily implement it using LangChain's ```JsonOutputParser```. There are 2 ways to generate output in JSON format: 

- Using ```Pydantic```
- Not using ```Pydantic```

Follow the steps below to implement it:

Start by importing the necessary modules.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
```

```python
# Create an OpenAI object
model = ChatOpenAI(temperature=0, model_name="gpt-4o")
```

Define the output data schema format.

```python
# Use Pydantic to define the data schema for the output format.
class Topic(BaseModel):
    description: str = Field(description="A concise description of the topic")
    hashtags: str = Field(description="Keywords in hashtag format (at least 2)")
```

Set up the parser using ```JsonOutputParser``` and inject instructions into the prompt template.

```python
# Write your question
question = "Please explain the severity of global warming."

# Set up the parser and inject the instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Topic)
print(parser.get_format_instructions())
```

<pre class="custom">The output should be formatted as a JSON instance that conforms to the JSON schema below.
    
    As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
    the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
    
    Here is the output schema:
    ```
    {"properties": {"description": {"description": "A concise description of the topic", "title": "Description", "type": "string"}, "hashtags": {"description": "Keywords in hashtag format (at least 2)", "title": "Hashtags", "type": "string"}}, "required": ["description", "hashtags"]}
    ```
</pre>

```python
# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly AI assistant. Answer questions concisely."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Combine the prompt, model, and JsonOutputParser into a chain
chain = prompt | model | parser

# Run the chain with your question
answer = chain.invoke({"question": question})
```

```python
# Check the type.
type(answer)
```




<pre class="custom">dict</pre>



```python
# Output the answer object.
answer
```




<pre class="custom">{'description': "Global warming is a critical environmental issue characterized by the increase in Earth's average surface temperature due to rising levels of greenhouse gases. It leads to severe weather changes, rising sea levels, and impacts on biodiversity and human life.",
     'hashtags': '#GlobalWarming #ClimateChange #EnvironmentalImpact'}</pre>



## Using ```JsonOutputParser``` without ```Pydantic```

You can generate output in JSON format without ```Pydantic```. 

Follow the steps below to implement it:

```python
# Write your question
question = "Please provide information about global warming. Include the explanation in description and the related keywords in `hashtags`."

# Initialize JsonOutputParser
parser = JsonOutputParser()

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly AI assistant. Answer questions concisely."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

# Inject instruction to prompt
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Combine the prompt, model, and JsonOutputParser into a chain
chain = prompt | model | parser

# Run the chain with your question
response = chain.invoke({"question": question})
print(response)
```

<pre class="custom">{'description': "Global warming refers to the long-term increase in Earth's average surface temperature due to human activities, primarily the emission of greenhouse gases like carbon dioxide, methane, and nitrous oxide. These emissions result from burning fossil fuels, deforestation, and industrial processes, leading to the greenhouse effect, where heat is trapped in the atmosphere. This warming has significant impacts on weather patterns, sea levels, and ecosystems, contributing to climate change and posing risks to biodiversity and human societies.", 'hashtags': ['#GlobalWarming', '#ClimateChange', '#GreenhouseGases', '#CarbonEmissions', '#FossilFuels', '#Deforestation', '#Sustainability', '#EnvironmentalImpact']}
</pre>
