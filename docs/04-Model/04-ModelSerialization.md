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

# Model Serialization

- Author: [Mark](https://github.com/obov)
- Peer Review : [Jaemin Hong](https://github.com/geminii01), [Dooil Kwak](https://github.com/back2zion)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/04-ModelSerialization.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/04-ModelSerialization.ipynb)
## Overview

Serialization is the process of converting an object into a format that can be easily stored, shared, or transmitted, and later reconstructed. In the LangChain framework, classes implement standard methods for serialization, providing several advantages:

- **Separation of Secrets**: Sensitive information, such as API keys, is separated from other parameters and can be securely reloaded into the object during deserialization.

- **Version Compatibility**: Deserialization remains compatible across different package versions, ensuring that objects serialized with one version of LangChain can be properly deserialized with another.

All LangChain objects inheriting from ```Serializable``` are JSON-serializable, including messages, document objects (e.g., those returned from retrievers), and most Runnables such as chat models, retrievers, and chains implemented with the LangChain Expression Language.

### Saving and Loading LangChain Objects

To effectively manage LangChain objects, you can serialize and deserialize them using the following functions:

- **```dumpd```**: Returns a dictionary representation of an object, suitable for JSON serialization.

- **```dumps```**: Returns a JSON string representation of an object.

- **```load```**: Reconstructs an object from its dictionary representation.

- **```loads```**: Reconstructs an object from its JSON string representation.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Dumps and Loads](#dumps-and-loads)
- [Dumpd and Load](#dumpd-and-load)
- [Serialization with pickle](#serialization-with-pickle)
- [Is Every Runnable Serializable?](#is-every-runnable-serializable?)

### References

- [How to save and load LangChain objects](https://python.langchain.com/docs/how_to/serialization/)
- [dumpd](https://python.langchain.com/api_reference/core/load/langchain_core.load.dump.dumpd.html)
- [dumps](https://python.langchain.com/api_reference/core/load/langchain_core.load.dump.dumps.html)
- [load](https://python.langchain.com/api_reference/core/load/langchain_core.load.load.load.html)
- [loads](https://python.langchain.com/api_reference/core/load/langchain_core.load.load.loads.html)
- [pickle - Python object serialization](https://docs.python.org/3/library/pickle.html)

---


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
        "OPENAI_API_KEY": "Your API KEY",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Caching",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Alternatively, one can set environmental variables with load_dotenv
from dotenv import load_dotenv


load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Create model
llm = ChatOpenAI(model_name="gpt-4o-mini")

# Generate prompt
prompt = PromptTemplate.from_template(
    "Sumarize about the {country} in about 200 characters"
)

# Create chain
chain = prompt | llm
```

## ```Dumps``` and ```Loads```

- ```dumps``` : LangChain object into a JSON-formatted string
- ```loads``` : JSON-formatted string into a LangChain object


```python
from langchain_core.load.dump import dumps

# Serialize LangChain object to JSON like string

serialized_llm = dumps(llm, pretty=True)
print(serialized_llm)
print(type(serialized_llm))

serialized_prompt = dumps(prompt)
print(serialized_prompt[:100] + " ...")
print(type(serialized_prompt))

serialized_chain = dumps(chain)
print(serialized_chain[:100] + " ...")
print(type(serialized_chain))
```

<pre class="custom">{
      "lc": 1,
      "type": "constructor",
      "id": [
        "langchain",
        "chat_models",
        "openai",
        "ChatOpenAI"
      ],
      "kwargs": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "openai_api_key": {
          "lc": 1,
          "type": "secret",
          "id": [
            "OPENAI_API_KEY"
          ]
        },
        "max_retries": 2,
        "n": 1
      },
      "name": "ChatOpenAI"
    }
    <class 'str'>
    {"lc": 1, "type": "constructor", "id": ["langchain", "prompts", "prompt", "PromptTemplate"], "kwargs ...
    <class 'str'>
    {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "runnable", "RunnableSequence"], "kwa ...
    <class 'str'>
</pre>

```python
from langchain_core.load.load import loads

# Deserialize JSON like string to LangChain object

deserialized_llm = loads(serialized_llm)
print(type(deserialized_llm))

deserialized_prompt = loads(serialized_prompt)
print(type(deserialized_prompt))

deserialized_chain = loads(serialized_chain)
print(type(deserialized_chain))
```

<pre class="custom"><class 'langchain_openai.chat_models.base.ChatOpenAI'>
    <class 'langchain_core.prompts.prompt.PromptTemplate'>
    <class 'langchain_core.runnables.base.RunnableSequence'>
</pre>

    /var/folders/q_/52ctm0y10h589cwbbptjsvrw0000gp/T/ipykernel_82468/2405148250.py:5: LangChainBetaWarning: The function `loads` is in beta. It is actively being worked on, so the API may change.
      deserialized_llm = loads(serialized_llm)
    

```python
# Invoke chains

response = chain.invoke({"country": "South Korea"})
print(response.content)

deserialized_response = deserialized_chain.invoke({"country": "South Korea"})
print(deserialized_response.content)

deserialized_response_composed = (deserialized_prompt | deserialized_llm).invoke(
    {"country": "South Korea"}
)
print(deserialized_response_composed.content)
```

<pre class="custom">South Korea, located on the Korean Peninsula, is known for its vibrant culture, advanced technology, and rich history. Major cities include Seoul and Busan, and it has a strong economy and global influence.
    South Korea, located on the Korean Peninsula, is known for its vibrant culture, technological advancements, and dynamic economy. Seoul is its capital, and the nation is famous for K-pop, cuisine, and rich history.
    South Korea, located on the Korean Peninsula, is a vibrant democracy known for its advanced technology, rich culture, and K-pop music. It's a global leader in innovation and economic development.
</pre>

## ```Dumpd``` and ```Load```

- ```dumpd``` : LangChain object into a dictionary
- ```load``` : dictionary into a LangChain object


```python
from langchain_core.load.dump import dumpd

# Serialize LangChain object to dictionary

serialized_llm = dumpd(llm)
print(type(serialized_llm))

serialized_prompt = dumpd(prompt)
print(type(serialized_prompt))

serialized_chain = dumpd(chain)
print(type(serialized_chain))
```

<pre class="custom"><class 'dict'>
    <class 'dict'>
    <class 'dict'>
</pre>

```python
from langchain_core.load.load import load

# Deserialize dictionary to LangChain object

deserialized_llm = load(serialized_llm)
print(type(deserialized_llm))

deserialized_prompt = load(serialized_prompt)
print(type(deserialized_prompt))

deserialized_chain = load(serialized_chain)
print(type(deserialized_chain))
```

<pre class="custom"><class 'langchain_openai.chat_models.base.ChatOpenAI'>
    <class 'langchain_core.prompts.prompt.PromptTemplate'>
    <class 'langchain_core.runnables.base.RunnableSequence'>
</pre>

    /var/folders/q_/52ctm0y10h589cwbbptjsvrw0000gp/T/ipykernel_82468/4209275167.py:5: LangChainBetaWarning: The function `load` is in beta. It is actively being worked on, so the API may change.
      deserialized_llm = load(serialized_llm)
    

```python
# Invoke chains

response = chain.invoke({"country": "South Korea"})
print(response.content)

deserialized_response = deserialized_chain.invoke({"country": "South Korea"})
print(deserialized_response.content)

deserialized_response_composed = (deserialized_prompt | deserialized_llm).invoke(
    {"country": "South Korea"}
)
print(deserialized_response_composed.content)
```

<pre class="custom">South Korea, located on the Korean Peninsula, is known for its vibrant culture, advanced technology, and K-pop music. Its capital, Seoul, is a bustling metropolis blending tradition and modernity.
    South Korea is a vibrant East Asian nation known for its technological advancements, rich culture, K-pop, and delicious cuisine. It has a strong economy and a unique blend of tradition and modernity.
    South Korea, located on the Korean Peninsula, is known for its vibrant culture, advanced technology, and economic strength. Major cities include Seoul and Busan. It has a rich history and a strong global presence.
</pre>

## Serialization with ```pickle```

The ```pickle``` module in Python is used for serializing and deserializing Python object structures, also known as _pickling_ and _unpickling_. Serialization involves converting a Python object hierarchy into a byte stream, while deserialization reconstructs the object hierarchy from the byte stream.

[```pickle``` - Python object serialization for more details](https://docs.python.org/3/library/pickle.html)

### Key Functions

- **```pickle.dump(obj, file)```**: Serializes ```obj``` and writes it to the open file object ```file```.

- **```pickle.load(file)```**: Reads a byte stream from the open file object ```file``` and deserializes it back into a Python object.


```python
from langchain_core.load.dump import dumpd

# Serialize LangChain object to dictionary

serialized_llm = dumpd(llm)
print(type(serialized_llm))

serialized_prompt = dumpd(prompt)
print(type(serialized_prompt))

serialized_chain = dumpd(chain)
print(type(serialized_chain))
```

<pre class="custom"><class 'dict'>
    <class 'dict'>
    <class 'dict'>
</pre>

```python
import pickle
import os

# Serialize dictionary to pickle file

os.makedirs("data", exist_ok=True)

with open("data/serialized_llm.pkl", "wb") as f:
    pickle.dump(serialized_llm, f)

with open("data/serialized_prompt.pkl", "wb") as f:
    pickle.dump(serialized_prompt, f)

with open("data/serialized_chain.pkl", "wb") as f:
    pickle.dump(serialized_chain, f)
```

```python
# Deserialize pickle file to dictionary

with open("data/serialized_llm.pkl", "rb") as f:
    loaded_llm = pickle.load(f)
    print(type(loaded_llm))

with open("data/serialized_prompt.pkl", "rb") as f:
    loaded_prompt = pickle.load(f)
    print(type(loaded_prompt))

with open("data/serialized_chain.pkl", "rb") as f:
    loaded_chain = pickle.load(f)
    print(type(loaded_chain))
```

<pre class="custom"><class 'dict'>
    <class 'dict'>
    <class 'dict'>
</pre>

```python
from langchain_core.load.load import load

# Deserialize dictionary to LangChain object

deserialized_llm = load(serialized_llm)
print(type(deserialized_llm))

deserialized_prompt = load(serialized_prompt)
print(type(deserialized_prompt))

deserialized_chain = load(serialized_chain)
print(type(deserialized_chain))
```

<pre class="custom"><class 'langchain_openai.chat_models.base.ChatOpenAI'>
    <class 'langchain_core.prompts.prompt.PromptTemplate'>
    <class 'langchain_core.runnables.base.RunnableSequence'>
</pre>

```python
# Invoke chains

response = chain.invoke({"country": "South Korea"})
print(response.content)

deserialized_response = deserialized_chain.invoke({"country": "South Korea"})
print(deserialized_response.content)

deserialized_response_composed = (deserialized_prompt | deserialized_llm).invoke(
    {"country": "South Korea"}
)
print(deserialized_response_composed.content)
```

<pre class="custom">South Korea, located on the Korean Peninsula, is known for its technological advancements, rich culture, and history. Major cities like Seoul blend modernity with tradition, while K-pop and cuisine gain global popularity.
    South Korea, located on the Korean Peninsula, is known for its advanced technology, vibrant culture, and rich history. It features a dynamic economy, popular K-pop music, and delicious cuisine.
    South Korea is a vibrant East Asian nation known for its advanced technology, rich culture, and historical landmarks. It's famous for K-pop, delicious cuisine, and significant economic growth post-war.
</pre>

## Is Every Runnable Serializable?

LangChain's ```dumps``` and ```dumpd``` methods attempt to serialize objects as much as possible, but the resulting data may be incomplete.

1. Even if the ```is_lc_serializable``` method does not exist or returns ```False```, a result is still generated.
2. Even if the ```is_lc_serializable``` method returns ```True``` and serialization is successful, deserialization may fail.

After serialization, it is essential to check if the JSON data contains ```"type": "not_implemented"```. Only then can the ```load``` or ```loads``` functions be used safely.


```python
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads


def custom_function(llm_response):
    return llm_response.content


# Define chains that make same results
chain_with_custom_function = chain | custom_function
print(type(chain_with_custom_function))
chain_with_str_output_parser = chain | StrOutputParser()
print(type(chain_with_str_output_parser))

response = chain_with_custom_function.invoke({"country": "South Korea"})
print(response)

response = chain_with_str_output_parser.invoke({"country": "South Korea"})
print(response)
```

<pre class="custom"><class 'langchain_core.runnables.base.RunnableSequence'>
    <class 'langchain_core.runnables.base.RunnableSequence'>
    South Korea, located in East Asia, is known for its rich culture, advanced technology, and vibrant economy. It features bustling cities, traditional cuisine, and global influence through K-pop and cinema.
    South Korea, located in East Asia, is known for its advanced technology, rich culture, and vibrant economy. It's famous for K-pop, cuisine, and historical sites, blending tradition with modernity.
</pre>

```python
# Both of them are serializable
print(chain_with_custom_function.is_lc_serializable())
print(chain_with_str_output_parser.is_lc_serializable())
```

<pre class="custom">True
    True
</pre>

```python
try:
    print(
        "...\n"
        # You can see that the serialized string contains "type": "not_implemented"
        + ((serialized_str := dumps(chain_with_custom_function, pretty=True)))[-270:]
    )
    # First one fail to deserialize
    loads(serialized_str)
except Exception as e:
    print("Error : \n", e)

print(type(deserialized_chain := loads(dumps(chain_with_str_output_parser))))
print(deserialized_chain.invoke({"country": "South Korea"}))
```

<pre class="custom">...
    
        ],
        "last": {
          "lc": 1,
          "type": "not_implemented",
          "id": [
            "langchain_core",
            "runnables",
            "base",
            "RunnableLambda"
          ],
          "repr": "RunnableLambda(custom_function)"
        }
      },
      "name": "RunnableSequence"
    }
    Error : 
     Trying to load an object that doesn't implement serialization: {'lc': 1, 'type': 'not_implemented', 'id': ['langchain_core', 'runnables', 'base', 'RunnableLambda'], 'repr': 'RunnableLambda(custom_function)'}
    <class 'langchain_core.runnables.base.RunnableSequence'>
    South Korea, located on the Korean Peninsula, is known for its advanced technology, rich culture, and vibrant economy. It has a democratic government and is famous for K-pop, cuisine, and historical sites.
</pre>

```python
# RunnableLambda and custom_function has no is_lc_serializable method
# But they are serializable

try:
    print(RunnableLambda(custom_function).is_lc_serializable())
except Exception as e:
    print("Error : \n", e)

print(dumps(RunnableLambda(custom_function), pretty=True))

try:
    print(custom_function.is_lc_serializable())
except Exception as e:
    print("Error : \n", e)

print(dumps(custom_function, pretty=True))
```

<pre class="custom">Error : 
     'RunnableLambda' object has no attribute 'is_lc_serializable'
    {
      "lc": 1,
      "type": "not_implemented",
      "id": [
        "langchain_core",
        "runnables",
        "base",
        "RunnableLambda"
      ],
      "repr": "RunnableLambda(custom_function)"
    }
    Error : 
     'function' object has no attribute 'is_lc_serializable'
    {
      "lc": 1,
      "type": "not_implemented",
      "id": [
        "__main__",
        "custom_function"
      ],
      "repr": "<function custom_function at 0x114b99440>"
    }
</pre>

```python
from langchain_core.load.serializable import Serializable

# Serializable has is_lc_serializable method
# But it returns False
print(Serializable.is_lc_serializable())

# But also it is serializable
print(dumps(Serializable, pretty=True))
print(dumpd(Serializable))
```

<pre class="custom">False
    {
      "lc": 1,
      "type": "not_implemented",
      "id": [
        "langchain_core",
        "load",
        "serializable",
        "Serializable"
      ],
      "repr": "<class 'langchain_core.load.serializable.Serializable'>"
    }
    {'lc': 1, 'type': 'not_implemented', 'id': ['langchain_core', 'load', 'serializable', 'Serializable'], 'repr': "<class 'langchain_core.load.serializable.Serializable'>"}
</pre>
