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

# RecursiveJsonSplitter

- Author: [HeeWung Song(Dan)](https://github.com/kofsitho87)
- Peer Review : [BokyungisaGod](https://github.com/BokyungisaGod), [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/08-RecursiveJsonSplitter.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/08-RecursiveJsonSplitter.ipynb)
## Overview

This JSON splitter generates smaller JSON chunks by performing a depth-first traversal of JSON data.

The splitter aims to keep nested JSON objects intact as much as possible. However, to ensure chunk sizes remain within the ```min_chunk_size``` and ```max_chunk_size```, it will split objects if needed. Note that very large string values (those not containing nested JSON) are not subject to splitting.

If precise control over chunk size is required, you can use a **recursive text splitter** on the chunks this splitter creates.

**Splitting Criteria**

1. Text splitting method: Based on JSON values
2. Chunk size: Determined by character count


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Basic JSON Splitting](#basic-json-splitting)
- [Handling JSON Structure](#handling-json-structure)


### References

- [Langchain RecursiveJsonSplitter](https://python.langchain.com/api_reference/text_splitters/json/langchain_text_splitters.json.RecursiveJsonSplitter.html#langchain_text_splitters.json.RecursiveJsonSplitter)
- [Langchain How-to-split-JSONdata](https://python.langchain.com/docs/how_to/recursive_json_splitter/)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The ```langchain-opentutorial``` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
    ]
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
        "LANGCHAIN_PROJECT": "RecursiveJsonSplitter",
    }
)
```

Alternatively, you can set and load ```OPENAI_API_KEY``` from a ```.env``` file.

**[Note]** This is only necessary if you haven't already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```


<pre class="custom">True</pre>


## Basic JSON Splitting

Let's explore the basic methods of splitting JSON data using the ```RecursiveJsonSplitter```.

- JSON data preparation
- ```RecursiveJsonSplitter``` configuration
- Three splitting methods (```split_json```, ```create_documents```, and ```split_text```)
- Chunk size verification

```python
import requests

# Load the JSON data.
json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()
```

```python
json_data
```

Here is an example of splitting JSON data with the ```RecursiveJsonSplitter```.

```python
from langchain_text_splitters import RecursiveJsonSplitter

# Create a RecursiveJsonSplitter object that splits JSON data into chunks with a maximum size of 300
splitter = RecursiveJsonSplitter(max_chunk_size=300)
```

Use the ```splitter.split_json()``` method to recursively split JSON data.

```python
# Recursively split JSON data. Use this when you need to access or manipulate small JSON fragments.
json_chunks = splitter.split_json(json_data=json_data)
```

The following code demonstrates two methods for splitting JSON data using a splitter object (like an instance of ```RecursiveJsonSplitter```): use the ```splitter.create_documents()``` method to convert JSON data into ```Document``` objects, and use the ```splitter.split_text()``` method to split JSON data into a list of strings.

```python
# Create documents based on JSON data.
docs = splitter.create_documents(texts=[json_data])

# Create string chunks based on JSON data.
texts = splitter.split_text(json_data=json_data)

# Print the first string.
print(docs[0].page_content)

print("===" * 20)

# Print the split string chunks.
print(texts[0])
```

<pre class="custom">{"openapi": "3.1.0", "info": {"title": "LangSmith", "version": "0.1.0"}, "paths": {"/api/v1/sessions/{session_id}": {"get": {"tags": ["tracer-sessions"], "summary": "Read Tracer Session", "description": "Get a specific session."}}}}
    ============================================================
    {"openapi": "3.1.0", "info": {"title": "LangSmith", "version": "0.1.0"}, "paths": {"/api/v1/sessions/{session_id}": {"get": {"tags": ["tracer-sessions"], "summary": "Read Tracer Session", "description": "Get a specific session."}}}}
</pre>

## Handling JSON Structure

Let's explore how the ```RecursiveJsonSplitter``` handles different JSON structures and its limitations.

- Verification of list object size
- Parsing JSON structures
- Using the ```convert_lists``` parameter for list transformation

By examining ```texts[2]``` (one of the larger chunks), we can confirm it contains a list object.

- The second chunk exceeds the size limit (300) because it contains a list.
- The ```RecursiveJsonSplitter``` is designed not to split list objects.

```python
# Let's check the size of the chunks
print([len(text) for text in texts][:10])

# When examining one of the larger chunks, we can see that it contains a list object
print(texts[2])
```

<pre class="custom">[232, 197, 469, 210, 213, 237, 271, 191, 232, 215]
    {"paths": {"/api/v1/sessions/{session_id}": {"get": {"parameters": [{"name": "session_id", "in": "path", "required": true, "schema": {"type": "string", "format": "uuid", "title": "Session Id"}}, {"name": "include_stats", "in": "query", "required": false, "schema": {"type": "boolean", "default": false, "title": "Include Stats"}}, {"name": "accept", "in": "header", "required": false, "schema": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Accept"}}]}}}}
</pre>

You can parse the chunk at index 2 using the ```json``` module.

```python
import json

json_data = json.loads(texts[2])
json_data["paths"]
```




<pre class="custom">{'/api/v1/sessions/{session_id}': {'get': {'parameters': [{'name': 'session_id',
         'in': 'path',
         'required': True,
         'schema': {'type': 'string', 'format': 'uuid', 'title': 'Session Id'}},
        {'name': 'include_stats',
         'in': 'query',
         'required': False,
         'schema': {'type': 'boolean',
          'default': False,
          'title': 'Include Stats'}},
        {'name': 'accept',
         'in': 'header',
         'required': False,
         'schema': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
          'title': 'Accept'}}]}}}</pre>



Setting the ```convert_lists``` parameter to ```True``` transforms JSON lists into ```key:value``` pairs (formatted as ```index:item```).

```python
# The following preprocesses JSON and converts lists into dictionaries with index:item as key:value pairs
texts = splitter.split_text(json_data=json_data, convert_lists=True)
```

```python
# The list has been converted to a dictionary, and we'll check the result.
print(texts[2])
```

<pre class="custom">{"paths": {"/api/v1/sessions/{session_id}": {"get": {"parameters": {"2": {"name": "accept", "in": "header", "required": false, "schema": {"anyOf": {"0": {"type": "string"}, "1": {"type": "null"}}, "title": "Accept"}}}}}}}
</pre>

You can access specific documents within the ```docs``` list using their index.

```python
# Check the document at index 2.
print(docs[2])
```

<pre class="custom">page_content='{"paths": {"/api/v1/sessions/{session_id}": {"get": {"parameters": [{"name": "session_id", "in": "path", "required": true, "schema": {"type": "string", "format": "uuid", "title": "Session Id"}}, {"name": "include_stats", "in": "query", "required": false, "schema": {"type": "boolean", "default": false, "title": "Include Stats"}}, {"name": "accept", "in": "header", "required": false, "schema": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Accept"}}]}}}}'
</pre>
