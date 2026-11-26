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

# Caching

- Author: [Joseph](https://github.com/XaviereKU)
- Peer Review : [Teddy Lee](https://github.com/teddylee777), [BAEM1N](https://github.com/BAEM1N)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/03-Cache.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/03-Cache.ipynb)
## Overview

```LangChain``` provides optional caching layer for LLMs.

This is useful for two reasons:
- When requesting the same completions multiple times, it can **reduce the number of API calls** to the LLM provider and thus save costs.
- By **reduing the number of API calls** to the LLM provider, it can **improve the running time of the application.**

In this tutorial, we will use ```gpt-4o-mini``` OpenAI API and utilize two kinds of cache, ```InMemoryCache``` and ```SQLiteCache```.  
At end of each section we will compare wall times between before and after caching.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [InMemoryCache](#inmemorycache)
- [SQliteCache](#sqlitecache)

### References
- [SQLIteCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.SQLiteCache.html#langchain_community.cache.SQLiteCache)
- [InMemoryCache](https://python.langchain.com/api_reference/core/caches/langchain_core.caches.InMemoryCache.html)
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
        "langchain_community",
        "langchain_openai",
        # "vllm", # this is for optional section
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
        "OPENAI_API_KEY": "You OpenAI API KEY",
        "LANGCHAIN_API_KEY": "LangChain API KEY",
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




<pre class="custom">False</pre>



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

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response.content)
```

<pre class="custom">South Korea, located on the Korean Peninsula, is known for its rich culture, advanced technology, and vibrant economy. It features bustling cities like Seoul, renowned cuisine, and historic landmarks.
    CPU times: total: 93.8 ms
    Wall time: 1.54 s
</pre>

## ```InMemoryCache```
First, cache the answer to the same question using ```InMemoryCache```.

```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Set InMemoryCache
set_llm_cache(InMemoryCache())
```

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response.content)
```

<pre class="custom">South Korea is a technologically advanced country known for its fast-paced lifestyle, vibrant culture, and delicious cuisine. It is a leader in industries such as electronics, automotive, and entertainment. The country also has a rich history and beautiful landscapes, making it a popular destination for tourists.
    CPU times: total: 0 ns
    Wall time: 996 ms
</pre>

Now we invoke the chain with the same question.

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response.content)
```

<pre class="custom">South Korea is a technologically advanced country known for its fast-paced lifestyle, vibrant culture, and delicious cuisine. It is a leader in industries such as electronics, automotive, and entertainment. The country also has a rich history and beautiful landscapes, making it a popular destination for tourists.
    CPU times: total: 0 ns
    Wall time: 3 ms
</pre>

Note that if we set ```InMemoryCache``` again, the cache will be lost and the wall time will increase.

```python
set_llm_cache(InMemoryCache())
```

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response.content)
```

<pre class="custom">South Korea is a tech-savvy, modern country known for its vibrant culture, delicious cuisine, and booming economy. It is a highly developed nation with advanced infrastructure, high standards of living, and a strong emphasis on education. The country also has a rich history and is famous for its K-pop music and entertainment industry.
    CPU times: total: 0 ns
    Wall time: 972 ms
</pre>

## ```SQLiteCache```
Now, we cache the answer to the same question by using ```SQLiteCache```.

```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import os

# Create cache directory
if not os.path.exists("cache"):
    os.makedirs("cache")

# Set SQLiteCache
set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))
```

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response.content)
```

<pre class="custom">South Korea is a technologically advanced country in East Asia, known for its booming economy, vibrant pop culture, and rich history. It is home to K-pop, Samsung, and delicious cuisine like kimchi. The country also faces tensions with North Korea and strives for reunification.
    CPU times: total: 31.2 ms
    Wall time: 953 ms
</pre>

Now we invoke the chain with the same question.

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response.content)
```

<pre class="custom">South Korea is a technologically advanced country in East Asia, known for its booming economy, vibrant pop culture, and rich history. It is home to K-pop, Samsung, and delicious cuisine like kimchi. The country also faces tensions with North Korea and strives for reunification.
    CPU times: total: 375 ms
    Wall time: 375 ms
</pre>

Note that if we use ```SQLiteCache```, setting caching again does not delete stored cache.

```python
set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))
```

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response.content)
```

<pre class="custom">South Korea is a technologically advanced country in East Asia, known for its booming economy, vibrant pop culture, and rich history. It is home to K-pop, Samsung, and delicious cuisine like kimchi. The country also faces tensions with North Korea and strives for reunification.
    CPU times: total: 0 ns
    Wall time: 4.01 ms
</pre>
