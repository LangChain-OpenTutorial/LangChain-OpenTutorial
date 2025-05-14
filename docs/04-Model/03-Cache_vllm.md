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

# Caching VLLM

- Author: [Joseph](https://github.com/XaviereKU)
- Peer Review : [Teddy Lee](https://github.com/teddylee777), [BAEM1N](https://github.com/BAEM1N)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/03-Cache_vllm.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/03-Cache_vllm.ipynb)
## Overview

```LangChain``` provides optional caching layer for LLMs.

This is useful for two reasons:
- When requesting the same completions multiple times, it can **reduce the number of API calls** to the LLM provider and thus save costs.
- By **reduing the number of API calls** to the LLM provider, it can **improve the running time of the application.**

But sometimes you need to deploy your own LLM service, like on-premise system where you cannot reach cloud services.
In this tutorial, we will use ```vllm``` OpenAI compatible API and utilize two kinds of cache, ```InMemoryCache``` and ```SQLiteCache```.  
At end of each section we will compare wall times between before and after caching.

Even though this is a tutorial for local LLM service case, we will remind you about how to use cache with OpenAI API service first.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [InMemoryCache](#inmemorycache)
- [SQlite Cache](#sqlite-cache)
- [Setup Local LLM with VLLM](#setup-local-llm-with-vllm)
- [InMemoryCache + Local VLLM](#inmemorycache--local-vllm)
- [SQLite Cache + Local VLLM](#sqlite-cache--local-vllm)

### References
- [SQLIteCache](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.SQLiteCache.html#langchain_community.cache.SQLiteCache)
- [InMemoryCache](https://python.langchain.com/api_reference/core/caches/langchain_core.caches.InMemoryCache.html)
- [vLLM](https://docs.vllm.ai/en/latest/)
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

<pre class="custom">South Korea, located on the Korean Peninsula, is known for its rich culture, technological advancements, and vibrant economy. It features a mix of traditional heritage and modern innovation, highlighted by K-pop and cuisine.
    CPU times: total: 93.8 ms
    Wall time: 1.09 s
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

## Setup Local LLM with ```VLLM```
vLLM supports various cases, but for the most stable setup we utilize ```docker``` to serve local LLM model with ```vLLM```.

### Device & Serving information - Windows
- CPU : AMD 5600X
- OS : Windows 10 Pro
- RAM : 32 Gb
- GPU : Nividia 3080Ti, 12GB VRAM
- CUDA : 12.6
- Driver Version : 560.94
- Docker Image : nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
- model : Qwen/Qwen2.5-0.5B-Instruct
- Python version : 3.10
- docker run script :
    ```
    docker run -itd --name vllm --gpus all --entrypoint /bin/bash -p 6001:8888 nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
    ```
- vllm serving script : 
    ```
    python3 -m vllm.entrypoints.openai.api_server --model='Qwen/Qwen2.5-0.5B-Instruct' --served-model-name 'qwen-2.5' --port 8888 --host 0.0.0.0 --gpu-memory-utilization 0.80 --max-model-len 4096 --swap-space 1 --dtype bfloat16 --tensor-parallel-size 1 
    ```

```python
from langchain_community.llms import VLLMOpenAI

# create model using OpenAI compatible class VLLMOpenAI
llm = VLLMOpenAI(
    model="qwen-2.5", openai_api_key="EMPTY", openai_api_base="http://localhost:6001/v1"
)

# Generate prompt
prompt = PromptTemplate.from_template(
    "Sumarize about the {country} in about 200 characters"
)

# Create chain
chain = prompt | llm
```

## InMemoryCache + Local VLLM
Same ```InMemoryCache``` section above, we set ```InMemoryCache```.

```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Set InMemoryCache
set_llm_cache(InMemoryCache())
```

Invoke chain with local LLM, do note that we print ```response``` not ```response.content```

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response)
```

<pre class="custom">.
    South Korea is a country in East Asia, with a population of approximately 55.2 million as of 2023. It borders North Korea to the east, Japan to the northeast, and China to the southeast. The country is known for its advanced technology, leading industries, and significant contributions to South Korean culture. It is often referred to as the "Globe and a Couple" due to its diverse landscapes, rich history, and frontiers with neighboring countries. South Korea's economy is growing, with a strong technological sector and a strong economy, making it a significant player on the global stage. Overall, South Korea is a significant global player, with a rich history, advanced technology, and a cultural influence. With its advanced technology and unique culture, South Korea is a fascinating country to explore. Its diverse landscapes, rich history, and remarkable economic performance have made it a popular destination for travelers. South Korea's contribution to the global economy and its strong technological sector have made it a significant player on the world stage. Its cultural influence and trade partnerships have created a unique culture that is hard to replicate elsewhere. South Korea's diverse landscapes, rich history, and technological advancements have made it a popular destination for travelers. Its cultural influence, trade partnerships, and
    CPU times: total: 15.6 ms
    Wall time: 1.03 s
</pre>

Now we invoke chain again, with the same question.

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response)
```

<pre class="custom">.
    South Korea is a country in East Asia, with a population of approximately 55.2 million as of 2023. It borders North Korea to the east, Japan to the northeast, and China to the southeast. The country is known for its advanced technology, leading industries, and significant contributions to South Korean culture. It is often referred to as the "Globe and a Couple" due to its diverse landscapes, rich history, and frontiers with neighboring countries. South Korea's economy is growing, with a strong technological sector and a strong economy, making it a significant player on the global stage. Overall, South Korea is a significant global player, with a rich history, advanced technology, and a cultural influence. With its advanced technology and unique culture, South Korea is a fascinating country to explore. Its diverse landscapes, rich history, and remarkable economic performance have made it a popular destination for travelers. South Korea's contribution to the global economy and its strong technological sector have made it a significant player on the world stage. Its cultural influence and trade partnerships have created a unique culture that is hard to replicate elsewhere. South Korea's diverse landscapes, rich history, and technological advancements have made it a popular destination for travelers. Its cultural influence, trade partnerships, and
    CPU times: total: 0 ns
    Wall time: 2.61 ms
</pre>

## SQLite Cache + Local VLLM
Same as ```SQLiteCache``` section above, set ```SQLiteCache```.  
Note that we set db name to be ```vllm_cache.db``` to distinguish from the cache used in ```SQLiteCache``` section.

```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import os

# Create cache directory
if not os.path.exists("cache"):
    os.makedirs("cache")

# Set SQLiteCache
set_llm_cache(SQLiteCache(database_path="cache/vllm_cache.db"))
```

Invoke chain with local LLM, again, note that we print ```response``` not ```response.content```.

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response)
```

<pre class="custom">.
    
    South Korea, a nation that prides itself on its history, culture, and natural beauty. Known for its bustling cityscapes, scenic valleys, and delicious cuisine. A major player in South East Asia and a global hub for technology, fashion, and entertainment. Home to industries like electronics, automotive, and media. With a strong economy, South Korea is among the top economies in the world, known for its efficient and inclusive societies. A country that has been a significant player in global politics for decades. The country is also home to many influential figures like Kim Jong-un and Kim Jong-un, who have led North Korea and the country’s military. Known for its national sports, including football (soccer), baseball, and gymnastics. South Korea is also home to many museums, art galleries, and historical sites, showcasing the country’s rich cultural heritage. The country is a leader in technology, with many leading companies based in the South Korean capital, Seoul. The South Korean economy, despite global challenges, continues to be resilient and strong, with an average annual growth rate of 2.5%. The country has a diverse population and is known for its high standard of living, which is a source of pride for many South Koreans. With a strong tradition of education
    CPU times: total: 0 ns
    Wall time: 920 ms
</pre>

Now we invoke chain again, with the same question.

```python
%%time
# Invoke chain
response = chain.invoke({"country": "South Korea"})
print(response)
```

<pre class="custom">.
    
    South Korea, a nation that prides itself on its history, culture, and natural beauty. Known for its bustling cityscapes, scenic valleys, and delicious cuisine. A major player in South East Asia and a global hub for technology, fashion, and entertainment. Home to industries like electronics, automotive, and media. With a strong economy, South Korea is among the top economies in the world, known for its efficient and inclusive societies. A country that has been a significant player in global politics for decades. The country is also home to many influential figures like Kim Jong-un and Kim Jong-un, who have led North Korea and the country’s military. Known for its national sports, including football (soccer), baseball, and gymnastics. South Korea is also home to many museums, art galleries, and historical sites, showcasing the country’s rich cultural heritage. The country is a leader in technology, with many leading companies based in the South Korean capital, Seoul. The South Korean economy, despite global challenges, continues to be resilient and strong, with an average annual growth rate of 2.5%. The country has a diverse population and is known for its high standard of living, which is a source of pride for many South Koreans. With a strong tradition of education
    CPU times: total: 0 ns
    Wall time: 3 ms
</pre>
