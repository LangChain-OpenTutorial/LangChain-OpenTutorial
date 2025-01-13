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
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/02-Cache.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/02-Cache.ipynb)

## Overview

`LangChain` provides optional caching layer for LLMs.

This is useful for two reasons:
- When requesting the same completions multiple times, it can **reduce the number of API calls** to the LLM provider and thus save costs.
- By **reduing the number of API calls** to the LLM provider, it can **improve the running time of the application.**

In this tutorial, we will use `gpt-4o-mini` OpenAI API and utilize two kinds of cache, **InMemoryCache** and **SQLite Cache** .  
At end of each section we will compare wall times between before and after caching.

Optionally, we will use local LLM served with VLLM.

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [InMemoryCache](#inmemorycache)
- [SQlite Cache](#sqlite-cache)
- [(Optional) With local model](#optional-with-local-model)
- [(Optional) InMemoryCache + Local LLM](#optional-inmemorycache--local-llm)
- [(Optional) SQLite Cache + Local LLM](#optional-sqlite-cache--local-llm)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
!pip install langchain-opentutorial
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


load_dotenv()
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

<pre class="custom">c:\Users\harry\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-6kFp1u2S-py3.11\Lib\site-packages\langsmith\client.py:256: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
      warnings.warn(
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=cc32ca74-11e6-466b-a13a-48d2b26101f0,id=cc32ca74-11e6-466b-a13a-48d2b26101f0; trace=cc32ca74-11e6-466b-a13a-48d2b26101f0,id=02c668f8-d14d-4ded-8ec8-8a89c9cbb70c; trace=cc32ca74-11e6-466b-a13a-48d2b26101f0,id=fcba4fb1-5c84-48db-95cb-f7dbe006208c
</pre>

    South Korea is a highly developed country in East Asia known for its technological advancements, vibrant culture, and economic prosperity. It has a rich history, beautiful landscapes, and a strong emphasis on education. The country is also a major player in the global economy and a leading exporter of electronics, automobiles, and other goods.
    CPU times: total: 46.9 ms
    Wall time: 1.2 s
    

    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=cc32ca74-11e6-466b-a13a-48d2b26101f0,id=cc32ca74-11e6-466b-a13a-48d2b26101f0; trace=cc32ca74-11e6-466b-a13a-48d2b26101f0,id=fcba4fb1-5c84-48db-95cb-f7dbe006208c
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=24f75e9f-19f0-40c8-a65b-654a96e640a7,id=24f75e9f-19f0-40c8-a65b-654a96e640a7; trace=24f75e9f-19f0-40c8-a65b-654a96e640a7,id=402fd17a-3e29-41f2-b67e-5d244065d264; trace=24f75e9f-19f0-40c8-a65b-654a96e640a7,id=818d688d-545c-460f-9882-214d23817033
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=24f75e9f-19f0-40c8-a65b-654a96e640a7,id=24f75e9f-19f0-40c8-a65b-654a96e640a7; trace=24f75e9f-19f0-40c8-a65b-654a96e640a7,id=818d688d-545c-460f-9882-214d23817033
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=ecdebc40-e7b4-4e14-a363-c203c90acaef,id=ecdebc40-e7b4-4e14-a363-c203c90acaef; trace=ecdebc40-e7b4-4e14-a363-c203c90acaef,id=e99e2982-6a8c-468e-84eb-88615067c282; trace=ecdebc40-e7b4-4e14-a363-c203c90acaef,id=651e4a1c-1ec3-4105-a02d-7ae26df842f7
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=9a968827-8fb0-441e-a0b9-d872ef071914,id=9a968827-8fb0-441e-a0b9-d872ef071914; trace=9a968827-8fb0-441e-a0b9-d872ef071914,id=0a2ad228-9e97-4dd5-8e0f-6ba3b645be87; trace=9a968827-8fb0-441e-a0b9-d872ef071914,id=16c41a87-b694-4890-a578-b779acc07501
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=4d6bc72d-f969-4468-ad59-f2f093809d96,id=4d6bc72d-f969-4468-ad59-f2f093809d96; trace=4d6bc72d-f969-4468-ad59-f2f093809d96,id=22b4a960-1c1a-471b-b46a-f0a95a1924af; trace=4d6bc72d-f969-4468-ad59-f2f093809d96,id=a860a43c-3987-4391-97d0-eada9a368593
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=4d6bc72d-f969-4468-ad59-f2f093809d96,id=4d6bc72d-f969-4468-ad59-f2f093809d96; trace=4d6bc72d-f969-4468-ad59-f2f093809d96,id=a860a43c-3987-4391-97d0-eada9a368593
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=d92bbf71-3e55-4f29-98d1-adb835281a4d,id=d92bbf71-3e55-4f29-98d1-adb835281a4d; trace=d92bbf71-3e55-4f29-98d1-adb835281a4d,id=621f0537-c923-401c-bd55-71074adf5a45; trace=d92bbf71-3e55-4f29-98d1-adb835281a4d,id=88316077-6495-4235-8221-5fadc50f6baa
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=d92bbf71-3e55-4f29-98d1-adb835281a4d,id=d92bbf71-3e55-4f29-98d1-adb835281a4d; trace=d92bbf71-3e55-4f29-98d1-adb835281a4d,id=88316077-6495-4235-8221-5fadc50f6baa
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=6986d559-9583-4297-a7bd-30fde6cc9da1,id=6986d559-9583-4297-a7bd-30fde6cc9da1; trace=6986d559-9583-4297-a7bd-30fde6cc9da1,id=f11d8686-dca8-4c6d-b878-88695589c105; trace=6986d559-9583-4297-a7bd-30fde6cc9da1,id=19da0ec1-4f3b-4f0d-bb1a-65b1cdb99bc6
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=6986d559-9583-4297-a7bd-30fde6cc9da1,id=6986d559-9583-4297-a7bd-30fde6cc9da1; trace=6986d559-9583-4297-a7bd-30fde6cc9da1,id=19da0ec1-4f3b-4f0d-bb1a-65b1cdb99bc6
    Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError: Authentication failed for https://api.smith.langchain.com/runs/multipart. HTTPError('401 Client Error: Unauthorized for url: https://api.smith.langchain.com/runs/multipart', '{"detail":"Invalid token"}')trace=4ac6b33f-2331-4bdf-950b-43e09d58806a,id=4ac6b33f-2331-4bdf-950b-43e09d58806a; trace=4ac6b33f-2331-4bdf-950b-43e09d58806a,id=81340a1e-99f5-429f-addf-2c567dd31553; trace=4ac6b33f-2331-4bdf-950b-43e09d58806a,id=ae75af79-897a-495a-842c-b8dbd1de11e6
    

## InMemoryCache
First, cache the answer to the same question using InMemoryCache.

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

Note that if we set InMemoryCache again, the cache will be lost and the wall time will increase

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

## SQLite Cache
Now, we cache the answer to the same question by using SQLite cache.

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

Note that if we use SQLite Cache, setting caching again does not delete store cache

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

## (Optional) With local model
In this optional section, we utilize `docker` to serve local LLM model.
Note that this used miniconda to set environment easily.

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

### Device & Serving information - Mac OS
- Device : M2 Macbook Air 15
- RAM : 16GB
- macOS : Sequoia 15.1.1
- Docker Image : 
Build from the [docker image]['https://docs.vllm.ai/en/latest/getting_started/arm-installation.html'] written by official vLLM.

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

## (Optional) InMemoryCache + Local LLM
Same InMemoryCache section above, we set InMemoryCache.

```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Set InMemoryCache
set_llm_cache(InMemoryCache())
```

Invoke chain with local LLM, do note that we print **response** not **response.content**

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

## (Optional) SQLite Cache + Local LLM
Same as SQLite Cache section above, set SQLite Cache.  
Note that we set db name to be **vllm_cache.db** to distinguish from the cache used in SQLite Cache section.

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

Invoke chain with local LLM, again, note that we print **response** not **response.content**

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

```python

```
