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

# Prompt Caching

- Author: [PangPangGod](https://github.com/pangpanggod)
- Peer Review : [byoon](https://github.com/acho98), [Wonyoung Lee](https://github.com/BaBetterB)
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/05-Prompt-Caching.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/05-Prompt-Caching.ipynb)
## Overview

Prompt caching is a powerful feature that optimizes API usage by enabling resumption from specific prefixes in your prompts.  
This method greatly reduces processing time and costs for repetitive tasks or prompts with consistent components.

Prompt Caching is especially useful for this situations:

- Prompts with many examples
- Large amounts of context or background information
- Repetitive tasks with consistent instructions
- Long multi-turn conversations

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Fetch Data](#fetch-data)
- [OpenAI](#openai)
- [Anthropic](#anthropic)
- [GoogleAI](#googleai)

### References

- [OpenAI Prompt Caching Documentation](https://platform.openai.com/docs/guides/prompt-caching)
- [Anthropic Prompt Caching Documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Google Gemini API - Context Caching](https://ai.google.dev/gemini-api/docs/caching)
- [LangChain Google Generative AI - ChatGoogleGenerativeAI](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#chatgooglegenerativeai)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials.
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain-core",
        "langchain-openai",
        "langchain-anthropic",
        "langchain-google-genai",
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
        "ANTHROPIC_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Prompt-Caching",
    }
)
```

## Fetch Data

The easiest way to verify prompt caching is by including large amounts of context or background information.  
To demonstrate this, I have provided a simple example using a long document retrieved from Wikipedia.

```python
import urllib.parse
import urllib.request
import json

def fetch_wikipedia_page(title: str, lang: str = "en"):
    """
    Fetch the content of a Wikipedia page using the Wikipedia API.
    
    Args:
        title (str): The title of the Wikipedia page to fetch.
        lang (str): The language code for the Wikipedia (default: "en").
    
    Returns:
        str: The plain text content of the Wikipedia page.
    """
    # Wikipedia API endpoint
    endpoint = f"https://{lang}.wikipedia.org/w/api.php"
    
    # Query parameters
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "explaintext": True
    }
    
    # Encode the parameters and create the URL
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    
    # Send the request and read the response
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    
    # Extract page content
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if "extract" in page:
            return page["extract"]
    
    return "No content found for the given title."
```

```python
# fetch data from wikipedia
title = "World War II"
content = fetch_wikipedia_page(title)
```

## OpenAI

OpenAI Prompt Caching works automatically on all your API requests (no code changes required) and has no additional fees associated with it.  
This can reduce latency by up to **80%** and costs by **50%** for long prompts. Caching is available for prompts containing 1024 tokens or more.

### Models Supporting Prompt Caching

| Model                                    | Text Input Cost | Audio Input Cost |
|------------------------------------------|-----------------|------------------|
| gpt-4o (excludes gpt-4o-2024-05-13 and chatgpt-4o-latest) | 50% less         | n/a              |
| gpt-4o-mini                              | 50% less         | n/a              |
| gpt-4o-realtime-preview                  | 50% less         | 80% less         |
| o1-preview                               | 50% less         | n/a              |
| o1-mini                                  | 50% less         | n/a              |

for detailed reference, please check link below.  
[OpenAI Prompt caching](https://platform.openai.com/docs/guides/prompt-caching)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            #The {content} is sourced from the Wikipedia article mentioned above.
            "You are an assistant who answers questions based on the provided document.\n<document>{content}</document>"
        ),
        (
            "human",
            "{question}"
        )
    ]
)

chain = prompt | llm
first_response = chain.invoke({"content": content,"question":"When did Australia and New Zealand join the war?"})
second_response = chain.invoke({"content": content,"question":"Where did the first battle between Australia, New Zealand, and Japan take place?"})

# You can see only cache read in 'prompt_tokens_details' -> 'cached_tokens' in langchain 0.3.29 OpenAI calls.
print(f"Answer: {first_response.content}")
print(f"Token Usage: {first_response.response_metadata}")
print()
print(f"Caching Answer: {second_response.content}")
print(f"Token Usage: {second_response.response_metadata}")
```

<pre class="custom">Answer: Australia and New Zealand joined World War II shortly after the outbreak of the war in Europe. Both countries declared war on Germany on 3 September 1939, following the United Kingdom's declaration of war on Germany after the invasion of Poland.
    Token Usage: {'token_usage': {'completion_tokens': 49, 'prompt_tokens': 17389, 'total_tokens': 17438, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}
    
    Caching Answer: The first battle between Australia, New Zealand, and Japan took place at the Battle of Rabaul, which occurred in January 1942. This battle was part of the broader conflict in the Pacific during World War II.
    Token Usage: {'token_usage': {'completion_tokens': 46, 'prompt_tokens': 17395, 'total_tokens': 17441, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 17152}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}
</pre>

## Anthropic

Anthropic Prompt Caching provides the following token limits for caching:
- **1024 tokens** for Claude 3.5 Sonnet and Claude 3 Opus
- **2048 tokens** for Claude 3.5 Haiku and Claude 3 Haiku

**[Note]**
- Shorter prompts cannot be cached, even if marked with ```cache_control```.
- The cache has a **5-minute time to live (TTL)**. Currently, ```ephemeral``` is the only supported cache type, corresponding to this 5-minute lifetime.

### Models Supporting Prompt Caching
- Claude 3.5 Sonnet
- Claude 3.5 Haiku
- Claude 3 Haiku
- Claude 3 Opus

While it has the drawback of requiring adherence to the Anthropic Message Style, a key advantage of Anthropic Prompt Caching is that it enables caching with fewer tokens.  

For detailed reference, please check link below.   
[Anthropic Prompt Caching Documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)


```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model = "claude-3-5-haiku-latest")

messages = [
    {
        "role": "system",
        "content": [{
            "type": "text",
            #The {content} is sourced from the Wikipedia article mentioned above.
            "text": f"You are an assistant who answers questions based on the provided document.\n<document>{content}</document>", 
            "cache_control": {"type": "ephemeral"}
        }]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Why was Yugoslavia invaded?"}]
    }
]

first_response = llm.invoke(messages)

print(f"Answer: {first_response.content}")
# You can see cache read in 'input_token_details' -> 'cache_creation_tokens' or 'cache_read_input_tokens'.
print(f"Token Usage: {first_response.response_metadata}")
```

<pre class="custom">Answer: According to the document, Yugoslavia was invaded by Germany and Italy as part of their broader operations in the Balkans. The specific details are mentioned in this passage:
    
    "By late March 1941, Bulgaria and Yugoslavia signed the Tripartite Pact; however, the Yugoslav government was overthrown two days later by pro-British nationalists. Germany and Italy responded with simultaneous invasions of both Yugoslavia and Greece, commencing on 6 April 1941; both nations were forced to surrender within the month."
    
    The invasion appears to have been a response to the overthrow of the government that had previously signed the Tripartite Pact. Germany and Italy saw this as a threat to their strategic interests in the region and quickly moved to occupy Yugoslavia. After the invasion, partisan warfare broke out against the Axis occupation, which continued until the end of the war.
    Token Usage: {'id': 'msg_01N6edkmZ6NGT5RmZs85uFya', 'model': 'claude-3-5-haiku-20241022', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 18837, 'cache_read_input_tokens': 0, 'input_tokens': 12, 'output_tokens': 186}}
</pre>

```python
messages = [
    {
        "role": "system",
        "content": [{
            "type": "text",
            #The {content} is sourced from the Wikipedia article mentioned above.
            "text": f"You are an assistant who answers questions based on the provided document.\n<document>{content}</document>", 
            "cache_control": {"type": "ephemeral"}
        }]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Where was invaded after Yugoslavia?"}]
    }
]

second_response = llm.invoke(messages)

print(f"Answer: {second_response.content}")
# You can see cache read in 'input_token_details' -> 'cache_creation_tokens' or 'cache_read_input_tokens'.
print(f"Token Usage: {second_response.response_metadata}")
```

<pre class="custom">Answer: According to the document, after Yugoslavia was invaded by Germany and Italy, Greece was also invaded. Specifically, the text states: "Germany and Italy responded with simultaneous invasions of both Yugoslavia and Greece, commencing on 6 April 1941; both nations were forced to surrender within the month."
    Token Usage: {'id': 'msg_019t8wXVpXpYbasNRb7WBrsv', 'model': 'claude-3-5-haiku-20241022', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 18837, 'input_tokens': 13, 'output_tokens': 66}}
</pre>

## GoogleAI

Google refers to it as Context Caching, not Prompt Caching, and it is primarily used for analyzing various data types, such as code analysis, large document collections, long videos, and multiple audio files.

Therefore, we will demonstrate how to use caching in ```google.generativeai``` through ```ChatGoogleGenerativeAI``` from ```langchain_google_genai```.

For more information, please refer to the following links:  
- [Google Gemini API - Context Caching](https://ai.google.dev/gemini-api/docs/caching)
- [LangChain Google Generative AI - ChatGoogleGenerativeAI](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#chatgooglegenerativeai)

### Fetching Data For GoogleAI Context Caching

At least **32,768** tokens are required for Prompt Caching (which Google refers to as Context Caching).   
Therefore, we decided to implement this in a simple way and demonstrate its usage by including three lengthy Wikipedia documents.

```python
longest_featured_list_in_wikipedia = "List of Falcon 9 and Falcon Heavy launches"
falcon_wiki = fetch_wikipedia_page(longest_featured_list_in_wikipedia)

longest_biography_in_wikipedia = "Vladimir Putin"
putin_wiki = fetch_wikipedia_page(longest_biography_in_wikipedia)

python_wiki_page = "Python (programming language)"
python_wiki = fetch_wikipedia_page(python_wiki_page)
```

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import caching
import datetime

cache = caching.CachedContent.create(
    model='models/gemini-1.5-flash-001',
    display_name='wikipedia-document-pages', # used to identify the cache.
    system_instruction=(
        'You are an expert in analyze very long text, and your job is to answer '
        'the user\'s query based on the video file you have access to.'
    ), # if long, complex system instruction needed, you can provide with this format.
    contents=[falcon_wiki, putin_wiki, python_wiki], # you can pass each documents in list format.
    ttl=datetime.timedelta(minutes=5),
)
```

```python
print(cache) # When caching, the model name provided must be the same when creating an instance of ChatGoogleGenerativeAI.
```

<pre class="custom">CachedContent(
        name='cachedContents/7odha6ycbqsi',
        model='models/gemini-1.5-flash-001',
        display_name='wikipedia-document-pages',
        usage_metadata={
            'total_token_count': 43394,
        },
        create_time=2025-02-04 17:59:43.621411+00:00,
        update_time=2025-02-04 17:59:43.621411+00:00,
        expire_time=2025-02-04 18:04:30.411653+00:00
    )
</pre>

```python
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-001", cached_content=cache.name) # provide cache's name parameter for trackability.
response = llm.invoke("How many Falcon Rockets launch conducted in 2025?")

print(response.content) 
print(response.usage_metadata) # you can see 'input_token_details' actually works!
```

<pre class="custom">The text states that as of February 4th, 2025, SpaceX has conducted **15** Falcon family launches in 2025. All of these launches were conducted using the Falcon 9 rocket, with no Falcon Heavy launches. 
    
    {'input_tokens': 43408, 'output_tokens': 53, 'total_tokens': 43461, 'input_token_details': {'cache_read': 43394}}
</pre>
