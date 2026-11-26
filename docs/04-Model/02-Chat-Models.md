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

# Chat Models

- Author: [PangPangGod](https://github.com/pangpanggod)
- Peer Review : [YooKyung Jeon](https://github.com/sirena1)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/02-Chat-Models.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/02-Chat-Models.ipynb)
## Overview

This tutorial covers an explanation of various Chat models (OpenAI, Anthropic, etc.) along with brief usage examples.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [OpenAI](#openai)
- [Anthropic](#anthropic)
- [Perplexity](#perplexity)
- [Together AI](#together-ai)
- [Cohere](#cohere)
- [Upstage](#upstage)
- [Open LLM Leaderboard](#open-llm-leaderboard)
- [Vellum LLM Leaderboard](#vellum-llm-leaderboard)

### References

- [OpenAI Model Specifications](https://platform.openai.com/docs/models)
- [LangChain ChatOpenAI API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [Anthropic Model Specifications](https://docs.anthropic.com/en/docs/about-claude/models)
- [LangChain ChatAnthropic API reference](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html)
- [Perplexity Model Cards](https://docs.perplexity.ai/guides/model-cards)
- [LangChain ChatPerplexity API reference](https://api.python.langchain.com/en/latest/community/chat_models/langchain_community.chat_models.perplexity.ChatPerplexity.html)
- [Together AI Model Specifications](https://api.together.xyz/models)
- [LangChain ChatTogether API reference](https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html)
- [Cohere Model Specifications](https://docs.cohere.com/docs/models)
- [LangChain ChatCohere API reference](https://python.langchain.com/api_reference/cohere/chat_models/langchain_cohere.chat_models.ChatCohere.html)
- [Upstage Model Specifications](https://console.upstage.ai/docs/capabilities/chat)
- [LangChain ChatUpstage API reference](https://python.langchain.com/api_reference/upstage/chat_models/langchain_upstage.chat_models.ChatUpstage.html)
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [Vellum LLM Leaderboard](https://www.vellum.ai/llm-leaderboard)
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

<pre class="custom">
    [notice] A new release of pip is available: 24.0 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_community",
        "langchain_together",
        "langchain_cohere",
        "langchain_upstage",
    ],
    verbose=False,
    upgrade=False,
)
```

If you want to get automated tracing of your model calls you can also set your LangSmith API key by uncommenting below code:

```python
# Set environment variables

# from langchain_opentutorial import set_env

# set_env(
#     {
#         "LANGCHAIN_API_KEY": "",
#         "LANGCHAIN_TRACING_V2": "true",
#         "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
#         "LANGCHAIN_PROJECT": "01-Chat-Models",
#     }
# )
```

## OpenAI

OpenAI is an AI research and deployment company based in San Francisco, dedicated to ensuring that artificial general intelligence benefits all of humanity. Models include the GPT series of language models, such as **GPT-4** and **GPT-4o**, as well as the **DALL·E** series for image generation.

### Model Description

| Model             | Description                                                                                 | Context Length   | Max Output Tokens | Training Data        |
|--------------------|---------------------------------------------------------------------------------------------|------------------|-------------------|----------------------|
| gpt-4o            | A versatile, high-intelligence flagship model, cheaper and faster than GPT-4 Turbo.        | 128,000 tokens   | 16,384 tokens     | Up to October 2023   |
| chatgpt-4o-latest | The continuously updated version of GPT-4o used in ChatGPT.                                 | 128,000 tokens   | 16,384 tokens     | Continuously updated |
| gpt-4o-mini       | A smaller, faster model with better performance than GPT-3.5 Turbo.                         | 128,000 tokens   | 16,384 tokens     | Up to October 2023   |
| gpt-4-turbo       | The latest GPT-4 Turbo model with vision, JSON mode, and function calling capabilities.      | 128,000 tokens   | 4,096 tokens      | Up to December 2023  |
| gpt-4o-realtime   | A beta model optimized for real-time API use with audio and text inputs.                    | 128,000 tokens   | 4,096 tokens      | Up to October 2023   |
| gpt-4o-audio      | A beta model capable of handling audio inputs and outputs via the Chat Completions API.     | 128,000 tokens   | 16,384 tokens     | Up to October 2023   |
| gpt-3.5-turbo     | Optimized for chat and non-chat tasks with natural language and code generation capabilities.| 16,385 tokens    | 4,096 tokens      | Up to September 2021 |

OpenAI offers a variety of model options.  
A detailed specification of these models can be found at the following link:  
[OpenAI Model Specifications](https://platform.openai.com/docs/models)

### Basic Model Options

The basic API options are as follows:

- ```model_name``` : ```str```  
  This option allows you to select the applicable model. can be aliased as ```model```.

- ```temperature``` : ```float``` = 0.7    
  This option sets the sampling ```temperature```. Values can range between 0 and 2. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make the output more focused and deterministic.

- ```max_tokens``` : ```int``` | ```None``` = ```None```    
  Specifies the maximum number of tokens to generate in the chat completion. This option controls the length of text the model can generate in one instance.

Detailed information about the available API options can be found [here](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html).

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI

# Create a ChatOpenAI object
model = ChatOpenAI(
    model_name="gpt-4o-mini",  # Model name (can be aliased as 'model')
    temperature=0,
)
```

The code provided assumes that your ```OPENAI_API_KEY``` is set in your environment variables. If you would like to manually specify your API key and also choose a different model, uncomment following section before using the code:

```python
# model = ChatOpenAI(temperature=0, api_key="YOUR_API_KEY_HERE", model="gpt-4o-mini")
```

```python
query = "Tell me one joke about Computer Science"

# Stream the response instead of invoking it directly
response = model.stream(query)

# Print the streamed response token by token
for token in response:
    print(token.content, end="", flush=True)
```

<pre class="custom">Why do programmers prefer dark mode?
    
    Because light attracts bugs!</pre>

## Anthropic

Anthropic is an AI safety and research company based in San Francisco, dedicated to building reliable, interpretable, and steerable AI systems.   
Their primary offering is the **Claude family** of large language models, including **Claude 3.5 Sonnet** and **Claude 3.5 Haiku**, designed for various applications such as reasoning, coding, and multilingual tasks.

### Model Description

| Model              | Description                                            | Context Length   | Max Output Tokens | Training Data        |
|--------------------|--------------------------------------------------------|------------------|-------------------|----------------------|
| Claude 3.5 Sonnet  | The most intelligent model in the Claude family.       | 200,000 tokens   | 8,192 tokens      | Up to April 2024     |
| Claude 3.5 Haiku   | The fastest model with blazing speed.                  | 200,000 tokens   | 8,192 tokens      | Up to July 2024      |
| Claude 3 Opus      | Powerful model for highly complex tasks.               | 200,000 tokens   | 4,096 tokens      | Up to August 2023    |
| Claude 3 Sonnet    | Balanced model offering strong utility for scaled deployments. | 200,000 tokens   | 4,096 tokens      | Up to August 2023    |
| Claude 3 Haiku     | Fastest and most compact model for near-instant responsiveness. | 200,000 tokens   | 4,096 tokens      | Up to August 2023    |

A detailed specification of these models can be found at the following link:  
[Anthropic Model Specifications](https://docs.anthropic.com/en/docs/about-claude/models)

### Basic Model Options

The basic API options are as follows:

- ```model_name``` : ```str```  
  This option allows you to select the applicable model. can be aliased as ```model```.

- ```temperature``` : ```float``` = 0.7    
  This option sets the sampling ```temperature```. Values can range between 0 and 2. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make the output more focused and deterministic.

- ```max_tokens``` : ```int``` | ```None``` = ```None```    
  Specifies the maximum number of tokens to generate in the chat completion. This option controls the length of text the model can generate in one instance.

Detailed information about the available API options can be found [here](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html).

```python
import getpass
import os

if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

from langchain_anthropic import ChatAnthropic

# Create a ChatAnthropic object
model = ChatAnthropic(
    model_name="claude-3-5-haiku-latest", # Model name (can be aliased as 'model')
    temperature=0,
)
```

The code provided assumes that your ```ANTHROPIC_API_KEY``` is set in your environment variables. If you would like to manually specify your API key and also choose a different model, uncomment following section before using the code:

```python
# model = ChatAnthropic(temperature=0, api_key="YOUR_API_KEY_HERE", model="claude-3-5-haiku-latest")
```

```python
query = "Tell me one joke about Computer Science"

# Stream the response instead of invoking it directly
response = model.stream(query)

# Print the streamed response token by token
for token in response:
    print(token.content, end="", flush=True)
```

<pre class="custom">Here's a classic computer science joke:
    
    Why do programmers prefer dark mode?
    
    Because light attracts bugs! 😄
    
    (This is a play on words, referencing both computer "bugs" (errors) and actual insects being attracted to light.)</pre>

## Perplexity

Perplexity AI is a conversational search engine that integrates advanced large language models (LLMs) to provide direct answers to user queries with source citations.  
Their platform supports the following models, optimized for chat completion tasks, with extended context capabilities:

### Supported Models

| Model                              | Parameter Count | Context Length   | Model Type       |
|------------------------------------|-----------------|------------------|------------------|
| llama-3.1-sonar-small-128k-online  | 8B              | 127,072 tokens   | Chat Completion  |
| llama-3.1-sonar-large-128k-online  | 70B             | 127,072 tokens   | Chat Completion  |
| llama-3.1-sonar-huge-128k-online   | 405B            | 127,072 tokens   | Chat Completion  |

A detailed specification of these models can be found at the following link:  
[Perplexity Model Cards](https://docs.perplexity.ai/guides/model-cards)

### Basic Model Options

The basic API options are as follows:

- ```model``` : str  
  Specifies the language model to use (e.g., **llama-3.1-sonar-small-128k-online**). This determines the performance and capabilities of the response.

- ```temperature``` : float = 0.7  
  Controls the randomness of responses. A value of 0 is deterministic, while 1 allows for the most random outputs.

- ```max_tokens``` : int | None = None    
  Specifies the maximum number of tokens to generate in the chat completion. This option controls the length of text the model can generate in one instance.

For more detailed information about the available API options, visit [Perplexity API Reference](https://api.python.langchain.com/en/latest/community/chat_models/langchain_community.chat_models.perplexity.ChatPerplexity.html).

```python
import getpass
import os

if not os.environ.get("PPLX_API_KEY"):
  os.environ["PPLX_API_KEY"] = getpass.getpass("Enter API key for Perplexity: ")

from langchain_community.chat_models import ChatPerplexity

# Create a ChatPerplexity object
model = ChatPerplexity(
    model="llama-3.1-sonar-large-128k-online",
    temperature=0,
)
```

The code provided assumes that your ```PPLX_API_KEY``` is set in your environment variables. If you would like to manually specify your API key and also choose a different model, uncomment following section before using the code:

```python
# model = ChatPerplexity(temperature=0, pplx_api_key="YOUR_API_KEY_HERE", model="llama-3.1-sonar-large-128k-online")
```

```python
# print out response
response = model.invoke("Who won the 2024 Nobel Prize in Literature?")

print(response.content)
print()
# ChatPerplexity stores the sources of knowledge and information in the additional_kwargs["citations"] field.
for i, citation in enumerate(response.additional_kwargs["citations"]):
    print(f"[{i+1}] {citation}")
```

<pre class="custom">The 2024 Nobel Prize in Literature was awarded to South Korean author Han Kang. She was recognized "for her intense poetic prose that confronts historical traumas and exposes the fragility of human life"[1][3][5].
    
    [1] https://now.uiowa.edu/news/2024/10/international-writing-program-participant-wins-2024-nobel-prize-literature
    [2] https://www.nobelprize.org/prizes/literature/2024/prize-announcement/
    [3] https://www.weforum.org/stories/2024/10/nobel-prize-winners-2024/
    [4] https://www.nobelprize.org/prizes/literature/2024/han/facts/
    [5] https://en.wikipedia.org/wiki/2024_Nobel_Prize_in_Literature
</pre>

## Together AI

Together AI is a San Francisco-based company specializing in decentralized cloud services for training and deploying generative AI models. Founded in 2022, they offer a cloud platform that enables researchers, developers, and organizations to train, fine-tune, and run AI models efficiently at scale. Their services include GPU clusters featuring NVIDIA GB200, H200, and H100, and they contribute to open-source AI research, models, and datasets to advance the field.

### Together Inference  
- Offers the fastest inference stack in the industry, up to 4x faster than vLLM.
- Operates at 11x lower cost compared to **GPT-4** when using **Llama-3 70B**.
- Features auto-scaling capabilities that adjust capacity based on API request volume.

### Together Custom Models  
- Supports customized AI model training and fine-tuning.
- Incorporates cutting-edge optimization technologies like FlashAttention-3.
- Ensures full ownership of trained models.

### Performance Optimization  
- Proprietary inference engine integrating FlashAttention-3 kernels and custom kernels.
- Implements speculative decoding algorithms like Medusa and SpecExec.
- Employs unique quantization techniques for maximum accuracy and performance.

### Security and Privacy
- User data is not used for training new models without explicit consent.
- Provides users with complete control over data storage.

### Supported Models  
- Supports over 200 open-source models, including Google Gemma, Meta's **Llama 3.3**, **Qwen2.5**, and **Mistral**/**Mixtral** from Mistral AI.
- Enables multimodal AI models to process various types of data.
- A detailed specification of these models can be found at the following link:  
  [Together AI Models](https://api.together.xyz/models)

### Basic Model Options

The basic API options are as follows:

- ```model_name``` : ```str```  
  This option allows you to select the applicable model. can be aliased as ```model```.

- ```temperature``` : ```float``` = 0.7    
  This option sets the sampling ```temperature```. Values can range between 0 and 2. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make the output more focused and deterministic.

- ```max_tokens``` : ```int``` | ```None``` = ```None```    
  Specifies the maximum number of tokens to generate in the chat completion. This option controls the length of text the model can generate in one instance.

Detailed information about the available API options can be found [here](https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html).


```python
import getpass
import os

if not os.environ.get("TOGETHER_API_KEY"):
  os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together AI: ")

from langchain_together.chat_models import ChatTogether

# Create a ChatPerplexity object
model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
)
```

The code provided assumes that your ```TOGETHER_API_KEY``` is set in your environment variables. If you would like to manually specify your API key and also choose a different model, uncomment following section before using the code:

```python
# model = ChatTogether(temperature=0, api_key="YOUR_API_KEY_HERE", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
```

```python
query = "What is the biggest Animal on The planet?"

# Stream the response instead of invoking it directly
response = model.stream(query)

# Print the streamed response token by token
for token in response:
    print(token.content, end="", flush=True)
```

<pre class="custom">The biggest animal on the planet is the **blue whale** (Balaenoptera musculus). On average, an adult blue whale can grow up to:
    
    * 82 feet (25 meters) in length
    * Weigh around 150-170 tons (136,000-152,000 kilograms)
    
    To put that into perspective, that's equivalent to the weight of about 50 elephants or a large building. Blue whales are not only the largest animals on Earth, but they are also the largest known animals to have ever existed on our planet.
    
    These massive creatures can be found in all of the world's oceans, from the Arctic to the Antarctic, and are known for their distinctive blue-gray color and massive size. Despite their enormous size, blue whales are incredibly streamlined and can swim at speeds of up to 30 miles (48 kilometers) per hour.
    
    It's worth noting that while blue whales are the largest animals, there are other large animals on the planet, such as fin whales, humpback whales, and African elephants, which are also impressive in their own right. However, the blue whale remains the largest of them all.</pre>

## Cohere

Cohere is a leading AI company specializing in enterprise AI solutions, enabling businesses to easily adopt and utilize AI technologies through advanced large language models (LLMs).  
Their platform is tailored for natural language processing tasks, providing scalable and efficient tools for real-world applications.

- **Founded:** 2020  
- **Key Investors:** Inovia Capital, NVIDIA, Oracle, Salesforce Ventures  
- **Series C Funding:** Raised $270 million  
- **Mission:** To provide an AI platform tailored for enterprise needs  

### Supported Models 
| Model               | Description                                                                                           | Context Length   | Max Output Tokens |
|---------------------|-------------------------------------------------------------------------------------------------------|------------------|-------------------|
| command-r-7b        | A small, fast update of the Command R+ model, excelling at RAG, tool use, agents, and multi-step reasoning. | 128,000 tokens   | 4,000 tokens      |
| command-r-plus      | An instruction-following conversational model excelling in RAG, tool use, and multi-step reasoning.   | 128,000 tokens   | 4,000 tokens      |
| command-r           | A conversational model designed for high-quality language tasks and complex workflows like RAG and coding. | 128,000 tokens   | 4,000 tokens      |
| command             | An instruction-following conversational model for high-quality tasks, more reliable and longer context than base models. | 4,000 tokens     | 4,000 tokens      |
| command-nightly     | Nightly version of the command model with experimental and regularly updated features. Not for production use. | 128,000 tokens   | 4,000 tokens      |
| command-light       | A smaller, faster version of the command model, maintaining near-equal capability with improved speed. | 4,000 tokens     | 4,000 tokens      |
| command-light-nightly | Nightly version of the command-light model, experimental and regularly updated. Not for production use. | 4,000 tokens     | 4,000 tokens      |
| c4ai-aya-expanse-8b | A highly performant 8B multilingual model serving 23 languages, designed for superior monolingual performance. | 8,000 tokens     | 4,000 tokens      |
| c4ai-aya-expanse-32b| A highly performant 32B multilingual model serving 23 languages, designed for superior monolingual performance. | 128,000 tokens   | 4,000 tokens      |

- A detailed specification of cohere's models can be found at the following link:  
  [Cohere Models](https://docs.cohere.com/docs/models)

### Basic Model Options

The basic API options are as follows:

- ```model_name``` : ```str```  
  This option allows you to select the applicable model. can be aliased as ```model```.

- ```temperature``` : ```float``` = 0.7    
  This option sets the sampling ```temperature```. Values can range between 0 and 2. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make the output more focused and deterministic.

- ```max_tokens``` : ```int``` | ```None``` = ```None```    
  Specifies the maximum number of tokens to generate in the chat completion. This option controls the length of text the model can generate in one instance.

Detailed information about the available API options can be found [here](https://python.langchain.com/api_reference/cohere/chat_models/langchain_cohere.chat_models.ChatCohere.html).

```python
import getpass
import os

if not os.environ.get("COHERE_API_KEY"):
  os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

from langchain_cohere import ChatCohere

# Create a ChatCohere object
model = ChatCohere(
    model_name="command-r7b-12-2024", # can be alisaed as 'model'
    temperature=0,
)
```

The code provided assumes that your ```COHERE_API_KEY``` is set in your environment variables. If you would like to manually specify your API key and also choose a different model, uncomment following section before using the code:

```python
# model = ChatCohere(temperature=0, cohere_api_key="YOUR_API_KEY_HERE", model="command-r7b-12-2024")
```

```python
query = "What are the Biggest hit songs of Earth, Wind & Fire?"

# Stream the response instead of invoking it directly
response = model.stream(query)

# Print the streamed response token by token
for token in response:
    print(token.content, end="", flush=True)
```

<pre class="custom">Earth, Wind & Fire, the iconic American band, has had numerous hits throughout their illustrious career, spanning multiple genres such as R&B, soul, funk, jazz, disco, pop, and Afro pop. Here are some of their biggest hit songs:
    
    - "September": Released in 1978, "September" is arguably the band's most recognizable and popular song. It topped the US Billboard Hot R&B Songs chart and peaked at number eight on the Billboard Hot 100. The song's catchy melody and uplifting lyrics have made it a timeless classic, often played at celebrations and events.
    
    - "Shining Star": This was Earth, Wind & Fire's first single to reach number one on the Billboard Hot 100 in 1975. The song won a Grammy Award for Best R&B Performance by a Duo or Group with Vocals. "Shining Star" is known for its powerful vocals and inspiring message.
    
    - "Boogie Wonderland": A collaboration with The Emotions, this disco-funk track became a massive hit in 1979. It reached number six on the Billboard Hot 100 and is remembered for its infectious groove and energetic vocals.
    
    - "After the Love Has Gone": This ballad showcases the band's versatility and won a Grammy Award for Best R&B Song in 1980. It peaked at number two on the Billboard Hot 100 and is considered one of the band's most emotionally powerful songs.
    
    - "Let's Groove": Released in 1981, this funk and disco-influenced song became a commercial success, reaching number three on the Billboard Hot 100. "Let's Groove" is known for its catchy rhythm and danceable vibe.
    
    - "Fantasy": From their 1977 album "All 'N All," "Fantasy" is another fan favorite. It peaked at number twelve on the Billboard Hot 100 and is characterized by its dreamy atmosphere and smooth vocals.
    
    - "Got to Get You into My Life": A cover of the Beatles song, Earth, Wind & Fire's version became a hit in its own right in 1978. It reached number nine on the Billboard Hot 100 and won a Grammy Award for Best Instrumental Arrangement Accompanying Vocalist(s).
    
    These songs are just a selection of Earth, Wind & Fire's extensive catalog of hits, showcasing their incredible musical range and enduring popularity.</pre>

## Upstage

Upstage is a South Korean startup specializing in artificial intelligence (AI) technologies, particularly large language models (LLMs) and document AI.  
Their solutions are designed to deliver cost-efficient, high-performance AI capabilities across various industries.

- **Key Product: Solar LLM**  
  Upstage's flagship large language model known for its speed, efficiency, and scalability. Utilizing Depth-Up Scaling (DUS) technology, Solar LLM maximizes performance and is seamlessly integrated into platforms like Amazon SageMaker JumpStart via API.

- **Document AI Pack**  
  A comprehensive document processing solution powered by advanced OCR technology. This tool accurately extracts and digitizes essential information from complex documents.

- **AskUp Seargest**  
  An upgraded version of the AskUp chatbot, offering personalized search and recommendation services, building upon the integration with ChatGPT.

Upstage provides cutting-edge tools for enterprises to enhance automation, streamline workflows, and deliver AI-powered insights.

### Supported Models 

| Model        | Description                                                                                           | Context Length   | Training Data        |
|--------------|-------------------------------------------------------------------------------------------------------|------------------|----------------------|
| solar-pro    | An enterprise-grade LLM designed for exceptional instruction-following and processing structured formats like HTML and Markdown. It excels in multilingual performance in English, Korean, and Japanese, with domain expertise in Finance, Healthcare, and Legal. | 32,768 tokens    | Up to May 2024       |
| solar-mini   | A compact 10.7B parameter LLM for businesses seeking AI solutions. Solar Mini is an instruction-following conversational model supporting English, Korean, and Japanese. It excels at fine-tuning, providing seamless integration and high-quality language processing. | 32,768 tokens    | Up to December 2023  |
| solar-mini-ja| Solar Mini with enhanced capabilities in Japanese language processing. It is an instruction-following conversational model supporting Japanese as well as English and Korean. | 32,768 tokens    | Up to December 2023  |

- A detailed specification of Upstage's models can be found at the following link:  
  [Upstage Models](https://console.upstage.ai/docs/capabilities/chat)

### Basic Model Options

The basic API options are as follows:

- ```model_name``` : ```str```  
  This option allows you to select the applicable model. can be aliased as ```model```.

- ```temperature``` : ```float``` = 0.7    
  This option sets the sampling ```temperature```. Values can range between 0 and 2. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make the output more focused and deterministic.

- ```max_tokens``` : ```int``` | ```None``` = ```None```    
  Specifies the maximum number of tokens to generate in the chat completion. This option controls the length of text the model can generate in one instance.

Detailed information about the available API options can be found [here](https://python.langchain.com/api_reference/upstage/chat_models/langchain_upstage.chat_models.ChatUpstage.html).

```python
import getpass
import os

if not os.environ.get("UPSTAGE_API_KEY"):
  os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Enter API key for Upstage: ")

from langchain_upstage import ChatUpstage 

# Create a ChatUpstage object
model = ChatUpstage(
    model_name="solar-mini", # can be alisaed as 'model'
    temperature=0,
)
```

The code provided assumes that your ```UPSTAGE_API_KEY``` is set in your environment variables. If you would like to manually specify your API key and also choose a different model, uncomment following section before using the code:

```python
# model = ChatUpstage(temperature=0, upstage_api_key="YOUR_API_KEY_HERE", model="solar-mini")
```

```python
query = "Tell me one joke about Computer Science"

# Stream the response instead of invoking it directly
response = model.stream(query)

# Print the streamed response token by token
for token in response:
    print(token.content, end="", flush=True)
```

<pre class="custom">Why did the computer scientist go broke?
    
    He invested in a company that was developing a new programming language, but it turned out to be a "dead end" language.</pre>

## Open LLM Leaderboard

The Open LLM Leaderboard is a community-driven platform hosted by Hugging Face that tracks, ranks, and evaluates open-source large language models (LLMs) and chatbots. It provides datasets, score results, queries, and collections for various models, enabling users to compare performance across different benchmarks.

By utilizing the Open LLM Leaderboard, you can identify high-performing models suitable for integration with platforms like Hugging Face and Ollama, facilitating seamless deployment and interaction with LLMs.

For more information, visit the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).


## Vellum LLM Leaderboard

The Vellum LLM Leaderboard is a platform that compares leading commercial and open-source large language models (LLMs) based on capabilities, pricing, and context window sizes. It provides insights into model performance across various tasks, including multitask reasoning, coding, and mathematics, assisting users in selecting the most suitable LLM for their specific needs.

By utilizing the Vellum LLM Leaderboard, you can identify high-performing models suitable for integration with platforms like Hugging Face and Ollama, facilitating seamless deployment and interaction with LLMs.

For more information, visit the [Vellum LLM Leaderboard](https://www.vellum.ai/llm-leaderboard).
