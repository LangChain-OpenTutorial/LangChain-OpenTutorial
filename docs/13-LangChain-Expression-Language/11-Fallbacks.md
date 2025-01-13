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

# Fallbacks

- Author: [Haseom Shin](https://github.com/IHAGI-c)
- Design: []()
- Peer Review: []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/11-Fallbacks.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/11-Fallbacks.ipynb)

## Overview

This tutorial covers how to implement fallback mechanisms in LangChain applications to handle various types of failures and errors gracefully.

`Fallbacks` are crucial for building robust LLM applications that can handle API errors, rate limits, and other potential failures without disrupting the user experience.

In this tutorial, we will explore different fallback strategies and implement practical examples using multiple LLM providers.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is Fallbacks?](#what-is-fallbacks)
- [How to Handle LLM API Errors](#how-to-handle-llm-api-errors)
- [Introduction to Rate Limit Testing](#introduction-to-rate-limit-testing)
- [Why Handle Rate Limit Errors?](#why-handle-rate-limit-errors)
- [Benefits of Mock Testing](#benefits-of-mock-testing)
- [Setting up LLM Fallback Configuration](#setting-up-llm-fallback-configuration)
- [Testing API Rate Limits with Fallback Models](#testing-api-rate-limits-with-fallback-models)
- [If you specify an error that needs to be handled](#if-you-specify-an-error-that-needs-to-be-handled)
- [Specifying multiple models in fallback sequentially](#specifying-multiple-models-in-fallback-sequentially)
- [Using Different Prompt Templates for Each Model](#using-different-prompt-templates-for-each-model)
- [Automatic Model Switching Based on Context Length](#automatic-model-switching-based-on-context-length)


### Key Concepts

1. **Fundamentals of Fallbacks**
   - Core concepts of fallback mechanisms
   - Setting up basic fallback configurations
   - Understanding error handling patterns
   - Implementation of simple fallback chains

2. **API Error Management**
   - Handling rate limit errors effectively
   - Managing API downtime scenarios
   - Implementing retry strategies
   - Simulating errors through mock testing

3. **Advanced Fallback Patterns**
   - Configuring multiple fallback models
   - Custom exception handling setup
   - Sequential fallback execution
   - Context-aware model switching
   - Model-specific prompt templating

4. **Practical Implementation**
   - Integration with OpenAI and Anthropic models
   - Building resilient chains with fallbacks
   - Real-world usage patterns and best practices
   - Performance optimization techniques

### References

- [LangChain Expression Language Documentation](https://python.langchain.com/docs/expression_language/)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/errors)
- [LangChain Fallbacks Guide](https://python.langchain.com/docs/how_to/fallbacks/)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Fallbacks",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## What is Fallbacks?

In LLM applications, there are various errors or failures such as LLM API issues, degradation in model output quality, and other integration-related issues. The `fallback` feature can be utilized to gracefully handle and isolate these problems.

Importantly, fallbacks can be applied not only at the LLM level but also at the entire executable level.

## How to Handle LLM API Errors

Handling LLM API errors is one of the most common use cases for using `fallbacks`.

Requests to the LLM API can fail for various reasons. The API might be down, you might have reached a rate limit, or there could be several other issues. Using `fallbacks` can help protect against these types of problems.

**Important**: By default, many LLM wrappers capture errors and retry. When using `fallbacks`, it is advisable to disable this default behavior. Otherwise, the first wrapper will keep retrying and not fail.

## Introduction to Rate Limit Testing

First, let's perform a mock test for the `RateLimitError` that can occur with OpenAI. A `RateLimitError` is **an error that occurs when you exceed the API usage limits** of the OpenAI API.

## Why Handle Rate Limit Errors?

When this error occurs, API requests are restricted for a certain period, so applications need to handle this situation appropriately. Through mock testing, we can verify how the application behaves when a `RateLimitError` occurs and check the error handling logic.

## Benefits of Mock Testing

This allows us to prevent potential issues that could arise in production environments and ensure stable service delivery.

```python
from openai import RateLimitError
from unittest.mock import patch
import httpx

request = httpx.Request("GET", "/")
response = httpx.Response(
    200, request=request
)  # Generate a response with a 200 status code.

# Generate a RateLimitError with the message ‘rate limit’, the response, and an empty body.
error = RateLimitError("rate limit", response=response, body="")
```

## Setting up LLM Fallback Configuration

Create a `ChatOpenAI` object and assign it to the `openai_llm` variable, setting the `max_retries` parameter to 0 to **prevent retry attempts** that might occur due to API call limits or restrictions.

Using the `with_fallbacks` method, configure `anthropic_llm` as the `fallback` LLM and assign this configuration to the `llm` variable.


```python
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_openai.chat_models.base import ChatOpenAI

# Create an openai_llm object using OpenAI's ChatOpenAI model.
# Set max_retries to 0 to prevent retries due to rate limits, etc.
openai_llm = ChatOpenAI(max_retries=0)

# Create an anthropic_llm object using Anthropic's ChatAnthropic model.
anthropic_llm = ChatAnthropic(model="claude-3-opus-20240229")

# use openai_llm as default, and anthropic_llm as fallback on failure.
llm = openai_llm.with_fallbacks([anthropic_llm])
```

## Testing API Rate Limits with Fallback Models

In this example, we'll simulate OpenAI API rate limits and test how the system behaves when encountering API cost limitation errors.

You'll see that when the OpenAI GPT model encounters an error, the fallback model (Anthropic) successfully takes over and performs the inference instead.

When a fallback model is configured using `with_fallbacks()` and successfully executes, the `RateLimitError` won't be raised, ensuring continuous operation of your application.

> 💡 This demonstrates how LangChain's fallback mechanism provides resilience against API limitations and ensures your application continues to function even when the primary model is unavailable.

```python
# Use OpenAI LLM first to show error.
with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
    try:
        print(openai_llm.invoke("Why did the chicken cross the road?"))
    except RateLimitError:
        # If error occurs, print error.
        print("Hit error")
```

<pre class="custom">Hit error
</pre>

```python
# Code to replace with Anthropic if an error occurs when calling the OpenAI API
with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
    try:
        print(llm.invoke("Why did the chicken cross the road?"))
    except RateLimitError:
        print("Hit error")
```

<pre class="custom">content='The classic answer to the joke "Why did the chicken cross the road?" is:\n\n"To get to the other side."\n\nThis answer is an anti-joke, meaning that the answer is purposely obvious and straightforward, lacking the expected punch line or humor that a joke typically has. The humor, if any, comes from the fact that the answer is so simple and doesn\'t really provide any meaningful explanation for the chicken\'s actions.\n\nThere are, of course, many variations and alternative answers to this joke, but the one mentioned above remains the most well-known and traditional response.' additional_kwargs={} response_metadata={'id': 'msg_01EnWEZFHrLnPx8DeYWAwKcY', 'model': 'claude-3-opus-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 15, 'output_tokens': 124}} id='run-c83ea304-76f5-4bc3-b33b-4ce1ecaa7220-0' usage_metadata={'input_tokens': 15, 'output_tokens': 124, 'total_tokens': 139, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}
</pre>

A model set to `llm.with_fallbacks()` will also behave the same as a regular runnable model.

The code below also doesn't throw an ‘error’ because the fallbacks model did a good job.

```python
from langchain_core.prompts.chat import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "Please keep your answers short and concise.",
        ),
        ("human", "What is the capital of {country}?"),
    ]
)

good_chain = prompt | llm  # Linking prompts and language models to create chains
bad_chain = prompt | ChatOpenAI()  # it will output an ‘Hit error’ statement.

with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
    try:
        print(good_chain.invoke({"country": "South Korea"}))
    except RateLimitError:
        print("Hit error")
```

<pre class="custom">content='The capital of South Korea is Seoul.' additional_kwargs={} response_metadata={'id': 'msg_013Uqu28KjoN25xFPEmP1Uca', 'model': 'claude-3-opus-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 24, 'output_tokens': 11}} id='run-73627f85-a617-4044-9363-be5f451d79b9-0' usage_metadata={'input_tokens': 24, 'output_tokens': 11, 'total_tokens': 35, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}
</pre>

## If you specify an error that needs to be handled

When working with fallbacks, you can precisely define when the `fallback` should be triggered. This allows for more granular control over the fallback mechanism's behavior.

For example, you can specify certain exception classes or error codes that will trigger the fallback logic. This approach helps you to **reduce unnecessary fallback calls and improve error handling efficiency.**

In the example below, you'll see an "error occurred" message printed. This happens because we've configured the `exceptions_to_handle` parameter to only trigger the fallback when a `KeyboardInterrupt` exception occurs. As a result, the `fallback` won't be triggered for any other exceptions.


```python
llm = openai_llm.with_fallbacks(
    # Use anthropic_llm as the fallback LLM and specify KeyboardInterrupt as the exception to handle.
    [anthropic_llm],
    exceptions_to_handle=(KeyboardInterrupt,),  # Specify the exception to handle.
)

# Linking prompts and LLM to create chains.
chain = prompt | llm
with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
    try:
        print(chain.invoke({"country": "South Korea"}))
    except RateLimitError:
        # If a RateLimitError occurs, print "Hit error".
        print("Hit error")
```

<pre class="custom">Hit error
</pre>

## Specifying multiple models in fallback sequentially

You can specify multiple models in the `fallback` model, not just one. When multiple models are specified, they will be tried sequentially.


```python
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI

# Create a prompt
prompt_template = (
    "Please keep your answers short and concise.\n\nQuestion:\n{question}\n\nAnswer:"
)
prompt = PromptTemplate.from_template(prompt_template)
```

Create two chains, one that causes an error and one that works normally.


```python
# Here, we'll create a chain using a model name that easily causes an error.
chat_model = ChatOpenAI(model_name="gpt-fake")
bad_chain = prompt | chat_model
```

```python
# Create fallback chains.
fallback_chain1 = prompt | ChatOpenAI(model="gpt-3.6-turbo")  # error
fallback_chain2 = prompt | ChatOpenAI(model="gpt-3.5-turbo")  # normal
fallback_chain3 = prompt | ChatOpenAI(model="gpt-4-turbo-preview")  # normal
```

```python
# Combine two chains to create a final chain.
chain = bad_chain.with_fallbacks([fallback_chain1, fallback_chain2, fallback_chain3])
# Call the created chain and pass the input value.
chain.invoke({"question": "What is the capital of South Korea?"})
```




<pre class="custom">AIMessage(content='Seoul', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 27, 'total_tokens': 30, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-92fc9db1-ad13-4f38-8f2d-d136bf419691-0', usage_metadata={'input_tokens': 27, 'output_tokens': 3, 'total_tokens': 30, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



## Using Different Prompt Templates for Each Model
You can use different prompt templates tailored to each model's characteristics. For example, GPT-4 can handle complex instructions while GPT-3.5 can work with simpler ones.

```python
# Set up model-specific prompt templates
gpt4_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a professional AI assistant. Provide detailed and academic responses.",
        ),
        ("human", "{question}"),
    ]
)

gpt35_prompt = ChatPromptTemplate(
    [("system", "Provide simple and clear responses."), ("human", "{question}")]
)

# Set up models
primary_chain = gpt4_prompt | ChatOpenAI(model="gpt-4", max_retries=1)
fallback_chain = gpt35_prompt | ChatOpenAI(model="gpt-3.5-turbo")
```

```python
# Test function
def test_chain(question):
    print(f"Question: {question}\n")
    try:
        # Primary model (GPT-4) response
        primary_response = primary_chain.invoke({"question": question})
        print("GPT-4 Response:")
        print(f"{primary_response}\n")
    except Exception as e:
        print(f"GPT-4 Error: {e}\n")

    try:
        # Fallback model (GPT-3.5) response
        fallback_response = fallback_chain.invoke({"question": question})
        print("GPT-3.5 Response:")
        print(f"{fallback_response}\n")
    except Exception as e:
        print(f"GPT-3.5 Error: {e}\n")

    # Full fallback chain execution
    print("Fallback Chain Response:")
    final_response = chain.invoke({"question": question})
    print(f"{final_response}\n")
    print("-" * 50)


# Test with various questions
questions = [
    "What is artificial intelligence?",
    "Explain quantum mechanics in 5 lines.",
]

for question in questions:
    test_chain(question)
```

<pre class="custom">Question: What is artificial intelligence?
    
    GPT-4 Response:
    content="Artificial Intelligence, often abbreviated as AI, is a branch of computer science that aims to create systems capable of performing tasks that normally require human intelligence. These tasks include learning and adapting to new inputs, understanding human language (natural language processing), recognizing patterns (pattern recognition), solving problems (problem-solving), and making decisions.\n\nAI can be classified into two types: Narrow AI and General AI. \n\n1. Narrow AI, also known as Weak AI, is designed to perform a narrow task, such as voice recognition, recommendation systems, or image recognition. It operates under a limited set of constraints and is focused on a single narrow task. The AI systems we currently have, like virtual assistants (Siri, Alexa), are examples of Narrow AI.\n\n2. General AI, also known as Strong AI, is the kind of AI that has the potential to outperform humans at nearly every cognitive task. This type of AI is not only able to perform tasks that a human can do, but it can also understand, learn, adapt, and implement knowledge from different domains. As of now, General AI is a theoretical concept and doesn't exist.\n\nAI technologies include machine learning (where a computer is fed a lot of data which it uses to teach itself to perform a task), neural networks (complex systems that mimic the human brain), and natural language processing (enabling AI to understand and respond in human language). AI has wide-ranging uses, from medical diagnosis and autonomous vehicles to customer service and data analysis." additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 301, 'prompt_tokens': 29, 'total_tokens': 330, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-cf4f126e-d150-4cb6-bda0-0b3b00faf800-0' usage_metadata={'input_tokens': 29, 'output_tokens': 301, 'total_tokens': 330, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    GPT-3.5 Response:
    content='Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. It involves tasks such as learning, reasoning, problem-solving, perception, and language understanding.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 36, 'prompt_tokens': 22, 'total_tokens': 58, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-c1713439-5e33-4525-885f-4321771fbfab-0' usage_metadata={'input_tokens': 22, 'output_tokens': 36, 'total_tokens': 58, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    Fallback Chain Response:
    content='Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 24, 'total_tokens': 42, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-2a46589c-8c18-4ca4-952d-afe580857d29-0' usage_metadata={'input_tokens': 24, 'output_tokens': 18, 'total_tokens': 42, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    --------------------------------------------------
    Question: Explain quantum mechanics in 5 lines.
    
    GPT-4 Response:
    content="Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales of energy levels of atoms and subatomic particles. It introduces the concept of wave-particle duality, meaning particles can exhibit properties of both particles and waves. The principle of superposition stipulates that a physical system exists partially in all its theoretical states simultaneously. Quantum entanglement allows particles to be connected, such that the state of one can instantly affect the state of another, regardless of the distance between them. Lastly, Heisenberg's uncertainty principle posits that the position and velocity of a particle cannot be simultaneously measured with high precision." additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 126, 'prompt_tokens': 33, 'total_tokens': 159, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-3d644553-7f02-45a0-8ee3-bc83f7509ae5-0' usage_metadata={'input_tokens': 33, 'output_tokens': 126, 'total_tokens': 159, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    GPT-3.5 Response:
    content='Quantum mechanics is a branch of physics that deals with the behavior of very small particles, such as atoms and subatomic particles. It describes how these particles can exist in multiple states at the same time, known as superposition. Quantum mechanics also involves the concept of wave-particle duality, where particles can exhibit both wave-like and particle-like behavior. Additionally, quantum mechanics includes the uncertainty principle, which states that we cannot simultaneously know both the exact position and momentum of a particle. Overall, quantum mechanics provides a mathematical framework to understand the behavior of these tiny particles at the quantum level.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 119, 'prompt_tokens': 26, 'total_tokens': 145, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-0065fb17-6d01-4849-aa8e-7e3eaae0c699-0' usage_metadata={'input_tokens': 26, 'output_tokens': 119, 'total_tokens': 145, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    Fallback Chain Response:
    content='Quantum mechanics is a branch of physics that deals with the behavior of very small particles. It describes how particles like electrons and photons can exist in multiple states at the same time. This theory is based on probability and uncertainty, rather than deterministic laws. Quantum mechanics has led to technological advancements like quantum computing and cryptography. It remains a complex and fundamental theory in modern physics.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 75, 'prompt_tokens': 28, 'total_tokens': 103, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-8abe3419-c72a-4f2b-a958-1c9c40e8ff9c-0' usage_metadata={'input_tokens': 28, 'output_tokens': 75, 'total_tokens': 103, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    --------------------------------------------------
</pre>

## Automatic Model Switching Based on Context Length
When handling long contexts, you can automatically switch to models with larger context windows if token limits are exceeded.

```python
import time

# Model setup
standard_model = ChatOpenAI(model="gpt-4o", max_tokens=3000, max_retries=1)

large_context_model = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    max_tokens=10000,
)


def create_context_aware_chain(text):
    text_length = len(text)
    print(f"Input text length: {text_length} characters")

    # Model selection logic
    if text_length > 20000:
        print("Long text detected -> Using GPT-3.5-turbo-16k model")
        current_model = large_context_model
    else:
        print("Standard text detected -> Using GPT-4o model (with fallback)")
        current_model = standard_model.with_fallbacks([large_context_model])

    # Prompt template setup
    prompt = PromptTemplate.from_template(
        "Please summarize the following text in 3 lines:\n\n{text}\n\nSummary:"
    )

    # Chain composition
    chain = prompt | current_model
    return chain


def test_summarization(text, description):
    print(f"\n=== {description} Test ===")

    # Create chain
    start_time = time.time()
    chain = create_context_aware_chain(text)

    try:
        print("\nStarting summarization...")
        response = chain.invoke({"text": text})
        print("\nSummary Result:")
        print(response)

    except Exception as e:
        print(f"\nError occurred: {e}")

    end_time = time.time()
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    print("=" * 50)


# Prepare test texts
short_text = """
Artificial Intelligence (AI) is at the core of modern technology. Through machine learning 
and deep learning, it can solve various problems and is being utilized in multiple fields 
such as natural language processing, computer vision, and robotics.
"""

# Generate long text
long_text = (
    """
The advancement of artificial intelligence is significantly transforming our society. 
In healthcare, it's being used for disease diagnosis and treatment planning. In education, 
it provides personalized learning experiences. In finance, it supports risk assessment 
and investment decisions, while in manufacturing, it optimizes production processes. 
Recently, with the development of generative AI, its application has expanded into 
creative fields such as art, music, and writing.
"""
    * 50
)  # Repeat text 100 times to create long context

# Run tests
print("Automatic Model Switching Based on Context Length Test\n")
print("1. Short Text Test")
test_summarization(short_text, "Short Text")

print("\n2. Long Text Test")
test_summarization(long_text, "Long Text")
```

<pre class="custom">Automatic Model Switching Based on Context Length Test
    
    1. Short Text Test
    
    === Short Text Test ===
    Input text length: 252 characters
    Standard text detected -> Using GPT-4o model (with fallback)
    
    Starting summarization...
    
    Summary Result:
    content='Artificial Intelligence (AI) is central to modern technology, leveraging machine learning and deep learning to address diverse problems. It is applied across various fields, including natural language processing, computer vision, and robotics.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 41, 'prompt_tokens': 65, 'total_tokens': 106, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'stop', 'logprobs': None} id='run-a33441be-a753-40ad-b955-c97f7b9d3474-0' usage_metadata={'input_tokens': 65, 'output_tokens': 41, 'total_tokens': 106, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    Processing time: 1.01 seconds
    ==================================================
    
    2. Long Text Test
    
    === Long Text Test ===
    Input text length: 24350 characters
    Long text detected -> Using GPT-3.5-turbo-16k model
    
    Starting summarization...
    
    Summary Result:
    content='Artificial intelligence is transforming society in various sectors including healthcare, education, finance, and manufacturing. It is used for disease diagnosis, personalized learning, risk assessment, investment decisions, and production optimization. The development of generative AI has also expanded its application into creative fields such as art, music, and writing.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 63, 'prompt_tokens': 4319, 'total_tokens': 4382, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 4224}}, 'model_name': 'gpt-3.5-turbo-16k-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-d9fe27c5-c8a2-4e0d-892a-e2519cde6688-0' usage_metadata={'input_tokens': 4319, 'output_tokens': 63, 'total_tokens': 4382, 'input_token_details': {'audio': 0, 'cache_read': 4224}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    
    Processing time: 1.09 seconds
    ==================================================
</pre>
