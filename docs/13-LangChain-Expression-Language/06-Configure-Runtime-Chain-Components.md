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

# Configure-Runtime-Chain-Components

- Author: [HeeWung Song(Dan)](https://github.com/kofsitho87)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/06-MarkdownHeaderTextSplitter.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/06-MarkdownHeaderTextSplitter.ipynb)

## Overview

In this tutorial, we will explore how to dynamically configure various options when calls a chain.

There are two ways to implement dynamic configuration:

- First, the `configurable_fields` method allows you to configure specific fields of a Runnable object.
    - Dynamically modify specific field values at runtime
    - Example: Adjust individual parameters like `temperature`, `model_name` of an LLM

- Second, the `configurable_alternatives` method lets you specify alternatives for a particular Runnable object that can be set during runtime
    - Replace entire components with alternatives at runtime
    - Example: Switch between different LLM models or prompt templates

**[Note]**
The term **Configurable fields** refers to settings or parameters within a system that can be adjusted or modified by the user or administrator at runtime.

- Configuration Application
    - `with_config` method: A unified interface for applying all configuration settings
    - Ability to apply single or multiple settings simultaneously
    - Used consistently across special components like `HubRunnable`

In the following sections, we'll cover detailed usage of each method and practical applications. We'll explore real-world examples including prompt management through `HubRunnable`, setting various prompt alternatives, switching between LLM models, and more.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Configurable Fields](#configurable-fields)
- [Configurable Alternatives with HubRunnables](#configurable-alternatives-with-hubrunnables)
- [Switching between Runnables](#switching-between-runnables)
- [Setting Prompt Alternatives](#setting-prompt-alternatives)
- [Configuring Prompts and LLMs](#configuring-prompts-and-llms)
- [Saving Configurations](#saving-configurations)


### References

- [LangChain How to configure runtime chain internals](https://python.langchain.com/docs/how_to/configure/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/)
- [LangChain Chaining runnables](https://python.langchain.com/docs/how_to/sequence/)
- [LangChain HubRunnable](https://python.langchain.com/api_reference/langchain/runnables/langchain.runnables.hub.HubRunnable.html)
----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The `langchain-opentutorial` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "LANGCHAIN_PROJECT": "Configure-Runtime-Chain-Components",
    }
)
```

Alternatively, you can set and load `OPENAI_API_KEY` from a `.env` file.

**[Note]** This is only necessary if you haven't already set `OPENAI_API_KEY` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



## Configurable Fields

`Configurable fields` provide a way to dynamically modify specific parameters of a Runnable object at runtime. This feature is essential when you need to fine-tune the behavior of your chains or models without changing their core implementation.

- They allow you to specify which parameters can be modified during execution
- Each configurable field can include a description that explains its purpose
- You can configure multiple fields simultaneously
- The original chain structure remains unchanged, even when you modify configurations for different runs.

The `configurable_fields` method is used to specify which parameters should be treated as configurable, making your LangChain applications more flexible and adaptable to different use cases.

### Dynamic Property Configuration

Let's illustrate this with `ChatOpenAI`. When using ChatOpenAI, we can set various properties.

The `model_name` property is used to specify the version of GPT. For example, you can select different models by setting it to `gpt-4o`, `gpt-4o-mini`, or else.

To dynamically specify the model instead of using a fixed `model_name`, you can leverage the `ConfigurableField` and assign it to a dynamically configurable property value as follows:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, model_name="gpt-4o")

model.invoke("Where is the capital of the United States?").__dict__
```




<pre class="custom">{'content': 'The capital of the United States is Washington, D.C.',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 13,
       'prompt_tokens': 16,
       'total_tokens': 29,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-2024-08-06',
      'system_fingerprint': 'fp_d28bcae782',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-15070403-58a9-4724-aeed-3a10812fa498-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 16,
      'output_tokens': 13,
      'total_tokens': 29,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



```python
model = ChatOpenAI(temperature=0).configurable_fields(
    # model_name is an original field of ChatOpenAI
    model_name=ConfigurableField(
        # Set the unique identifier of the field
        id="gpt_version",  
        # Set the name for model_name
        name="Version of GPT",  
        # Set the description for model_name
        description="Official model name of GPTs. ex) gpt-4o, gpt-4o-mini",
    )
)
```

When calling `model.invoke()`, you can dynamically specify parameters using the format `config={"configurable": {"key": "value"}}`.

```python
model.invoke(
    "Where is the capital of the United States?",
    # Set gpt_version to gpt-3.5-turbo
    config={"configurable": {"gpt_vision": "gpt-3.5-turbo"}},
).__dict__
```




<pre class="custom">{'content': 'The capital of the United States is Washington, D.C.',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 13,
       'prompt_tokens': 16,
       'total_tokens': 29,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-3.5-turbo-0125',
      'system_fingerprint': None,
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-9f96f444-6e15-412d-8622-840a181452e5-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 16,
      'output_tokens': 13,
      'total_tokens': 29,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



Now let's try using the `gpt-4o-mini` model. Check the output to see the changed model.

```python
model.invoke(
    # Set gpt_version to gpt-4o-mini
    "Where is the capital of the United States?",
    config={"configurable": {"gpt_version": "gpt-4o-mini"}},
).__dict__
```




<pre class="custom">{'content': 'The capital of the United States is Washington, D.C.',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 13,
       'prompt_tokens': 16,
       'total_tokens': 29,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-mini-2024-07-18',
      'system_fingerprint': 'fp_0aa8d3e20b',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-d9f59e18-51ee-465f-bb52-118b25fef8a8-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 16,
      'output_tokens': 13,
      'total_tokens': 29,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



Alternatively, you can set `configurable` parameters using the `with_config()` method of the `model` object to achieve the same result.

```python
model.with_config(configurable={"gpt_version": "gpt-4o-mini"}).invoke(
    "Where is the capital of the United States?",
).__dict__
```




<pre class="custom">{'content': 'The capital of the United States is Washington, D.C.',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 12,
       'prompt_tokens': 16,
       'total_tokens': 28,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-mini-2024-07-18',
      'system_fingerprint': 'fp_d02d531b47',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-f1b5e176-8247-45bb-9ecd-5c93e3e1fc94-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 16,
      'output_tokens': 12,
      'total_tokens': 28,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



Or you can also use this function as part of a chain.

```python
# Create a prompt template from the template
prompt = PromptTemplate.from_template("Select a random number greater than {x}")
chain = (
    prompt | model
)  # Create a chain by connecting prompt and model. The prompt's output is passed as input to the model.
```

```python
# Call the chain and pass 0 as the input variable "x"
chain.invoke({"x": 0}).__dict__  
```




<pre class="custom">{'content': '73',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 2,
       'prompt_tokens': 15,
       'total_tokens': 17,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-3.5-turbo-0125',
      'system_fingerprint': None,
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-a8764bb6-93f2-4a35-bff3-d635aa1a4b9a-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 15,
      'output_tokens': 2,
      'total_tokens': 17,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



```python
# Call the chain with configuration settings
chain.with_config(configurable={"gpt_version": "gpt-4o"}).invoke({"x": 0}).__dict__
```




<pre class="custom">{'content': "Sure! Here's a random number greater than 0: 7.",
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 15,
       'prompt_tokens': 15,
       'total_tokens': 30,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-2024-08-06',
      'system_fingerprint': 'fp_5f20662549',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-3d5c27bd-c9d6-42cd-8a91-467adce4103e-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 15,
      'output_tokens': 15,
      'total_tokens': 30,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



## Configurable Alternatives with HubRunnables

Using `HubRunnable` simplifies dynamic prompt selection, allowing easy switching between prompts registered in the Hub

### Configuring LangChain Hub Settings

`HubRunnable` provide an option to configure which prompt template to pull from the LangChain Hub. This enables you to dynamically select different prompts based on the hub path specification.

```python
from langchain.runnables.hub import HubRunnable

prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    # ConfigurableField for setting owner repository commit
    owner_repo_commit=ConfigurableField(
        # Field ID
        id="hub_commit",
        # Field name
        name="Hub Commit",
        # Field description
        description="The Hub commit to pull from",
    )
)
prompt
```




<pre class="custom">RunnableConfigurableFields(default=HubRunnable(bound=ChatPromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})]), kwargs={}, config={}, config_factories=[], owner_repo_commit='rlm/rag-prompt'), fields={'owner_repo_commit': ConfigurableField(id='hub_commit', name='Hub Commit', description='The Hub commit to pull from', annotation=None, is_shared=False)})</pre>



If you call the `prompt.invoke()` method without specifying a `with_config`, the Runnable will automatically pull and use the prompt that was initially registered in the set **\"rlm/rag-prompt\"** hub.

```python
# Call the prompt object's invoke method with "question" and "context" parameters
prompt.invoke({"question": "Hello", "context": "World"}).messages
```




<pre class="custom">[HumanMessage(content="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: Hello \nContext: World \nAnswer:", additional_kwargs={}, response_metadata={})]</pre>



```python
prompt.with_config(
    # Set hub_commit to teddynote/summary-stuff-documents
    configurable={"hub_commit": "teddynote/summary-stuff-documents"}
).invoke({"context": "Hello"})
```




<pre class="custom">StringPromptValue(text='Please summarize the sentence according to the following REQUEST.\nREQUEST:\n1. Summarize the main points in bullet points.\n2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.\n3. Use various emojis to make the summary more interesting.\n4. DO NOT include any unnecessary information.\n\nCONTEXT:\nHello\n\nSUMMARY:"\n')</pre>



## Switching between Runnables

**Configurable alternatives** provide a way to select between different Runnable objects that can be set at runtime.

For example, the configurable language model of `ChatAnthropic` provides high degree of flexibility that can be applied to various tasks and contexts.

To enable dynamic switching, we can define the model's parameters as `ConfigurableField` objects.

- `model`: Specifies the base language model to be used.

- `temperature`: Controls the randomness of the model's sampling (which values between 0 and 1). Lower values result in more deterministic and repetitive outputs, while higher values lead to more diverse and creative responses.

### Setting Alternatives for LLM Objects

Let's explore how to implement configurable alternatives using a Large Language Model (LLM).

[Note]

- To use the `ChatAnthropic` model, you need to obtain an API key from the Anthropic console: https://console.anthropic.com/dashboard.
- You can uncomment and directly set the API key (as shown below) or store it in your `.env` file.

Set the `ANTHROPIC_API_KEY` environment variable in your code.

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "Enter your ANTHROPIC API KEY here."
```

```python
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

llm = ChatAnthropic(
    temperature=0, model="claude-3-5-sonnet-20241022"
).configurable_alternatives(
    # Assign an ID to this field.
    # This ID will be used to configure the field when constructing the final runnable object.
    ConfigurableField(id="llm"),
    # Set the default key.
    # When this key is specified, it will use the default LLM (ChatAnthropic) initialized above.
    default_key="anthropic",
    # Add a new option named 'openai', which is equivalent to `ChatOpenAI(model="gpt-4o-mini")`.
    openai=ChatOpenAI(model="gpt-4o-mini"),
    # Add a new option named 'gpt4o', which is equivalent to `ChatOpenAI(model="gpt-4o")`.
    gpt4o=ChatOpenAI(model="gpt-4o"),
    # You can add more configuration options here.
)
prompt = PromptTemplate.from_template("Please briefly explain about {topic}.")
chain = prompt | llm
```

Here's how you can invoke a chain using the default `ChatAnthropic` model using `chain.invoke()`.

```python
# Invoke using Anthropic as the default.
chain.invoke({"topic": "NewJeans"}).__dict__
```




<pre class="custom">{'content': 'NewJeans is a South Korean girl group formed by ADOR (All Doors One Room), a subsidiary of HYBE Corporation. The group debuted on July 1, 2022, and consists of five members: Minji, Hanni, Danielle, Haerin, and Hyein. They quickly rose to prominence with their debut single "Attention" and follow-up hits like "Hype Boy" and "Ditto."\n\nThe group is known for their fresh, teen-crush concept and retro-influenced music style that combines various genres including R&B, pop, and hip-hop. Their name "NewJeans" is a play on words, referring both to new genes (representing a new generation) and blue jeans (a timeless fashion item).\n\nNewJeans has achieved significant commercial success and critical acclaim since their debut, breaking several records and winning multiple awards. They\'re particularly noted for their strong international appeal and distinctive marketing approach. The group has become one of the most successful fourth-generation K-pop girl groups, known for their sophisticated style and high-quality music productions.',
     'additional_kwargs': {},
     'response_metadata': {'id': 'msg_01LKcn9YZo8eeXHx38ucdKfW',
      'model': 'claude-3-5-sonnet-20241022',
      'stop_reason': 'end_turn',
      'stop_sequence': None,
      'usage': {'cache_creation_input_tokens': 0,
       'cache_read_input_tokens': 0,
       'input_tokens': 15,
       'output_tokens': 242}},
     'type': 'ai',
     'name': None,
     'id': 'run-1f9b71ce-a931-487f-9a26-5e9a6a3c2d22-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 15,
      'output_tokens': 242,
      'total_tokens': 257,
      'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}</pre>



You may specify a different model to use the `llm` by using `chain.with_config(configurable={"llm": "model"})`.

```python
# Invoke by changing the chain's configuration.
chain.with_config(configurable={"llm": "openai"}).invoke({"topic": "NewJeans"}).__dict__
```




<pre class="custom">{'content': 'NewJeans is a South Korean girl group formed by ADOR, a subsidiary of HYBE Corporation, which is known for managing successful acts like BTS and TXT. Debuting in August 2022, NewJeans quickly gained popularity for their fresh sound, catchy songs, and unique fashion sense. The group is characterized by their retro-inspired aesthetics and a blend of pop, R&B, and hip-hop influences in their music.\n\nThe members of NewJeans include Minji, Hanni, Danielle, Haerin, and Hyein. Their debut EP, titled "New Jeans," featured hit tracks like "Attention" and "Hype Boy," which showcased their vocal talents and captivating choreography. The group has been recognized for their innovative approach to music and marketing, including engaging with fans through social media and unique promotional strategies. NewJeans has established itself as a prominent name in the K-pop industry, garnering a dedicated fanbase both in South Korea and internationally.',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 196,
       'prompt_tokens': 15,
       'total_tokens': 211,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-mini-2024-07-18',
      'system_fingerprint': 'fp_0aa8d3e20b',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-2df17b41-3e7b-47d9-a168-6d7df4e03bdb-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 15,
      'output_tokens': 196,
      'total_tokens': 211,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



Now, change the chain's configuration to use `gpt4o` as the language model.

```python
# Invoke by changing the chain's configuration.
chain.with_config(configurable={"llm": "gpt4o"}).invoke({"topic": "NewJeans"}).__dict__
```




<pre class="custom">{'content': "NewJeans is a South Korean girl group formed by ADOR, a subsidiary under HYBE Corporation. The group debuted in July 2022 and quickly gained attention for their fresh concept and music style that sets them apart from traditional K-pop trends. Known for their retro-inspired sound and fashion, NewJeans aims to evoke a sense of nostalgia while appealing to a broad audience through their innovative approach to music and performance. Their debut was marked by the release of singles that showcase their distinctive blend of catchy melodies and youthful energy. The group has been praised for their artistic direction and the members' talents, contributing to their rapid rise in popularity within the K-pop industry.",
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 134,
       'prompt_tokens': 15,
       'total_tokens': 149,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-2024-08-06',
      'system_fingerprint': 'fp_5f20662549',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-60cbd11d-4053-4904-a2aa-c3a5d871cb55-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 15,
      'output_tokens': 134,
      'total_tokens': 149,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



For this time, change the chain's configuration to use `anthropic`.

```python
# Invoke by changing the chain's configuration.
chain.with_config(configurable={"llm": "anthropic"}).invoke(
    {"topic": "NewJeans"}
).__dict__
```




<pre class="custom">{'content': 'NewJeans is a South Korean girl group formed by ADOR (All Doors One Room), a subsidiary of HYBE Corporation. The group debuted on July 1, 2022, and consists of five members: Minji, Hanni, Danielle, Haerin, and Hyein. They quickly rose to prominence with their debut single "Attention" and follow-up hits like "Hype Boy" and "Ditto."\n\nThe group is known for their fresh, teen-crush concept and retro-influenced music style that combines various genres including R&B, pop, and hip-hop. Their name "NewJeans" is a play on words, referring both to new genes (representing a new generation) and blue jeans (a timeless fashion item).\n\nNewJeans has achieved significant commercial success and critical acclaim since their debut, breaking several records and winning multiple awards. They\'re particularly noted for their strong international appeal and distinctive marketing approach, which helped them become one of the most successful fourth-generation K-pop groups.',
     'additional_kwargs': {},
     'response_metadata': {'id': 'msg_01GZLTuXnbTyyFAnE9uh9diZ',
      'model': 'claude-3-5-sonnet-20241022',
      'stop_reason': 'end_turn',
      'stop_sequence': None,
      'usage': {'cache_creation_input_tokens': 0,
       'cache_read_input_tokens': 0,
       'input_tokens': 15,
       'output_tokens': 229}},
     'type': 'ai',
     'name': None,
     'id': 'run-c8e3a472-cbbf-41fe-878c-efe564a029af-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 15,
      'output_tokens': 229,
      'total_tokens': 244,
      'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}</pre>



## Setting Prompt Alternatives

Prompts can be configured in a similar pattern to the configuration of LLM alternatives that we previously set.

```python
# Initialize the language model and set the temperature to 0.
llm = ChatOpenAI(temperature=0)

prompt = PromptTemplate.from_template(
    # Default prompt template
    "Where is the capital of {country}?"
).configurable_alternatives(
    # Assign an ID to this field.
    ConfigurableField(id="prompt"),
    # Set the default key.
    default_key="capital",
    # Add a new option named 'area'.
    area=PromptTemplate.from_template("What is the area of {country}?"),
    # Add a new option named 'population'.
    population=PromptTemplate.from_template("What is the population of {country}?"),
    # Add a new option named 'eng'.
    kor=PromptTemplate.from_template("Translate {input} to Korean."),
    # You can add more configuration options here.
)

# Create a chain by connecting the prompt and language model.
chain = prompt | llm
```

If no configuration changes are made, the default prompt will be used.

```python
# Call the chain without any configuration changes.
chain.invoke({"country": "South Korea"})
```




<pre class="custom">AIMessage(content='The capital of South Korea is Seoul.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 15, 'total_tokens': 24, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-a2a888c2-cd2f-4159-aa97-0e6dee3b90ac-0', usage_metadata={'input_tokens': 15, 'output_tokens': 9, 'total_tokens': 24, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



To use a different prompt, use `with_config`.

```python
# Call the chain by changing the chain's configuration using with_config.
chain.with_config(configurable={"prompt": "area"}).invoke({"country": "South Korea"})
```




<pre class="custom">AIMessage(content='The total area of South Korea is approximately 100,363 square kilometers.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 15, 'total_tokens': 31, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-f6ad85fd-369b-47e9-a524-3cb15a113ff7-0', usage_metadata={'input_tokens': 15, 'output_tokens': 16, 'total_tokens': 31, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



```python
# Call the chain by changing the chain's configuration using with_config.
chain.with_config(configurable={"prompt": "population"}).invoke({"country": "South Korea"})
```




<pre class="custom">AIMessage(content='As of 2021, the population of South Korea is approximately 51.8 million.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 15, 'total_tokens': 35, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-ddb641f0-154f-470b-a803-ec409aa417f9-0', usage_metadata={'input_tokens': 15, 'output_tokens': 20, 'total_tokens': 35, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



Now let's use the `kor` prompt to request a translation, for example, pass the input using the `input` variable.

```python
# Call the chain by changing the chain's configuration using with_config.
chain.with_config(configurable={"prompt": "kor"}).invoke({"input": "apple is delicious!"})
```




<pre class="custom">AIMessage(content='사과는 맛있어요!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 15, 'total_tokens': 26, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-cce73072-7587-4d9d-8a73-4171346899fe-0', usage_metadata={'input_tokens': 15, 'output_tokens': 11, 'total_tokens': 26, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



## Configuring Prompts and LLMs

You can configure multiple aspects using prompts and LLMs simultaneously.

Here's an example that demonstrates how to use both prompts and LLMs to accomplish this:

```python
llm = ChatAnthropic(
    temperature=0, model="claude-3-5-sonnet-20241022"
).configurable_alternatives(
    # Assign an ID to this field.
    # When configuring the end runnable, we can then use this id to configure this field.
    ConfigurableField(id="llm"),
    # Set the default key.
    # When this key is specified, it will use the default LLM (ChatAnthropic) initialized above.
    default_key="anthropic",
    # Add a new option named 'openai', which is equivalent to `ChatOpenAI(model="gpt-4o-mini")`.
    openai=ChatOpenAI(model="gpt-4o-mini"),
    # Add a new option named 'gpt4o', which is equivalent to `ChatOpenAI(model="gpt-4o")`.
    gpt4o=ChatOpenAI(model="gpt-4o"),
    # You can add more configuration options here.
)

prompt = PromptTemplate.from_template(
    # Default prompt template
    "Describe {company} in 20 words or less."
).configurable_alternatives(
    # Assign an ID to this field.
    # When configuring the end runnable, we can then use this id to configure this field.
    ConfigurableField(id="prompt"),
    # Set the default key.
    default_key="description",
    # Add a new option named 'founder'.
    founder=PromptTemplate.from_template("Who is the founder of {company}?"),
    # Add a new option named 'competitor'.
    competitor=PromptTemplate.from_template("Who is the competitor of {company}?"),
    # You can add more configuration options here.
)
chain = prompt | llm
```

```python
# We can configure both the prompt and LLM simultaneously using .with_config(). Here we're using the founder prompt template with the OpenAI model.
chain.with_config(configurable={"prompt": "founder", "llm": "openai"}).invoke(
    # Request processing for the company provided by the user.
    {"company": "Apple"}
).__dict__
```




<pre class="custom">{'content': 'Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976.',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 26,
       'prompt_tokens': 14,
       'total_tokens': 40,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-mini-2024-07-18',
      'system_fingerprint': 'fp_0aa8d3e20b',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-36c146f6-d689-489f-afae-5c313b504e43-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 14,
      'output_tokens': 26,
      'total_tokens': 40,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>



```python
# If you want to configure the chain to use the Anthropic model, you can do so as follows:
chain.with_config(configurable={"llm": "anthropic"}).invoke(
    {"company": "Apple"}
).__dict__
```




<pre class="custom">{'content': 'Apple is a global technology company known for iPhones, Macs, and innovative consumer electronics, founded by Steve Jobs.',
     'additional_kwargs': {},
     'response_metadata': {'id': 'msg_01N4k3CcSbukLKN22H3FnGST',
      'model': 'claude-3-5-sonnet-20241022',
      'stop_reason': 'end_turn',
      'stop_sequence': None,
      'usage': {'cache_creation_input_tokens': 0,
       'cache_read_input_tokens': 0,
       'input_tokens': 18,
       'output_tokens': 29}},
     'type': 'ai',
     'name': None,
     'id': 'run-3a625785-71dd-4724-91f4-41a0ab157037-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 18,
      'output_tokens': 29,
      'total_tokens': 47,
      'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}</pre>



```python
# If you want to configure the chain to use the competitor prompt template, you can do so as follows:
chain.with_config(configurable={"prompt": "competitor"}).invoke(
    {"company": "Apple"}
).__dict__
```




<pre class="custom">{'content': "Apple has several major competitors across different product categories:\n\n1. Smartphones:\n- Samsung\n- Google\n- Huawei\n- Xiaomi\n- OnePlus\n\n2. Computers/Laptops:\n- Microsoft\n- Dell\n- HP\n- Lenovo\n- ASUS\n\n3. Tablets:\n- Samsung\n- Microsoft (Surface)\n- Amazon (Fire tablets)\n- Lenovo\n\n4. Smart Watches:\n- Samsung\n- Fitbit\n- Garmin\n- Huawei\n\n5. Music/Media Services:\n- Spotify\n- Amazon Music\n- YouTube Music\n- Netflix\n- Disney+\n\n6. Software/Operating Systems:\n- Microsoft\n- Google (Android/Chrome OS)\n\nSamsung is often considered Apple's biggest overall competitor, as it competes in many of the same product categories, particularly in the smartphone market. Microsoft and Google are also major competitors, especially in terms of operating systems and software services.",
     'additional_kwargs': {},
     'response_metadata': {'id': 'msg_017cQyFYwcV46bgo3ekijs7d',
      'model': 'claude-3-5-sonnet-20241022',
      'stop_reason': 'end_turn',
      'stop_sequence': None,
      'usage': {'cache_creation_input_tokens': 0,
       'cache_read_input_tokens': 0,
       'input_tokens': 14,
       'output_tokens': 217}},
     'type': 'ai',
     'name': None,
     'id': 'run-4385dff8-d1e7-4ba3-9ead-84bd335cdd69-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 14,
      'output_tokens': 217,
      'total_tokens': 231,
      'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}</pre>



```python
# If you want to use the default configuration, you can invoke the chain directly:
chain.invoke({"company": "Apple"}).__dict__
```




<pre class="custom">{'content': 'Apple is a global technology company known for iPhones, Macs, and innovative consumer electronics, founded by Steve Jobs.',
     'additional_kwargs': {},
     'response_metadata': {'id': 'msg_01V26p9PhkrZ85NATotyouiC',
      'model': 'claude-3-5-sonnet-20241022',
      'stop_reason': 'end_turn',
      'stop_sequence': None,
      'usage': {'cache_creation_input_tokens': 0,
       'cache_read_input_tokens': 0,
       'input_tokens': 18,
       'output_tokens': 29}},
     'type': 'ai',
     'name': None,
     'id': 'run-b34f7b20-1fff-4f49-8264-13382f91bfe1-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 18,
      'output_tokens': 29,
      'total_tokens': 47,
      'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}</pre>



## Saving Configurations

You can easily save configured chains as reusable objects. For example, after configuring a chain for a specific task, you can save it for later use in similar tasks.

```python
# Save the configured chain to a new variable.
gpt4o_competitor_chain = chain.with_config(
    configurable={"llm": "gpt4o", "prompt": "competitor"}
)
```

```python
# Call the chain.
gpt4o_competitor_chain.invoke({"company": "Apple"}).__dict__
```




<pre class="custom">{'content': 'Apple has several competitors across its different product lines. Some of the main competitors include:\n\n1. **Smartphones**: \n   - Samsung\n   - Google (Pixel)\n   - Huawei\n   - Xiaomi\n\n2. **Computers and Laptops**:\n   - Microsoft\n   - Dell\n   - HP\n   - Lenovo\n\n3. **Tablets**:\n   - Samsung\n   - Microsoft (Surface)\n   - Amazon (Fire tablets)\n\n4. **Wearable Technology**:\n   - Samsung\n   - Fitbit\n   - Garmin\n\n5. **Smart Speakers**:\n   - Amazon (Echo)\n   - Google (Nest)\n\n6. **Streaming Services**:\n   - Spotify (for Apple Music)\n   - Netflix, Disney+, Amazon Prime Video (for Apple TV+)\n\n7. **Cloud Services**:\n   - Google\n   - Microsoft (OneDrive)\n   - Amazon (AWS)\n\nThese competitors challenge Apple in different markets with their own range of products and services.',
     'additional_kwargs': {'refusal': None},
     'response_metadata': {'token_usage': {'completion_tokens': 203,
       'prompt_tokens': 14,
       'total_tokens': 217,
       'completion_tokens_details': {'accepted_prediction_tokens': 0,
        'audio_tokens': 0,
        'reasoning_tokens': 0,
        'rejected_prediction_tokens': 0},
       'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
      'model_name': 'gpt-4o-2024-08-06',
      'system_fingerprint': 'fp_f785eb5f47',
      'finish_reason': 'stop',
      'logprobs': None},
     'type': 'ai',
     'name': None,
     'id': 'run-76f4f809-530d-4a54-a522-8777955a0763-0',
     'example': False,
     'tool_calls': [],
     'invalid_tool_calls': [],
     'usage_metadata': {'input_tokens': 14,
      'output_tokens': 203,
      'total_tokens': 217,
      'input_token_details': {'audio': 0, 'cache_read': 0},
      'output_token_details': {'audio': 0, 'reasoning': 0}}}</pre>


