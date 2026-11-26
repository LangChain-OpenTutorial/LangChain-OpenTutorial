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
- Peer Review: 
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/06-Configure-Runtime-Chain-Components.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/06-Configure-Runtime-Chain-Components.ipynb)
## Overview

In this tutorial, we will explore how to dynamically configure various options when calling a chain.

There are two ways to implement dynamic configuration:

- First, the ```configurable_fields``` method allows you to configure specific fields of a Runnable object.
    - Dynamically modify specific field values at runtime
    - Example: Adjust individual parameters like ```temperature```, ```model_name``` of an LLM

- Second, the ```configurable_alternatives``` method lets you specify alternatives for a particular Runnable object that can be set during runtime
    - Replace entire components with alternatives at runtime
    - Example: Switch between different LLM models or prompt templates

**[Note]**
The term **Configurable fields** refers to settings or parameters within a system that can be adjusted or modified by the user or administrator at runtime.

- Applying configuration
    - ```with_config``` method: A unified interface for applying all configuration settings
    - Ability to apply single or multiple settings simultaneously
    - Used consistently across special components like ```HubRunnable```

In the following sections, we'll cover detailed usage of each method and practical applications. We'll explore real-world examples including prompt management through ```HubRunnable``` setting various prompt alternatives, switching between LLM models, and more.

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
- The ```langchain-opentutorial``` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```


    Running cells with 'Python 3.9.6' requires the ipykernel package.
    

    Run the following command to install 'ipykernel' into the Python environment. 
    

    Command: '/usr/bin/python3 -m pip install ipykernel -U --user --force-reinstall'


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

Alternatively, you can set and load ```OPENAI_API_KEY``` from a ```.env``` file.

**[Note]** This is only necessary if you haven't already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```

## Configurable Fields

```Configurable fields``` provide a way to dynamically modify specific parameters of a Runnable object at runtime. This feature is essential when you need to fine-tune the behavior of your chains or models without changing their core implementation.

- They allow you to specify which parameters can be modified during execution
- Each configurable field can include a description that explains its purpose
- You can configure multiple fields simultaneously
- The original chain structure remains unchanged, even when you modify configurations for different runs.

The ```configurable_fields``` method is used to specify which parameters should be treated as configurable, making your LangChain applications more flexible and adaptable to different use cases.

### Dynamic Property Configuration

Let's illustrate this with ```ChatOpenAI```. When using ChatOpenAI, we can set various properties.

The ```model_name``` property is used to specify the version of GPT. For example, you can select different models by setting it to ```gpt-4o```, ```gpt-4o-mini```, or else.

To dynamically specify the model instead of using a fixed ```model_name```, you can leverage the ```ConfigurableField``` and assign it to a dynamically configurable property value as follows:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, model_name="gpt-4o")

model.invoke("Where is the capital of the United States?").__dict__
```

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

When calling ```model.invoke()```, you can dynamically specify parameters using the format ```config={"configurable": {"key": "value"}}```.

```python
model.invoke(
    "Where is the capital of the United States?",
    # Set gpt_version to gpt-3.5-turbo
    config={"configurable": {"gpt_version": "gpt-3.5-turbo"}},
).__dict__
```

Now let's try using the ```gpt-4o-mini``` model. Check the output to see the changed model.

```python
model.invoke(
    # Set gpt_version to gpt-4o-mini
    "Where is the capital of the United States?",
    config={"configurable": {"gpt_version": "gpt-4o-mini"}},
).__dict__
```

Alternatively, you can set ```configurable``` parameters using the ```with_config()``` method of the ```model``` object to achieve the same result.

```python
model.with_config(configurable={"gpt_version": "gpt-4o-mini"}).invoke(
    "Where is the capital of the United States?",
).__dict__
```

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

```python
# Call the chain with configuration settings
chain.with_config(configurable={"gpt_version": "gpt-4o"}).invoke({"x": 0}).__dict__
```

## Configurable Alternatives with HubRunnables

Using ```HubRunnable``` simplifies dynamic prompt selection, allowing easy switching between prompts registered in the Hub

### Configuring LangChain Hub Settings

```HubRunnable``` provide an option to configure which prompt template to pull from the LangChain Hub. This enables you to dynamically select different prompts based on the hub path specification.

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

If you call the ```prompt.invoke()``` method without specifying a ```with_config```, the Runnable will automatically pull and use the prompt that was initially registered in the set **\"rlm/rag-prompt\"** hub.

```python
# Call the prompt object's invoke method with "question" and "context" parameters
prompt.invoke({"question": "Hello", "context": "World"}).messages
```

```python
prompt.with_config(
    # Set hub_commit to teddynote/summary-stuff-documents
    configurable={"hub_commit": "teddynote/summary-stuff-documents"}
).invoke({"context": "Hello"})
```

## Switching between Runnables

**Configurable alternatives** provide a way to select between different Runnable objects that can be set at runtime.

For example, the configurable language model of ```ChatAnthropic``` provides high degree of flexibility that can be applied to various tasks and contexts.

To enable dynamic switching, we can define the model's parameters as ```ConfigurableField``` objects.

- ```model```: Specifies the base language model to be used.

- ```temperature```: Controls the randomness of the model's sampling (which values between 0 and 1). Lower values result in more deterministic and repetitive outputs, while higher values lead to more diverse and creative responses.

### Setting Alternatives for LLM Objects

Let's explore how to implement configurable alternatives using a Large Language Model (LLM).

[Note]

- To use the ```ChatAnthropic``` model, you need to obtain an API key from the Anthropic console: https://console.anthropic.com/dashboard.
- You can uncomment and directly set the API key (as shown below) or store it in your ```.env``` file.

Set the ```ANTHROPIC_API_KEY``` environment variable in your code.

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

Here's how you can invoke a chain using the default ```ChatAnthropic``` model using ```chain.invoke()```.

```python
# Invoke using Anthropic as the default.
chain.invoke({"topic": "NewJeans"}).__dict__
```

You may specify a different model to use the ```llm``` by using ```chain.with_config(configurable={"llm": "model"})```.

```python
# Invoke by changing the chain's configuration.
chain.with_config(configurable={"llm": "openai"}).invoke({"topic": "NewJeans"}).__dict__
```

Now, change the chain's configuration to use ```gpt4o``` as the language model.

```python
# Invoke by changing the chain's configuration.
chain.with_config(configurable={"llm": "gpt4o"}).invoke({"topic": "NewJeans"}).__dict__
```

For this time, change the chain's configuration to use ```anthropic```.

```python
# Invoke by changing the chain's configuration.
chain.with_config(configurable={"llm": "anthropic"}).invoke(
    {"topic": "NewJeans"}
).__dict__
```

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

To use a different prompt, use ```with_config```.

```python
# Call the chain by changing the chain's configuration using with_config.
chain.with_config(configurable={"prompt": "area"}).invoke({"country": "South Korea"})
```

```python
# Call the chain by changing the chain's configuration using with_config.
chain.with_config(configurable={"prompt": "population"}).invoke({"country": "South Korea"})
```

Now let's use the ```kor``` prompt to request a translation, for example, pass the input using the ```input``` variable.

```python
# Call the chain by changing the chain's configuration using with_config.
chain.with_config(configurable={"prompt": "kor"}).invoke({"input": "apple is delicious!"})
```

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

```python
# If you want to configure the chain to use the Anthropic model, you can do so as follows:
chain.with_config(configurable={"llm": "anthropic"}).invoke(
    {"company": "Apple"}
).__dict__
```

```python
# If you want to configure the chain to use the competitor prompt template, you can do so as follows:
chain.with_config(configurable={"prompt": "competitor"}).invoke(
    {"company": "Apple"}
).__dict__
```

```python
# If you want to use the default configuration, you can invoke the chain directly:
chain.invoke({"company": "Apple"}).__dict__
```

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
