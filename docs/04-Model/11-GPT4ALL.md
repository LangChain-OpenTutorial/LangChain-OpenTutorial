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

# GPT4ALL

- Author: [Yoonji Oh](https://github.com/samdaseuss)
- Peer Review : [Joseph](https://github.com/XaviereKU), [Normalist-K](https://github.com/Normalist-K)
- Proofread : [frimer](https://github.com/brian604)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/11-GPT4ALL.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/11-GPT4ALL.ipynb)
## Overview
In this tutorial, we’re exploring ```GPT4ALL``` together! From picking the perfect model for your hardware to running it on your own, we’ll walk you through the process **step by step**. 

Ready? Let’s dive in and have some fun along the way!

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Installation](#installation)
- [What is GPT4ALL](#what-is-gpt4all)
- [Choosing a Model](#choosing-a-model)
- [Downloading a Model](#downloading-a-model)
- [Running GPT4ALL Models](#running-gpt4all-models)
- [Summary](#summary)

### References

- [GPT4All Python SDK](https://docs.gpt4all.io/gpt4all_python/home.html)
- [GPT4ALL docs](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.gpt4all.GPT4All.html#langchain_community.llms.gpt4all.GPT4All.backend)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "LANGCHAIN_PROJECT": "GPT4ALL",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can also create and use a ```.env``` file in the root directory as shown below.

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



## Installation

Ready to get started with ```gpt4all```? Let’s make sure you’ve got everything set up! We’ll guide you through installing the package using ```pip``` or ```poetry```. Don’t worry, it’s easy and quick.

---

### **Install the Python Package**

You can install ```gpt4all``` using **pip** or **poetry**, depending on your preferred package manager. Here’s how:

#### **1. Installation using pip**

If you’re using ```pip```, run the following command in your terminal:


```python
!pip install -qU gpt4all
```


#### **2. Installation using poetry**

Prefer ```poetry```? No problem! Here’s how to install ```gpt4all``` using poetry:


**Step 1: Add ```gpt4all``` to your project**  
Run this command to add the package to your ```pyproject.toml```:

```python
!poetry add gpt4all
```


**Step 2: Install dependencies**  
If the package is already added but not installed, simply run:

```python
!poetry install
```

Poetry will sync your environment and install all required dependencies.

## What is GPT4ALL

```GitHub:nomic-ai/gpt4all``` is an open-source chatbot ecosystem trained on a large amount of data, including code and chat-form conversations. 

In this example, we will explain how to interact with the GPT4All model using LangChain.

<div style="text-align: center;">
    <img src="./assets/10-GPT4ALL-01.png" alt="Image Description" width="500">
</div>

## Choosing a Model

It's the most crucial and decision-making time. Before diving into writing code, it's time to decide which model to use. Below, we explore popular models and help you choose the right one based on GPT4All's [Python Documentation](https://docs.gpt4all.io/gpt4all_python/home.html#load-llm).

---

### Model Selection Criteria
| **Model Name**                          | **Filesize** | **RAM Required** | **Parameters** | **Quantization** | **Developer**          | **License**           | **MD5 Sum (Unique Hash)**            |
|-----------------------------------------|--------------|------------------|----------------|------------------|------------------------|-----------------------|--------------------------------------|
| **Meta-Llama-3-8B-Instruct.Q4_0.gguf**  | 4.66 GB      | 8 GB            | 8 Billion      | q4_0             | Meta                   | Llama 3 License       | c87ad09e1e4c8f9c35a5fcef52b6f1c9    |
| **Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf** | 4.11 GB   | 8 GB            | 7 Billion      | q4_0             | Mistral & Nous Research | Apache 2.0           | Coa5f6b4eabd3992da4d7fb7f020f921eb  |
| **Phi-3-mini-4k-instruct.Q4_0.gguf**    | 2.18 GB      | 4 GB            | 3.8 Billion    | q4_0             | Microsoft              | MIT                   | f8347badde9bfc2efbe89124d78ddaf5    |
| **orca-mini-3b-gguf2-q4_0.gguf**        | 1.98 GB      | 4 GB            | 3 Billion      | q4_0             | Microsoft              | CC-BY-NC-SA-4.0       | 0e769317b90ac30d6e09486d61fefa26    |
| **gpt4all-13b-snoozy-q4_0.gguf**        | 7.37 GB      | 16 GB           | 13 Billion     | q4_0             | Nomic AI               | GPL                   | 40388eb2f8d16bb5d08c96fdfaac6b2c    |

#### Based on Use Case
Choose your model depending on the tasks you plan to perform:
- **Lightweight tasks** (e.g., simple conversation): ```orca-mini-3b-gguf2-q4_0.gguf``` or ```Phi-3-mini-4k-instruct.Q4_0.gguf```.  
- **Moderate tasks** (e.g., summarization or grammar correction): ```Meta-Llama-3-8B-Instruct.Q4_0.gguf``` or ```Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf```.  
- **Advanced tasks** (e.g., long text generation, research): ```gpt4all-13b-snoozy-q4_0.gguf```.

#### Based on System Specifications
Select a model based on your available hardware:
- For **4GB RAM or less**, use ```orca-mini-3b-gguf2-q4_0.gguf``` or ```Phi-3-mini-4k-instruct.Q4_0.gguf```.  
- For **8GB RAM or more**, use ```Meta-Llama-3-8B-Instruct.Q4_0.gguf``` or ```Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf```.  
- For **16GB RAM or more**, use ```gpt4all-13b-snoozy-q4_0.gguf```.


[NOTE]

- **```GGML```**: CPU-friendly and low memory usage.  
- **```GGUF```**: Latest format with GPU acceleration support.  
- **```q4_0 Quantization```**: Efficient for both CPU and GPU workloads, with reduced memory requirements.

## Downloading a Model
In this tutorial, we will be using Microsoft's ```Phi-3-Mini-4K-Instruct``` model.

1. **Download the Model**: Visit [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main) to download the required model (2.39 GB).

<div style="text-align: center;">
    <img src="./assets/10-GPT4ALL-02.png" alt="Image Description" width="500">
</div>

2. **Load Models in Python**: After downloading the model, create a folder named ```models``` and place the downloaded file in that folder. 

<div style="text-align: center;">
    <img src="./assets/10-GPT4ALL-03.png" alt="Image Description" width="500">
</div>

- Assign the local file path (e.g., ```Phi-3-mini-4k-instruct-q4.gguf```) to the ```local_path``` variable.


```python
local_path = "./models/Phi-3-mini-4k-instruct-q4.gguf"  # Replace with your desired local file path.
```

- You can replace this path with any local file path you prefer.

Use the [Python Documentation](https://docs.gpt4all.io/gpt4all_python/home.html#load-llm) to load and run your selected model in your project.

## Running GPT4ALL Models

```GPT4All``` is a ```powerful large-scale language model```, similar to ```GPT-3```, designed to support a variety of natural language processing tasks.  

This module allows you to ```easily load the GPT4All model``` and perform inference seamlessly.  

In the following example, we demonstrate how to ```load the GPT4All model``` and utilize it to ```answer a question``` by leveraging a ```custom prompt``` and ```inference pipeline```.

```python
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.llms.gpt4all import GPT4All
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
```

[NOTE]

Due to structural changes, in version ```0.3.13```, you need to replace ```from langchain.prompts import ChatPromptTemplate``` with ```from langchain_core.prompts import ChatPromptTemplate```.

### Creating a Prompt and Checking the Result

```python
template = """
<s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>
<s>Human: {question}</s>
<s>Assistant:
"""

prompt = ChatPromptTemplate.from_template(template)

result = prompt.invoke({"question": "where is the capital of United States?"})

print(result.messages[0].content)
```

<pre class="custom">
    <s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>
    <s>Human: where is the capital of United States?</s>
    <s>Assistant:
    
</pre>

The ```ChatPromptTemplate``` is responsible for creating prompt templates in LangChain and dynamically substituting variables. Without using the ```invoke()``` method, you can utilize the class's template methods to generate prompts. In this case, the template can be returned as a string using the ```format``` method.

```python
# Using format() instead of invoke()
result = prompt.format(question="What is the capital of United States?")
```

In a nutshell, the ```invoke()``` method is great for ```chain-based tasks```, while the ```format()``` method is perfect for ```returning simple strings```. 

|       | **.invoke()** | **.format()**  |
|-------|---------------|----------------|
| **Purpose**      | Creates a structured object for chain execution | Returns the template as a simple string |
| **Return Value** | ChatPromptValue object (structured LangChain object) | String                              |
| **Use Case**     | Used in LangChain chain operations          | Used for simple prompt generation and testing |
| **Complexity**   | Relatively complex                          | Simple and intuitive                |


```python
result = prompt.format(question="where is the capital of United States?")

print(result)
```

<pre class="custom">Human: 
    <s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>
    <s>Human: where is the capital of United States?</s>
    <s>Assistant:
    
</pre>

You might notice that ```Human:``` is automatically added to the output. If you'd like to avoid this behavior, you can use LangChain's ```PromptTemplate``` class instead of ```ChatPromptTemplate```. The ```PromptTemplate``` class doesn’t add any extra prefixes like this.

```python
from langchain_core.prompts.prompt import PromptTemplate

prompt = PromptTemplate.from_template(template)
formatted_prompt = prompt.format(question="Where is the capital of the United States?")
print(formatted_prompt)
```

<pre class="custom">
    <s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>
    <s>Human: Where is the capital of the United States?</s>
    <s>Assistant:
    
</pre>

We'll be using invoke for ```chain-based tasks```, so go ahead and forget about the format method for now! 😉

### Using Chains to Display Results in Real-Time

```python
# Prompt
prompt = ChatPromptTemplate.from_template(
    """ 
    <s>A chat between a user and an AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>
    <s>Human: {question}</s>
    <s>Assistant:
"""
)

# GPT4All Language Model Initialization
# Specify the path to the GPT4All model file in model
model = GPT4All(
    model=local_path,
    n_threads=8, # Number of threads to use.
    backend="gpu", # GPU Configuration
    streaming=True, # Streaming Configuration
    callbacks=[StreamingStdOutCallbackHandler()] # Callback Configuration
)

# Create the chain
chain = prompt | model | StrOutputParser()

# Execute the query
response = chain.invoke({"question": "where is the capital of United States?"})
```

<pre class="custom">===
    The capital of the United States is Washington, D.C., which stands for District of Columbia. It was established by the Constitution along with a federal district that would serve as the nation's seat of government and be under its exclusive jurisdiction. The city itself lies on the east bank of the Potomac River near its fall point where it empties into Chesapeake Bay, but Washington is not part of any U.S. state; instead, it has a special federal district status as outlined in Article I, Section 8 of the Constitution and further defined by the Residence Act of 1790 signed by President George Washington.
    
    Washington D.C.'s location was chosen to be near the nation's capital at that time—Philadelphia, Pennsylvania—and it also holds symbolic significance as a neutral ground for both northern and southern states during their early years in America. The city is home to many iconic landmarks such as the U.S. Capitol Building where Congress meets, the White House (the residence of the President), Supreme Court buildings, numerous museums like the Smithsonian Institution's National Museum of American History or Natural History and Air & Space, among others</pre>

## Summary

Today, we explored GPT4ALL together! **We didn’t just run models** — we took part in the decision-making process, from selecting a model to suit our needs to choosing the right methods based on our desired outcomes or execution direction. Along the way, we compared the performance of popular models and even ran the code ourselves.

Next time, we’ll dive into ```Video Q&A LLM (Gemini)```. Until then, try running today’s code with different models and see how they perform. See you soon! 😊
