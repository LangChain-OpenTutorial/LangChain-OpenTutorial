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

# HuggingFace Pipeline

- Author: [Sunworl Kim](https://github.com/sunworl)
- Peer Review: [effort-type](https://github.com/effort-type), [sunworl](https://github.com/sunworl), [ivybae](https://github.com/ivybae)
- Proofread : [frimer](https://github.com/brian604)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/09-HuggingFace-Pipelines.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/09-HuggingFace-Pipelines.ipynb)


## Overview

This tutorial covers how to run Hugging Face models locally through the **HuggingFacePipeline** class.

It explains how to load a model by specifying model parameters using the **from_model_id** method or by directly passing the **transformers pipeline**.

Using the generated **hf** object, it implements text generation for a given prompt.

By specifying parameters for the device, it also implements execution on a GPU device and batching.

- **Advantages**  
    - No usage fees.  
    - Lower risk of data leakage.  

- **Disadvantages**  
    - Requires significant computational resources.   


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Hugging Face Local pipelines](#hugging-face-local-pipelines)
- [Model Loading](#model-loading)
- [Usage of Gated Model](#usage-of-gated-model)
- [Create Chain](#create-chain)
- [GPU Inference](#gpu-inference)
- [Batch GPU Inference](#batch-gpu-inference) 


### References

- [Langchain: Hugging Face Local Pipelines](https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/)
- [Hugging Face: Phi-3-mini-4k-instuct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) 
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
        "langchain_core",
        "langchain.prompts",
        "langchain_huggingface",
        "huggingface_hub"
    ],
    verbose=False,
    upgrade=False,
)
```

```python
# Set environment variables
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "HUGGINGFACEHUB_API_TOKEN": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "Huggingface-Piplines",
        }
    )
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## Hugging Face Local Pipelines

The Hugging Face models can be run locally through the ```HuggingFacePipeline``` class.

The [Hugging Face model Hub](https://huggingface.co/models) hosts over 120k models, 20k datasets, and 50k demo apps (Spaces) on its online platform, all of which are open-source and publicly available, allowing people to easily collaborate and build ML together.

These can be used in LangChain either by calling them through this local pipeline wrapper or by calling hosted inference endpoints through the HuggingFaseHub class. For more information on hosted pipelines, please refer to the [HuggingFaseHub](https://huggingface.co/models) notebook.

To use this, you should have the [transformers python package](https://pypi.org/project/transformers/) installed, as well as [PyTorch](https://pytorch.org/get-started/locally/).

Additionally, you may install ```xformers``` for a more memory-efficient attention implementation.

```python
!pip install -qU transformers
!pip install -qU ipywidgets
```

<pre class="custom">
    [notice] A new release of pip is available: 24.0 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    
    [notice] A new release of pip is available: 24.0 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

```python
# installation pytorch
# !pip install --force-reinstall --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu121
```

Set the path to download the model.

```python
# Path to download Hugging Face models/tokenizers
import os

# ./cache/ Set to download to the specified path
os.environ["HF_HOME"] = "./cache/"
```

## Model Loading

Models can be loaded by specifying model parameters using the method ```from_model_id```.


- The ```langchain-opentutorial``` class is used to load a pre-trained model from Hugging Face.

- The ```from_model_id``` method is used to specify the ```microsoft/Phi-3-mini-4k-instruct``` model and set the task to "text-generation".

- The ```pipeline_kwargs``` parameter is used to limit the maximum number of tokens to be generated to 64.

- The loaded model is assigned to the ```hf``` variable, which can be used to perform text generation tasks.

The model used: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

```python
from langchain_huggingface import HuggingFacePipeline

# Download the HuggingFace model.
hf = HuggingFacePipeline.from_model_id(   
    model_id="microsoft/Phi-3-mini-4k-instruct",  # Specify the ID of the model to use.  
    task="text-generation",  # Specify the task to perform. Here, it's text generation.        
    pipeline_kwargs={"max_new_tokens": 64},  # Set additional arguments to pass to the pipeline. Here, we limit the maximum number of new tokens to 64.
)
```


<pre class="custom">Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]</pre>


    Device set to use cpu
    

You can also load by directly passing an existing ```transformers``` pipeline.

The text ageneration model is implemented using HuggingFacePipeline.


- ```AutoTokenizer``` and ```AutoModelForCausalLM``` are used to load the ```microsoft/Phi-3-mini-4k-instruct``` model and tokenizer.

- The ```pipeline``` function is used to create a "text-generation" pipeline, setting up the model and tokenizer. The maximum number of generated tokens is limited to 64.

- The ```HuggingFacePipeline``` class is used to create an ```hf``` object, and the generated pipeline is passed to it.


Using this created ```hf``` object, you can perform text generation for a given prompt.

```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Specify the ID of the model to use.
model_id = "microsoft/Phi-3-mini-4k-instruct" 
# Load the tokenizer for the specified model. 
tokenizer = AutoTokenizer.from_pretrained(model_id) 

# Load the specified model.
model = AutoModelForCausalLM.from_pretrained(model_id)  

# Create a text generation pipeline and set the maximum number of new tokens to be generated to 64.
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)

# Create a HuggingFacePipeline object and pass the generated pipeline to it.
hf = HuggingFacePipeline(pipeline=pipe)
```


<pre class="custom">Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]</pre>


    Device set to use cpu
    

## Usage of Gated Model

The ```Gated Model``` is a model that can be used under a license agreement from Hugging Face.

You must first visit the model page and agree to the terms before obtaining a Hugging Face token.

Below is an example of how to use the ```Gated``` Model. You need to specify the Hugging Face token as shown below.

```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Specify the model ID registered in the Hugging Face repository.
model_id = "microsoft/Phi-3-mini-4k-instruct" 

# Enter the Hugging Face token you received here.
your_huggingface_token = ""

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id, token=your_huggingface_token)

# Load the specified model.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=your_huggingface_token,
    # load_in_4bit=True, # If bitsandbytes is installed (Linux)
    # attn_implementation="flash_attention_2", # If you have an Ampere GPU
)

# Create the pipeline.
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)

# Create a HuggingFacePipeline object and pass the created pipeline.
hf_llm = HuggingFacePipeline(pipeline=pipe)
```


<pre class="custom">Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]</pre>


    Device set to use cpu
    

Execute and check the results

```python
for token in hf_llm.stream("What is the capital of France?"):
    print(token, end="", flush=True)
```

<pre class="custom">
    
    # Answer
    The capital of France is Paris.</pre>

## Create Chain

Once the model is loaded into memory, you can configure it with prompts to form a chain.


- A prompt template defining the question and answer format is created using the ```PromptTemplate``` class.

- Create a ```chain``` object by connecting the ```prompt``` object and the ```hf``` object in a pipeline.

- Call the ```chain.invoke()``` method to generate and output an answer for the given question.

```python
from huggingface_hub import login

login()
```


<pre class="custom">VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…</pre>


```python
from langchain_huggingface import ChatHuggingFace

llm = ChatHuggingFace(llm=hf)
```

```python
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """<|system|>You are a helpful assistant.<|end|>
<|user|>{question}<|end|>
<|assistant|>"""  # Template for defining question and answer formats

prompt = PromptTemplate.from_template(template)  # Create a prompt object using the template

# Create a chain by connecting the prompt and the language model
chain = prompt |llm| StrOutputParser()
question = "What is the capital of the United France?"  # Define the question
print(
    chain.invoke({"question": question})
)  # Call the chain to generate and output an answer to the question

```

<pre class="custom"><|user|>
    <|system|>You are a helpful assistant.<|end|>
    <|user|>What is the capital of the United France?<|end|>
    <|assistant|><|end|>
    <|assistant|>
     The capital of France is Paris.
</pre>

## GPU Inference

When running on a GPU, you can specify the ```device=n``` parameter to place the model on a specific device.

The default value is ```-1```, which means inference is performed on the CPU.

If you have multiple GPUs or if the model is too large for a single GPU, you can specify ```device_map="auto"```.

In this case, the [Accelerate](https://huggingface.co/docs/accelerate/index) library is required and is used to automatically determine how to load the model weights.

*Caution*: ```device``` and ```device_map``` should not be specified together, as this can cause unexpected behavior.



- Load the ```gpt2``` model using ```HuggingFacePipeline``` and set the ```device``` parameter to 0 to run it on the GPU.

- Limit the maximum number of tokens to be generated to 64 using the ```pipeline_kwargs``` parameter.

- Connect the ```prompt``` and ```gpu_llm``` in a pipeline to create the ```gpu_chain```.

- Call the ```gpu_chain.invoke()``` method to generate and output an answer for the given question.

```python
gpu_llm = HuggingFacePipeline.from_model_id(
    
    model_id="microsoft/Phi-3-mini-4k-instruct", 
    task="text-generation",      
    device=-1,    # Specifies the GPU device number. -1 stands for CPU.   
    pipeline_kwargs={"max_new_tokens": 64},  # Set additional arguments to be passed to the pipeline. In this case, limit the maximum number of tokens to be generated to 64.
)

prompt = PromptTemplate.from_template(template)  # Create a prompt object using the template

# Create a chain by connecting the prompt and the language model.
gpu_chain = prompt | gpu_llm | StrOutputParser()

question = "What is the capital of France?" 

#Invoke the chain to generate and output the answer to the question
print(gpu_chain.invoke({"question": question}))
```


<pre class="custom">Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]</pre>


    Device set to use cpu
    

    <|system|>You are a helpful assistant.<|end|>
    <|user|>What is the capital of France?<|end|>
    <|assistant|> The capital of France is Paris. It is not only the largest city in France but also one of the most important cultural and commercial centers in Europe. Paris is known for its historical landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which is the world
    

## Batch GPU Inference

When running on a GPU device, you can perform inference in batch mode on the GPU.


- Load the ```microsoft/Phi-3-mini-4k-instruct``` model using ```HuggingFacePipeline``` and set it to run on the GPU.

- When creating the ```gpu_llm```, set the ```batch_size``` to 2, ```temperature``` to 0, and ```max_length``` to 64.

- Connect the ```prompt``` and ```gpu_llm``` in a pipeline to create the ```gpu_chain```, and set the end token to "\n\n".

- Use ```gpu_chain.batch()``` to generate answers in parallel for the ```questions``` in the questions.

- Wrap each answer with <answer> tags and separate each answer with a line break.

```python
gpu_llm = HuggingFacePipeline.from_model_id(

    model_id="microsoft/Phi-3-mini-4k-instruct", 
    task="text-generation",    
    device=-1, # Specifies the GPU device number. -1 stands for CPU.       
    batch_size=2,  # Adjust the batch size. Set it appropriately based on GPU memory and model size.
    model_kwargs={
        "temperature": 0,
        "max_length": 64,
        "do_sample": True
    },  # Set additional arguments to be passed to the model.
)

# Create a chain by connecting the prompt and the language model.
gpu_chain = prompt | gpu_llm.bind(stop=["\n\n"]) 

questions = []
for i in range(4):
    # Generate a list of questions
    questions.append({"question": f"What is the number {i} in English?"})

answers = gpu_chain.batch(questions) 
for answer in answers:
    print(answer)
    print("")

```


<pre class="custom">Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]</pre>


    Device set to use cpu
    

    <|system|>You are a helpful assistant.<|end|>
    <|user|>What is the number 0 in English?<|end|>
    <|assistant|> The number 0 in English is called "zero."
    
    <|system|>You are a helpful assistant.<|end|>
    <|user|>What is the number 1 in English?<|end|>
    <|assistant|> The number 1 in English is simply called "one."
    
    <|system|>You are a helpful assistant.<|end|>
    <|user|>What is the number 2 in English?<|end|>
    <|assistant|> The number 2 in English is spelled "two."
    
    <|system|>You are a helpful assistant.<|end|>
    <|user|>What is the number 3 in English?<|end|>
    <|assistant|> The number 3 in English is spelled "three."
    
    
