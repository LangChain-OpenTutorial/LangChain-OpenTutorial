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

# @chain decorator

- Author: [Yejin Park](https://github.com/ppakyeah)
- Peer Review: []()
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial explains how to convert regular functions into Runnable objects using the `@chain` decorator.

It covers ChatPromptTemplate for creating prompts, function transformation with the `@chain` decorator, and chain monitoring through LangSmith.

The practical exercise builds a custom chain that converts text into Instagram-style posts with emojis.


### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Create a Runnable object using the @chain decorator](#create-a-runnable-object-using-the-@chain-decorator)
- [Comparison with RunnableLambda](#comparison-with-runnablelambda)

### References

- [LangChain: The convenience @chain decorator](https://python.langchain.com/docs/how_to/functions/#the-convenience-chain-decorator)
- [Chain API Referece](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.chain.html)
----

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
        "langchain_core",
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
        "LANGCHAIN_PROJECT": "@chain-decorator",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Create a Runnable object using the @chain decorator

You can convert any function into a chain by adding the `@chain` decorator.

This is functionally equivalent to wrapping the function in `RunnableLambda` .

Let's see how it works in practice!

Define two prompt templates using the ChatPromptTemplate class.

- `prompt1` requests a brief description of a given topic.
- `prompt2` requests to create an Instagram post using emojis.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

# Define prompt templates
prompt1 = ChatPromptTemplate.from_template("Please provide a brief description in English about {topic}.")
prompt2 = ChatPromptTemplate.from_template(
    "Please create an Instagram post using emojis for the following text: {sentence}"
)
```

The `custom_chain` function executes a custom chain based on the input text.
- The function is decorated with `@chain` , which makes it a `Runnable` object through decoration.

```python
@chain
def custom_chain(text):
    # Create a chain by connecting the first prompt, ChatOpenAI, and string output parser
    chain1 = prompt1 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    output1 = chain1.invoke({"topic": text})

    # Create a chain by connecting the second prompt, ChatOpenAI, and string output parser
    chain2 = prompt2 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    # Call the second chain with the parsed first result and return the final result
    return chain2.invoke({"sentence": output1})
```

Since custom_chain is now a runnable object, it must be executed using `invoke()` .

```python
# Call custom_chain
print(custom_chain.invoke("quantum mechanics"))
```

<pre class="custom">🌌✨ Dive into the mysterious world of #QuantumMechanics! 🔬💫 
    
    This fundamental branch of physics reveals how matter and energy behave at the tiniest scales, like atoms and subatomic particles. 🧬⚛️ 
    
    🌀 **Wave-Particle Duality**: Electrons can be both waves and particles! 🌊➡️⚛️ 
    
    🔄 **Superposition**: Systems can exist in multiple states at once! 🎭✨ 
    
    🔗 **Entanglement**: Particles can be connected in a way that the state of one affects the other, no matter the distance! 🌌❤️ 
    
    These mind-blowing concepts are shaping our understanding of the universe and powering technologies like semiconductors, lasers, and quantum computing! 💻🔋 
    
    #Physics #Science #Universe #Technology #Innovation #Quantum #ExploreTheUnknown 🚀🔍🧪
</pre>

## Comparison with RunnableLambda

The same functionality can also be implemented using `RunnableLambda` .

Choose the `@chain` decorator when you want a cleaner, more maintainable way to create Runnable objects in your LangChain applications.

```python
from langchain_core.runnables import RunnableLambda

# Using RunnableLambda
def instagram_post_generator(text):
    chain1 = prompt1 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    output1 = chain1.invoke({"topic": text})
    chain2 = prompt2 | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    return chain2.invoke({"sentence": output1})

runnable_chain = RunnableLambda(instagram_post_generator)
print(runnable_chain.invoke("quantum mechanics"))
```

<pre class="custom">🌌✨ Dive into the fascinating world of quantum mechanics! 🔬💫 
    
    🧬 This fundamental branch of physics explores the behavior of matter and energy at the tiniest scales—think atoms and subatomic particles! ⚛️💥
    
    📏 Unlike classical mechanics, quantum mechanics introduces mind-bending concepts like:
    🌊🌀 Wave-particle duality (particles can be both wave-like and particle-like!),
    🔄 Superposition (particles can exist in multiple states at once!),
    🔗 Entanglement (where particles are interconnected across distances! 😲)
    
    🔍 These principles reshape our understanding of reality and fuel groundbreaking technologies such as:
    💻 Quantum computing,
    💡 Lasers,
    🔌 Semiconductors!
    
    Join us on this journey to unravel the mysteries of the quantum realm! 🌟🔭 #QuantumMechanics #Physics #ScienceIsCool #WaveParticleDuality #QuantumComputing #Entanglement #Superposition #Technology #NatureOfReality
</pre>
