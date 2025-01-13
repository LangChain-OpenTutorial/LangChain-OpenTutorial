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

# Routing

- Author: [Jinu Cho](https://github.com/jinucho)
- Peer Review: 
- Proofread:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/06-ConversationSummaryMemory.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/06-ConversationSummaryMemory.ipynb)

## Overview

This tutorial introduces two key tools in LangChain: `RunnableBranch` and `RunnableLambda` , essential for managing dynamic workflows and conditional logic.  

`RunnableBranch` enables structured decision-making by routing input through predefined conditions, simplifying complex branching scenarios.  

`RunnableLambda` offers a flexible, function-based approach, ideal for lightweight transformations and inline processing.  

Through detailed explanations, practical examples, and comparisons, you'll gain clarity on when and how to use each tool effectively.  

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is the RunnableBranch](#what-is-the-runnablebranch)
- [RunnableLambda](#RunnableLambda)
- [RunnableBranch](#RunnableBranch)
- [Comparison of RunnableBranch and RunnableLambda](#comparison-of-runnablebranch-and-runnablelambda)


### References  
- [RunnableBranch API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.branch.RunnableBranch.html)  
- [RunnableLambda API Reference](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)  
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

[Note]
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
        "langchain",
        "langchain_openai",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.0[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
</pre>

You can alternatively set `OPENAI_API_KEY` in `.env` file and load it. 

[Note] This is not necessary if you've already set `OPENAI_API_KEY` in previous steps.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "04-Routing",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Load environment variables
# Reload any variables that need to be overwritten from the previous cell

from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## What is the RunnableBranch

`RunnableBranch` is a powerful tool that allows dynamic routing of logic based on input. It enables developers to flexibly define different processing paths depending on the characteristics of the input data.  

`RunnableBranch` helps implement complex decision trees in a simple and intuitive way. This greatly improves code readability and maintainability while promoting logic modularization and reusability.  

Additionally, `RunnableBranch` can dynamically evaluate branching conditions at runtime and select the appropriate processing routine, enhancing the system's adaptability and scalability.  

Due to these features, `RunnableBranch` can be applied across various domains and is particularly useful for developing applications with high input data variability and volatility.

By effectively utilizing `RunnableBranch`, developers can reduce code complexity and improve system flexibility and performance.

### Dynamic Logic Routing Based on Input

This section covers how to perform routing in LangChain Expression Language.

Routing allows you to create non-deterministic chains where the output of a previous step defines the next step. This helps bring structure and consistency to interactions with LLMs.

There are two primary methods for performing routing:

1. Returning a Conditionally Executable Object from `RunnableLambda` (*Recommended*)  
2. Using `RunnableBranch`

Both methods can be explained using a two-step sequence, where the first step classifies the input question as related to math, science, or other, and the second step routes it to the corresponding prompt chain.

### Simple Example

First, we will create a Chain that classifies incoming questions into one of three categories: math, science, or other.

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Classify the given user question into one of `math`, `science`, or `other`. Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
)

# Create the chain.
chain = (
    prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()  # Use a string output parser.
)
```

Use the created chain to classify the question.

```python
# Invoke the chain with a question.
chain.invoke({"question": "What is 2+2?"})
```




<pre class="custom">'math'</pre>



```python
# Invoke the chain with a question.
chain.invoke({"question": "What is the law of action and reaction?"})
```




<pre class="custom">'science'</pre>



```python
# Invoke the chain with a question.
chain.invoke({"question": "What is LangChain?"})
```




<pre class="custom">'other'</pre>



## RunnableLambda  

`RunnableLambda` is a type of `Runnable` designed to simplify the execution of a single transformation or operation using a lambda (anonymous) function. 

It is primarily used for lightweight, stateless operations where defining an entire custom `Runnable` class would be overkill.  

Unlike `RunnableBranch`, which focuses on conditional branching logic, `RunnableLambda` excels in straightforward data transformations or function applications.

Syntax  
- `RunnableLambda` is initialized with a single lambda function or callable object.  
- When invoked, the input value is passed directly to the lambda function.  
- The lambda function processes the input and returns the result.  

Now, let's create three sub-chains.

```python
math_chain = (
    PromptTemplate.from_template(
        """You are an expert in math. \
Always answer questions starting with "Pythagoras once said...". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)

science_chain = (
    PromptTemplate.from_template(
        """You are an expert in science. \
Always answer questions starting with "Isaac Newton once said...". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question concisely:

Question: {question}
Answer:"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
)
```

### Using Custom Functions  

This is the recommended approach in the official LangChain documentation. You can wrap custom functions with `RunnableLambda` to handle routing between different outputs.

```python
# Return each chain based on the contents included in the topic.


def route(info):
    if "math" in info["topic"].lower():
        return math_chain
    elif "science" in info["topic"].lower():
        return science_chain
    else:
        return general_chain
```

```python
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

full_chain = (
    {"topic": chain, "question": itemgetter("question")}
    | RunnableLambda(
        # Pass the routing function as an argument.
        route
    )
    | StrOutputParser()
)
```

```python
# Invoke the chain with a math-related question.
full_chain.invoke({"question": "Please explain the concept of calculus."})
```




<pre class="custom">'Pythagoras once said, "Mathematics is the language in which God has written the universe." In this spirit, calculus is a branch of mathematics that deals with the concepts of change and motion. It is divided into two main areas: differential calculus, which focuses on the rate of change of quantities (derivatives), and integral calculus, which concerns the accumulation of quantities (integrals). \n\nCalculus allows us to understand and describe the behavior of functions, enabling us to determine slopes of curves, areas under curves, and solve problems involving dynamic systems. Its principles are foundational in various fields, including physics, engineering, economics, and biology, making it a vital tool for analyzing and modeling real-world phenomena.'</pre>



```python
# Invoke the chain with a science-related question.
full_chain.invoke({"question": "How is gravitational acceleration calculated?"})
```




<pre class="custom">'Isaac Newton once said, "What goes up must come down," which reflects the essence of gravitational forces. Gravitational acceleration can be calculated using the formula \\( g = \\frac{F}{m} \\), where \\( g \\) is the acceleration due to gravity, \\( F \\) is the force acting on an object, and \\( m \\) is the mass of that object. Near the Earth\'s surface, this acceleration is approximately \\( 9.81 \\, \\text{m/s}^2 \\), and it can also be derived from Newton\'s law of universal gravitation, which states that \\( F = \\frac{G \\cdot m_1 \\cdot m_2}{r^2} \\), where \\( G \\) is the gravitational constant, \\( m_1 \\) and \\( m_2 \\) are the masses of the two objects, and \\( r \\) is the distance between their centers. By rearranging this equation, we can express gravitational acceleration as \\( g = \\frac{G \\cdot M}{r^2} \\), where \\( M \\) is the mass of the Earth and \\( r \\) is the distance from the center of the Earth to the object.'</pre>



```python
# Invoke the chain with a general question.
full_chain.invoke({"question": "What is RAG (Retrieval Augmented Generation)?"})
```




<pre class="custom">'RAG (Retrieval-Augmented Generation) is a machine learning approach that combines retrieval of relevant documents from a knowledge base with generative models to produce more informed and contextually accurate responses. It enhances the generation process by retrieving pertinent information, which can help improve the quality and relevance of the output in tasks like question answering and dialogue.'</pre>



## RunnableBranch

`RunnableBranch` is a special type of `Runnable` that allows you to define conditions and corresponding Runnable objects based on input values.

However, it does not provide functionality that cannot be achieved with custom functions, so using custom functions is generally recommended.

Syntax

- `RunnableBranch` is initialized with a list of (condition, Runnable) pairs and a default Runnable.
- When invoked, the input value is passed to each condition sequentially.
- The first condition that evaluates to True is selected, and the corresponding Runnable is executed with the input value.
- If no condition matches, the `default Runnable` is executed.

```python
from operator import itemgetter
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # Check if the topic contains "math" and execute math_chain if true.
    (lambda x: "math" in x["topic"].lower(), math_chain),
    # Check if the topic contains "science" and execute science_chain if true.
    (lambda x: "science" in x["topic"].lower(), science_chain),
    # If none of the above conditions match, execute general_chain.
    general_chain,
)

# Define the full chain that takes a topic and question, routes it, and parses the output.
full_chain = (
    {"topic": chain, "question": itemgetter("question")} | branch | StrOutputParser()
)
```

Execute the full chain with each question.

```python
full_chain.invoke({"question": "Please explain the concept of calculus."})
```




<pre class="custom">'Pythagoras once said that understanding the world requires a deep appreciation of its underlying patterns and changes. Calculus is a branch of mathematics that focuses on the study of change and motion, providing tools to analyze how quantities vary with one another. It is primarily divided into two main areas: differential calculus, which deals with the concept of the derivative and the rate of change, and integral calculus, which concerns the accumulation of quantities and the area under curves.\n\nBy exploring these concepts, calculus allows us to model and solve problems in physics, engineering, economics, and beyond, illustrating how we can understand and predict the behavior of dynamic systems in our universe.'</pre>



```python
full_chain.invoke({"question": "How is gravitational acceleration calculated?"})
```




<pre class="custom">'Isaac Newton once said, "What goes up must come down," which reflects the fundamental principle of gravity. Gravitational acceleration, denoted as \\( g \\), is calculated using the formula:\n\n\\[\ng = \\frac{G \\cdot M}{r^2}\n\\]\n\nwhere \\( G \\) is the universal gravitational constant (\\( 6.674 \\times 10^{-11} \\, \\text{m}^3\\text{kg}^{-1}\\text{s}^{-2} \\)), \\( M \\) is the mass of the object exerting the gravitational force (such as the Earth), and \\( r \\) is the distance from the center of the mass to the point where the acceleration is being measured. Near the surface of the Earth, \\( g \\) is approximately \\( 9.81 \\, \\text{m/s}^2 \\). This equation illustrates how gravitational acceleration is influenced by both mass and distance, embodying the laws of motion and universal gravitation that Newton famously formulated.'</pre>



```python
full_chain.invoke({"question": "What is RAG (Retrieval Augmented Generation)?"})
```




<pre class="custom">'Retrieval Augmented Generation (RAG) is a natural language processing technique that combines retrieval-based and generation-based approaches. It enhances a generative model by incorporating relevant information from a retrieval system, allowing the model to generate more accurate and contextually appropriate responses by accessing external knowledge sources during the generation process.'</pre>



## Comparison of RunnableBranch and RunnableLambda

| Criteria    | RunnableLambda                               | RunnableBranch                        |  
|------------------|--------------------------------------------------|-------------------------------------------|  
| Condition Definition | All conditions are defined within a single function (`route`) | Each condition is defined as a `(condition, runnable)` pair |  
| Readability | Very clear for simple logic                      | Becomes clearer as conditions increase    |  
| Maintainability | Can become complex if the function grows large  | Clear separation between conditions and runnables |  
| Flexibility | Allows more flexible condition writing           | Must follow the `(condition, runnable)` pattern |  
| Scalability | Expandable by modifying the function             | Requires adding new conditions and runnables |  
| Recommended Use Case | When conditions are relatively simple or function-based | When there are many conditions or maintainability is a priority |  
