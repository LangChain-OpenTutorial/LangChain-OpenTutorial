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

This tutorial introduces `RunnableBranch` and `RunnableLambda`, two key tools in LangChain for implementing dynamic workflows and conditional logic.

`RunnableBranch` enables structured decision-making by routing input through predefined conditions, making complex branching scenarios easier to manage.

`RunnableLambda` offers a flexible, function-based approach, ideal for performing lightweight transformations and inline processing.

Through detailed explanations, practical examples, and comparisons, you'll gain a clear understanding of when and how to use each tool effectively.

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

Alternatively, you can set and load `OPENAI_API_KEY` from a `.env` file.

**[Note]** This is only necessary if you haven't already set `OPENAI_API_KEY` in previous steps.

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



## What is the ```RunnableBranch```

```RunnableBranch``` dynamically routes logic based on input. It allows developers to define different processing paths depending on the characteristics of the input data.  

```RunnableBranch``` simplifies the implementation of complex decision trees in a simple and more intuitive way. This improves code readability and maintainability while promoting modularization and reusability of logic.  

Additionally, ```RunnableBranch``` dynamically evaluates branching conditions at runtime. This enables it to select the appropriate processing routine, which enhances the system's adaptability and scalability.  

Thanks to these features, ```RunnableBranch``` is applicable across various domains and is particularly useful for developing applications that handle highly variable and volatile input data.

By effectively utilizing ```RunnableBranch```, developers can reduce code complexity while improving both system flexibility and performance.

### Dynamic Logic Routing Based on Input

This section covers how to perform routing within LangChain Expression Language (LCEL).

Routing enables the creation of non-deterministic chains, where the output of a previous step determines the next step. This brings core structure and consistency to interactions with LLMs.

There are two primary methods available for implementing routing:

1. Returning a conditionally executable object from ```RunnableLambda``` (*Recommended*).
2. Using ```RunnableBranch```.

Both of these methods can be explained using a two-step sequence: first, classifying the input question into a category (math, science, or other), and second, routing the question to the corresponding prompt chain based on the category.

### Simple Example

Firstly, we will create a chain that classifies incoming questions into one of three categories: math, science, or other.

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

After creating the chain, use it to classify a test question and verify the result.

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



## ```RunnableLambda```  

```RunnableLambda``` is a type of runnable designed to simplify the execution of a single transformation or operation using a lambda (anonymous) function. 

It is primarily used for lightweight, stateless operations where defining an entire custom Runnable class would be overkill.  

Unlike ```RunnableBranch```, which focuses on conditional branching logic, ```RunnableLambda``` excels in straightforward data transformations or function applications.

Syntax  
- ```RunnableLambda``` is initialized with a single lambda function or callable object.  
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




<pre class="custom">'Pythagoras once said, "The study of mathematics is the study of the universe." Calculus, much like the harmony found in geometric shapes, is a branch of mathematics that focuses on change and motion. It is fundamentally divided into two main concepts: differentiation and integration.\n\nDifferentiation deals with the idea of rates of change, allowing us to understand how a function behaves as its input changes. It helps us determine slopes of curves at given points, providing insight into how quantities vary.\n\nIntegration, on the other hand, is concerned with the accumulation of quantities, such as areas under curves. It allows us to sum up infinitely small pieces to find total quantities, providing a way to calculate things like distances traveled over time.\n\nTogether, these concepts enable us to analyze complex systems in fields ranging from physics to economics, illustrating how the world evolves and changes. In essence, calculus is a powerful tool that helps us grasp the continuous nature of change in our universe.'</pre>



```python
# Invoke the chain with a science-related question.
full_chain.invoke({"question": "How is gravitational acceleration calculated?"})
```




<pre class="custom">'Isaac Newton once said, "What goes up must come down," highlighting the fundamental principle of gravity. Gravitational acceleration is calculated using the formula \\( g = \\frac{F}{m} \\), where \\( F \\) is the force of gravity acting on an object and \\( m \\) is the mass of that object. In a more specific context, near the surface of the Earth, gravitational acceleration can also be approximated using the formula \\( g = \\frac{G \\cdot M}{r^2} \\), where \\( G \\) is the gravitational constant, \\( M \\) is the mass of the Earth, and \\( r \\) is the distance from the center of the Earth to the object. This results in a standard gravitational acceleration of approximately \\( 9.81 \\, \\text{m/s}^2 \\).'</pre>



```python
# Invoke the chain with a general question.
full_chain.invoke({"question": "What is RAG (Retrieval Augmented Generation)?"})
```




<pre class="custom">'Retrieval Augmented Generation (RAG) is a machine learning approach that combines retrieval-based methods with generative models. It retrieves relevant information from a knowledge base or document corpus to enhance the context for generating responses, enabling the model to produce more accurate and informative outputs by leveraging external data.'</pre>



## ```RunnableBranch```

```RunnableBranch``` is a specialized Runnable designed for defining conditions and the corresponding Runnable objects based on input values.

However, it does not provide any functionality achievable with custom functions. So, using custom functions is often preferred.

**Syntax**

- ```RunnableBranch``` is initialized with a list of **(condition, Runnable)** pairs and a default Runnable.
- When ```RunnableBranch``` is invoked, the input value is sequentially passed to each condition.
- The first condition that evaluates to True determins which Runnable is executed with the input.
- If none of conditions evaluate to True, the **default Runnable** is executed.

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

Let's execute the full chain with each question.

```python
full_chain.invoke({"question": "Please explain the concept of calculus."})
```




<pre class="custom">'Pythagoras once said, "To understand the world, we must first understand the relationships between its parts." Calculus is a branch of mathematics that focuses on change and motion, allowing us to analyze how quantities vary. It is fundamentally divided into two main areas: differential calculus, which deals with the concept of the derivative and how functions change at any given point, and integral calculus, which concerns the accumulation of quantities and the area under curves.\n\nThrough the tools of limits, derivatives, and integrals, calculus provides powerful methods for solving problems in physics, engineering, economics, and many other fields. It helps us understand everything from the motion of planets to the growth of populations, emphasizing the continuous nature of change in our universe.'</pre>



```python
full_chain.invoke({"question": "How is gravitational acceleration calculated?"})
```




<pre class="custom">'Isaac Newton once said, "What goes up must come down," reflecting his profound understanding of gravity. Gravitational acceleration, often denoted as \\( g \\), is calculated using the formula:\n\n\\[\ng = \\frac{G \\cdot M}{r^2}\n\\]\n\nwhere \\( G \\) is the gravitational constant (approximately \\( 6.674 \\times 10^{-11} \\, \\text{m}^3 \\text{kg}^{-1} \\text{s}^{-2} \\)), \\( M \\) is the mass of the object exerting the gravitational force (like the Earth), and \\( r \\) is the distance from the center of that mass to the point where the gravitational acceleration is being calculated. Near the Earth\'s surface, this value is approximately \\( 9.81 \\, \\text{m/s}^2 \\).'</pre>



```python
full_chain.invoke({"question": "What is RAG (Retrieval Augmented Generation)?"})
```




<pre class="custom">'Retrieval Augmented Generation (RAG) is a framework that combines retrieval-based and generation-based approaches in natural language processing. It retrieves relevant documents or information from a knowledge base and uses that information to enhance the generation of responses or text, improving the accuracy and relevance of the output. RAG is particularly useful in tasks like question answering and conversational agents.'</pre>



## Comparison of ```RunnableBranch``` and ```RunnableLambda```

| Criteria    | ```RunnableLambda```                               | ```RunnableBranch```                        |  
|------------------|--------------------------------------------------|-------------------------------------------|  
| Condition Definition | All conditions are defined within a single function (`route`). | Each condition is defined as a **(condition, Runnable)** pair. |  
| Readability | Very clear for simple logic.                      | Becomes clearer as the number of conditions increases.    |  
| Maintainability | Can become complex to maintain if the function grows large.  | Provides a clear separation between conditions and their corresponding Runnables. |  
| Flexibility | Allows more flexibility in how conditions are written.           | Requires adherence to the **(condition, Runnable)** pattern. |  
| Scalability | Involves modifying the existing function.             | Requires adding new **(condition, Runnable)** pairs. |  
| Recommended Use Case | When conditions are relatively simple or primarily function-based transformations. | When dealing with many conditions or when maintainability is a primary concern. |  
