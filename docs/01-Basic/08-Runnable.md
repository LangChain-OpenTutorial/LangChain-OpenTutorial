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

# Runnable

- Author: [hyeyeoon](https://github.com/hyeyeoon)
- Peer Review : [hong-seongmin](https://github.com/hong-seongmin), [Wooseok Jeong](https://github.com/jeong-wooseok)
- Proofread : [Q0211](https://github.com/Q0211)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/08-Runnable.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/08-Runnable.ipynb)

## Overview

LangChain's ```Runnable``` objects provide a modular and flexible approach to designing workflows by enabling the chaining, parallel execution, and transformation of data. These utilities allow for efficient handling of structured inputs and outputs, with minimal code overhead.

Key Components is:

- **```RunnableLambda```**: A lightweight utility that enables the application of custom logic through lambda functions, ideal for dynamic and quick data transformations.
- **```RunnablePassthrough```**: Designed to pass input data unchanged or augment it with additional attributes when paired with the ```.assign()``` method.
- **```itemgetter```**: A Python ```operator``` module utility for efficiently extracting specific keys or indices from structured data such as dictionaries or tuples.

These tools can be combined to build powerful workflows, such as:

- Extracting and processing specific data elements using ```itemgetter```.
- Performing custom transformations with ```RunnableLambda```.
- Creating end-to-end data pipelines with ```Runnable``` chains.

By leveraging these components, users can design scalable and reusable pipelines for machine learning and data processing workflows.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Efficient Data Handling with RunnablePassthrough](#efficient-data-handling-with-runnablepassthrough)
- [Efficient Parallel Execution with RunnableParallel](#efficient-parallel-execution-with-runnableparallel)
- [Dynamic Processing with RunnableLambda](#dynamic-processing-with-runnablelambda)
- [Extracting Specific Keys Using itemgetter](#extracting-specific-keys-using-itemgetter)

### References

- [LangChain Documentation: Runnable](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html)
- [LangChain Documentation](https://python.langchain.com/docs/how_to/lcel_cheatsheet/)
- [Python operator module: itemgetter](https://docs.python.org/3/library/operator.html#operator.itemgetter)

---

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
        "langchain_openai",
    ],
    verbose=False,
    upgrade=False,
)
```

You can also load the ```OPEN_API_KEY``` from the ```.env``` file.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# Set local environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "05-Runnable",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## Efficient Data Handling with RunnablePassthrough

```RunnablePassthrough``` is a utility designed to streamline data processing workflows by either passing input data unchanged or enhancing it with additional attributes. Its flexibility makes it a valuable tool for handling data in pipelines where minimal transformation or selective augmentation is required.

1. **Simple Data Forwarding**

- Suitable for scenarios where no transformation is required, such as logging raw data or passing it to downstream systems.

2. **Dynamic Data Augmentation**

- Enables the addition of metadata or context to input data for use in machine learning pipelines or analytics systems.

---
- ```RunnablePassthrough``` can either pass the input unchanged or append additional keys to it.
- When ```RunnablePassthrough()``` is called on its own, it simply takes the input and passes it as is.
- When called using ```RunnablePassthrough.assign(...)```, it takes the input and adds additional arguments provided to the assign function.

### RunnablePassthrough

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Create the prompt and llm
prompt = PromptTemplate.from_template("What is 10 times {num}?")
llm = ChatOpenAI(temperature=0)

# Create the chain
chain = prompt | llm
```

When invoking the chain with ```invoke()```, the input data must be of type ```dictionary```.

```python
# Execute the chain : input dtype as 'dictionary'
chain.invoke({"num": 5})
```




<pre class="custom">AIMessage(content='10 times 5 is equal to 50.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 15, 'total_tokens': 26, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-420bc9dc-12eb-4f7a-a2c4-8e521b3d952d-0', usage_metadata={'input_tokens': 15, 'output_tokens': 11, 'total_tokens': 26, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



However, with the update to the LangChain library, if the template includes **only one variable**, it is also possible to pass just the value directly.

```python
# Execute the chain : input value directly
chain.invoke(5)
```




<pre class="custom">AIMessage(content='10 times 5 is equal to 50.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 15, 'total_tokens': 26, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-3723d11b-89e1-490c-8946-b724fbc2c46d-0', usage_metadata={'input_tokens': 15, 'output_tokens': 11, 'total_tokens': 26, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



Here is an example using ```RunnablePassthrough```.
```RunnablePassthrough``` is a ```runnable``` object with the following characteristics:

1. **Basic Operation**
   - Performs a simple pass-through function that forwards input values directly to output
   - Can be executed independently using the ```invoke()``` method

2. **Use Cases**
   - Useful when you need to pass data through chain steps without modification
   - Can be combined with other components to build complex data pipelines
   - Particularly helpful when you need to preserve original input while adding new fields

3. **Input Handling**
   - Accepts dictionary-type inputs
   - Can handle single values as well
   - Maintains data structure throughout the chain

```python
from langchain_core.runnables import RunnablePassthrough

# Runnable
RunnablePassthrough().invoke({"num": 10})
```




<pre class="custom">{'num': 10}</pre>



Here is an example of creating a chain using ```RunnablePassthrough```.

```python
runnable_chain = {"num": RunnablePassthrough()} | prompt | ChatOpenAI()

# The dict value has been updated with RunnablePassthrough().
runnable_chain.invoke(10)
```




<pre class="custom">AIMessage(content='10 times 10 is equal to 100.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 15, 'total_tokens': 26, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-dffc0a69-0ee5-43b1-adae-03ee863d5a68-0', usage_metadata={'input_tokens': 15, 'output_tokens': 11, 'total_tokens': 26, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>



Here is a comparison of the results when using ```RunnablePassthrough.assign()```.

```python
RunnablePassthrough().invoke({"num": 1})
```




<pre class="custom">{'num': 1}</pre>



```RunnablePassthrough.assign()```
- Combines the key/value pairs from the input with the newly assigned key/value pairs.

```python
# Input key: num, Assigned key: new_num
(RunnablePassthrough.assign(new_num=lambda x: x["num"] * 3)).invoke({"num": 1})
```




<pre class="custom">{'num': 1, 'new_num': 3}</pre>



## Efficient Parallel Execution with RunnableParallel

```RunnableParallel``` is a utility designed to execute multiple ```Runnable``` objects concurrently, streamlining workflows that require parallel processing. It distributes input data across different components, collects their results, and combines them into a unified output. This functionality makes it a powerful tool for optimizing workflows where tasks can be performed independently and simultaneously.


1. **Concurrent Execution**
   - Executes multiple ```Runnable``` objects simultaneously, reducing the time required for tasks that can be parallelized.

2. **Unified Output Management**
   - Combines the results from all parallel executions into a single, cohesive output, simplifying downstream processing.

3. **Flexibility**
   - Can handle diverse input types and support complex workflows by distributing the workload efficiently.

```python
from langchain_core.runnables import RunnableParallel

# Create an instance of RunnableParallel. This instance allows multiple Runnable objects to be executed in parallel.
runnable = RunnableParallel(
    # Pass a RunnablePassthrough instance as the 'passed' keyword argument. This simply passes the input data through without modification.
    passed=RunnablePassthrough(),
    # Use RunnablePassthrough.assign as the 'extra' keyword argument to assign a lambda function 'mult'. 
    # This function multiplies the value associated with the 'num' key in the input dictionary by 3.
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    # Pass a lambda function as the 'modified' keyword argument. 
    # This function adds 1 to the value associated with the 'num' key in the input dictionary.
    modified=lambda x: x["num"] + 1,
)

# Call the invoke method on the runnable instance, passing a dictionary {'num': 1} as input.
runnable.invoke({"num": 1})
```




<pre class="custom">{'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}</pre>



Chains can also be applied to RunnableParallel.

```python
chain1 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("What is the capital of {country}?")
    | ChatOpenAI()
)
chain2 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("What is the area of {country}?")
    | ChatOpenAI()
)
```

```python
combined_chain = RunnableParallel(capital=chain1, area=chain2)
combined_chain.invoke("United States of America")
```




<pre class="custom">{'capital': AIMessage(content='The capital of the United States of America is Washington, D.C.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 17, 'total_tokens': 32, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-29437a26-8661-4f15-a655-3b3ca6aa0e8c-0', usage_metadata={'input_tokens': 17, 'output_tokens': 15, 'total_tokens': 32, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     'area': AIMessage(content='The total land area of the United States of America is approximately 3.8 million square miles.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 17, 'total_tokens': 38, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-5004e08c-dd66-4c7c-bc3f-60821fecc403-0', usage_metadata={'input_tokens': 17, 'output_tokens': 21, 'total_tokens': 38, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}</pre>



## Dynamic Processing with RunnableLambda

```RunnableLambda``` is a flexible utility that allows developers to define custom data transformation logic using lambda functions. By enabling quick and easy implementation of custom processing workflows, ```RunnableLambda``` simplifies the creation of tailored data pipelines while ensuring minimal setup overhead.

1. **Customizable Data Transformation**
   - Allows users to define custom logic for transforming input data using lambda functions, offering unparalleled flexibility.

2. **Lightweight and Simple**
   - Provides a straightforward way to implement ad-hoc processing without the need for extensive class or function definitions.


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime

def get_today(a):
    # Get today's date
    return datetime.today().strftime("%b-%d")

# Print today's date
get_today(None)
```




<pre class="custom">'Jan-04'</pre>



```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Create the prompt and llm
prompt = PromptTemplate.from_template(
    "List {n} famous people whose birthday is on {today}. Include their date of birth."
)
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create the chain
chain = (
    {"today": RunnableLambda(get_today), "n": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
# Output
print(chain.invoke(3))
```

<pre class="custom">Here are three famous people born on January 4:
    
    1. **Isaac Newton** - Born on January 4, 1643 (according to the Gregorian calendar; December 25, 1642, in the Julian calendar), he was an English mathematician, physicist, astronomer, and author who is widely recognized as one of the most influential scientists of all time.
    
    2. **Louis Braille** - Born on January 4, 1809, he was a French educator and inventor of a system of reading and writing for use by the blind or visually impaired, known as Braille.
    
    3. **Michael Stipe** - Born on January 4, 1960, he is an American singer-songwriter and the lead vocalist of the alternative rock band R.E.M.
</pre>

## Extracting Specific Keys Using ```itemgetter```

```itemgetter``` is a utility function from Python's ```operator``` module with the following features and benefits:

1. **Core Functionality**
   - Efficiently extracts values from specific keys or indices in dictionaries, tuples, and lists
   - Capable of extracting multiple keys or indices simultaneously
   - Supports functional programming style

2. **Performance Optimization**
   - More efficient than regular indexing for repetitive key access operations
   - Optimized memory usage
   - Performance advantages when processing large datasets

3. **Usage in LangChain**
   - Data filtering in chain compositions
   - Selective extraction from complex input structures
   - Combines with other Runnable objects for data preprocessing


```python
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


# Function that returns the length of a sentence.
def length_function(text):
    return len(text)


# Function that returns the product of the lengths of two sentences.
def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


# Function that uses _multiple_length_function to return the product of the lengths of two sentences.
def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("What is {a} + {b}?")
model = ChatOpenAI()

chain1 = prompt | model

chain = (
    {
        "a": itemgetter("word1") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("word1"), "text2": itemgetter("word2")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
)
```

```python
chain.invoke({"word1": "hello", "word2": "world"})
```




<pre class="custom">AIMessage(content='5 + 25 = 30', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 8, 'prompt_tokens': 15, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-1cb2a062-52ba-4042-a4c1-a1eef155f6cc-0', usage_metadata={'input_tokens': 15, 'output_tokens': 8, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})</pre>


