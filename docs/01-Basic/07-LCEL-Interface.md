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

# LCEL Interface

- Author: [JeongGi Park](https://github.com/jeongkpa)
- Peer Review: [YooKyung Jeon](https://github.com/sirena1), [Wooseok Jeong](https://github.com/jeong-wooseok)
- Proofread : [Q0211](https://github.com/Q0211)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/07-LCEL-Interface.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/01-Basic/07-LCEL-Interface.ipynb)

## Overview

The LangChain Expression Language (LCEL) is a powerful interface designed to simplify the creation and management of custom chains in LangChain. 
It implements the Runnable protocol, providing a standardized way to build and execute language model chains.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [LCEL Runnable Protocol](#lcel-runnable-protocol)
- [stream: real-time output](#stream-real-time-output)
- [Invoke](#invoke)
- [batch: unit execution](#batch-unit-execution)
- [async stream](#async-stream)
- [async invoke](#async-invoke)
- [async batch](#async-batch)
- [Parallel](#parallel)
- [Parallelism in batches](#parallelism-in-batches)

### References

- [Langsmith DOC](https://docs.smith.langchain.com/)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

set environment variables is in .env.

Copy the contents of .env_sample and load it into your .env with the key you set.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
%%capture --no-stderr
!pip install langchain-opentutorial
```

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.0.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain-openai",
        "langchain",
        "python-dotenv",
        "langchain-core",
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
        "OPENAI_API_KEY": "<Your OpenAI API KEY>",
        "LANGCHAIN_API_KEY": "<Your LangChain API KEY>",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "LangSmith-Tracking-Setup",  # title 과 동일하게 설정해 주세요
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

## LCEL Runnable Protocol

---

To make it as easy as possible to create custom chains, we've implemented the ```Runnable``` protocol.

The ```Runnable``` protocol is implemented in most components.

It is a standard interface that makes it easy to define custom chains and call them in a standard way. The standard interface includes

- ```stream```: Streams a chunk of the response.
- ```invoke```: Invoke a chain on an input.
- ```batch```: Invoke a chain against a list of inputs.

There are also asynchronous methods

- ```astream```: Stream chunks of the response asynchronously.
- ```ainvoke```: Invoke a chain asynchronously on an input.
- ```abatch```: Asynchronously invoke a chain against a list of inputs.
- ```astream_log```: Streams the final response as well as intermediate steps as they occur.



### Log your trace

We provide multiple ways to log traces to LangSmith. Below, we'll highlight how to use traceable().

Use the code below to record a trace in LangSmith

```python
import openai
from langsmith import wrappers, traceable

# Auto-trace LLM calls in-context
client = wrappers.wrap_openai(openai.Client())

@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-4o-mini"
    )
    return result.choices[0].message.content

pipeline("Hello, world!")
# Out:  Hello there! How can I assist you today?
```




<pre class="custom">'Hello! How can I assist you today?'</pre>



Create a chain using LCEL syntax.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Instantiate the ChatOpenAI model.
model = ChatOpenAI()
# Create a prompt template that asks for jokes on a given topic.
prompt = PromptTemplate.from_template("Describe the {topic} in 3 sentences.")
# Connect the prompt and model to create a conversation chain.
chain = prompt | model | StrOutputParser()
```

## stream: real-time output

This function uses the ```chain.stream``` method to create a stream of data for a given topic, iterating over it and immediately outputting the ```content``` of each piece of data. 
The ```end=""``` argument disables newlines after output, and the ```flush=True``` argument causes the output buffer to be emptied immediately.

```python
# Use the chain.stream method to create a stream of data for a given topic, iterating over it and immediately outputting the content of each piece of data. 
for token in chain.stream({"topic": "multimodal"}):
    # Output the content of each piece of data without newlines.
    print(token, end="", flush=True)

# example output 
# The multimodal approach involves using multiple modes of communication, such as visual, auditory, and kinesthetic, to enhance learning and understanding. By incorporating different sensory inputs, learners are able to engage with material in a more holistic and immersive way. This approach is especially effective in catering to diverse learning styles and preferences.
```

<pre class="custom">The multimodal approach involves using multiple modes of communication, such as visual, auditory, and kinesthetic, to enhance learning and understanding. By incorporating various modalities, individuals can access information in ways that cater to their unique learning styles and preferences. This approach allows for a more comprehensive and inclusive learning experience that promotes deeper engagement and retention of information.</pre>

## Invoke

The ```invoke``` method of a ```chain``` object takes a topic as an argument and performs processing on that topic.






```python
# Call the invoke method of the chain object, passing a dictionary with the topic 'ChatGPT'.
chain.invoke({"topic": "ChatGPT"})
```




<pre class="custom">'ChatGPT is a state-of-the-art conversational AI model developed by OpenAI that can engage in natural and coherent conversations with users. It uses the GPT-3 architecture, which allows it to generate human-like responses based on the input it receives. ChatGPT is known for its ability to understand context, provide relevant information, and maintain engaging conversations across a wide range of topics.'</pre>



## batch: unit execution

The function ```chain.batch``` takes a list containing multiple dictionaries as arguments and performs batch processing using the values of the ```topic``` key in each dictionary.

```python
# Call a function to batch process a given list of topics
chain.batch([{"topic": "ChatGPT"}, {"topic": "Instagram"}])
```




<pre class="custom">['ChatGPT is an advanced conversational AI model developed by OpenAI that can engage in natural and engaging conversations with users. It is trained on a diverse range of internet text data to understand and generate human-like responses. ChatGPT is capable of providing information, answering questions, and engaging in meaningful dialogues with users on a variety of topics.',
     'Instagram is a social media platform where users can share photos and videos with their followers. The app allows users to edit and enhance their photos with filters and captions before posting. Users can also interact with others by liking, commenting, and messaging on their posts.']</pre>



You can use the ```max_concurrency``` parameter to set the number of concurrent requests.

The ```config``` dictionary uses the ```max_concurrency``` key to set the maximum number of operations that can be processed concurrently. Here, it is set to process up to three jobs concurrently.

```python
chain.batch(
    [
        {"topic": "ChatGPT"},
        {"topic": "Instagram"},
        {"topic": "multimodal"},
        {"topic": "programming"},
        {"topic": "machineLearning"},
    ],
    config={"max_concurrency": 3},
)
```




<pre class="custom">['ChatGPT is an AI-powered chatbot developed by OpenAI that can engage in text-based conversations with users. It uses a machine learning model trained on a diverse range of internet text to generate responses that are contextually relevant and engaging. ChatGPT can provide information, answer questions, or just have a casual conversation with users.',
     'Instagram is a social media platform where users can share photos and videos with their followers. It allows users to edit and enhance their photos with filters and editing tools. Users can also engage with others by liking, commenting, and sharing posts.',
     'The multimodal approach involves using multiple modes of communication, such as visual, auditory, and kinesthetic, to enhance learning and understanding. This approach recognizes that individuals learn in different ways and can benefit from a variety of sensory experiences. By incorporating different modes of communication, educators can cater to diverse learning styles and create a more engaging and effective learning environment.',
     'Programming involves writing instructions in a specific programming language to create software applications, websites, or other types of technology. It requires problem-solving skills, logic, and attention to detail to write code that functions correctly. Programmers use various tools and resources to develop, test, and debug their code to ensure it performs as intended.',
     'Machine learning is a branch of artificial intelligence that involves developing algorithms and models that allow computers to learn and improve from data without being explicitly programmed. It involves training a model on a dataset to make predictions or decisions based on new, unseen data. Machine learning algorithms can be used for a variety of tasks, such as image recognition, natural language processing, and recommendation systems.']</pre>



## async stream

The function ```chain.astream``` creates an asynchronous stream and processes messages for a given topic asynchronously.

It uses an asynchronous for loop (```async for```) to sequentially receive messages from the stream, and the print function to immediately print the contents of the messages (```s.content```). ```end=""``` disables line wrapping after printing, and ```flush=True``` forces the output buffer to be emptied to ensure immediate printing.


```python
# Use an asynchronous stream to process messages in the 'YouTube' topic.
async for token in chain.astream({"topic": "YouTube"}):
    # Print the message content. Outputs directly without newlines and empties the buffer.
    print(token, end="", flush=True)
```

<pre class="custom">YouTube is a video-sharing platform where users can upload, view, and interact with videos on a wide range of topics. It has become one of the largest and most popular websites on the internet, with millions of users visiting daily. Users can subscribe to channels, like and comment on videos, and create playlists to save their favorite content.</pre>

## async invoke

The ```ainvoke``` method of a ```chain``` object performs an operation asynchronously with the given arguments. Here, we are passing a dictionary with a key named ```topic``` and a value named ```NVDA``` (NVIDIA's ticker) as arguments. This method can be used to asynchronously request processing for a specific topic.

```python
# Handle the 'NVDA' topic by calling the 'ainvoke' method of the asynchronous chain object.
my_process = chain.ainvoke({"topic": "NVDA"})
```

```python
# Wait for the asynchronous process to complete.
await my_process
```




<pre class="custom">'The National Visa Center (NVDA) is a government agency responsible for processing immigrant visa applications for individuals seeking to live permanently in the United States. They review and approve the necessary documentation, schedule interviews at U.S. embassies or consulates, and ensure that applicants meet all eligibility requirements for a visa. The NVDA plays a crucial role in managing the visa application process and facilitating legal immigration to the United States.'</pre>



## async batch

The function ```abatch``` batches a series of actions asynchronously.

In this example, we are using the ```abatch``` method of the ```chain``` object to asynchronously process actions on ```topic``` .

The ```await``` keyword is used to wait for those asynchronous tasks to complete.

```python
# Performs asynchronous batch processing on a given topic.
my_abatch_process = chain.abatch(
    [{"topic": "YouTube"}, {"topic": "Instagram"}, {"topic": "Facebook"}]
)
```

```python
# Wait for the asynchronous batch process to complete.
await my_abatch_process
```




<pre class="custom">['YouTube is a popular video-sharing platform where users can upload, view, and share videos on a wide range of topics. It has become a major source of entertainment, education, and information for millions of people around the world. With features like live streaming, monetization options, and a vast library of content, YouTube has revolutionized the way we consume video content.',
     'Instagram is a popular social media platform where users can share photos and videos with their followers. It allows for creative expression through filters, captions, and hashtags. Users can also interact with others by liking, commenting, and messaging on posts.',
     'Facebook is a popular social media platform where users can connect with friends and family, share photos and updates, and join groups based on their interests. It was founded by Mark Zuckerberg in 2004 and has since grown to have billions of active users worldwide. Users can also follow pages and businesses to stay updated on news and events.']</pre>



## Parallel

Let's take a look at how the LangChain Expression Language supports parallel requests. For example, when you use ```RunnableParallel``` (often written in dictionary form), you execute each element in parallel.

Here's an example of running two tasks in parallel using the ```RunnableParallel``` class in the ```langchain_core.runnables``` module.

Create two chains (```chain1```, ```chain2```) that use the ```ChatPromptTemplate.from_template``` method to get the capital and area for a given ```country```.

These chains are connected via the ```model``` and pipe (```|```) operators, respectively. Finally, we use the ```RunnableParallel``` class to combine these two chains with the keys ```capital``` and ```area``` to create a ```combined``` object that can be run in parallel.

```python
from langchain_core.runnables import RunnableParallel

# Create a chain that asks for the capital of {country}.
chain1 = (
    PromptTemplate.from_template("What is the capital of {country}?")
    | model
    | StrOutputParser()
)

# Create a chain that asks for the area of {country}.
chain2 = (
    PromptTemplate.from_template("What is the area of {country}?")
    | model
    | StrOutputParser()
)

# Create a parallel execution chain that generates the above two chains in parallel.
combined = RunnableParallel(capital=chain1, area=chain2)
```

The ```chain1.invoke()``` function calls the ```invoke``` method of the ```chain1``` object.

As an argument, it passes a dictionary with the value ```Canada``` in the key named ```country```.

```python
# Run chain1 .
chain1.invoke({"country": "Canada"})
```




<pre class="custom">'Ottawa'</pre>



Call ```chain2.invoke()```, this time passing a different country, the ```United States```, for the country key.

```python
# Run chain2 .
chain2.invoke({"country": "USA"})
```




<pre class="custom">'The total land area of the United States is approximately 3.8 million square miles (9.8 million square kilometers).'</pre>



The ```invoke``` method of the ```combined``` object performs the processing for the given ```country```.

In this example, the topic ```USA``` is passed to the ```invoke``` method to run.

```python
# Run a parallel execution chain.
combined.invoke({"country": "USA"})
```




<pre class="custom">{'capital': 'The capital of the United States of America is Washington, D.C.',
     'area': 'The total area of the United States is approximately 9.8 million square kilometers (3.8 million square miles).'}</pre>



## Parallelism in batches

Parallelism can be combined with other executable code. Let's try using parallelism with batch.

The ```chain1.batch``` function takes a list containing multiple dictionaries as an argument, and processes the values corresponding to the "topic" key in each dictionary. In this example, we're batch processing two topics, "Canada" and "United States".

```python
# Perform batch processing.
chain1.batch([{"country": "Canada"}, {"country": "USA"}])

```




<pre class="custom">['Ottawa', 'The capital of the United States of America is Washington, D.C.']</pre>



The ```chain2.batch``` function takes in multiple dictionaries as a list and performs batch processing.

In this example, we request processing for two countries, ```Canada``` and the ```United States```.

```python
# Perform batch processing.
chain2.batch([{"country": "Canada"}, {"country": "USA"}])

```




<pre class="custom">['The total land area of Canada is approximately 9.98 million square kilometers (3.85 million square miles), making it the second largest country in the world by land area.',
     'The total land area of the United States is approximately 3.8 million square miles.']</pre>



The combined.batch function is used to process the given data in batches. 

In this example, it takes a list containing two dictionary objects as arguments and batches data for two countries, Canada and the United States, respectively.

```python
# Processes the given data in batches.
combined.batch([{"country": "Canada"}, {"country": "USA"}])

```




<pre class="custom">[{'capital': 'Ottawa',
      'area': 'The total area of Canada is approximately 9.98 million square kilometers (3.85 million square miles), making it the second-largest country in the world by land area.'},
     {'capital': 'The capital of the United States of America is Washington, D.C.',
      'area': 'The total land area of the United States is approximately 3.8 million square miles (9.8 million square kilometers).'}]</pre>


