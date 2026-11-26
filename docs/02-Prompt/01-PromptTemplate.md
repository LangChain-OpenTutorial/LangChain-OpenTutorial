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

# Prompt Template

- Author: [Hye-yoon Jeong](https://github.com/Hye-yoonJeong)
- Peer Review : [hyeyeoon](https://github.com/hyeyeoon), [Wooseok Jeong](https://github.com/jeong-wooseok)
- Proofread : [Q0211](https://github.com/Q0211)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/01-PromptTemplate.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/02-Prompt/01-PromptTemplate.ipynb)
## Overview
This tutorial covers how to create and utilize prompt templates using ```LangChain```.

Prompt templates are essential for generating dynamic and flexible prompts that cater to various use cases, such as conversation history, structured outputs, and specialized queries.

In this tutorial, we will explore methods for creating ```PromptTemplate``` objects, applying partial variables, managing templates through YAML files, and leveraging advanced tools like ```ChatPromptTemplate``` and ```MessagePlaceholder``` for enhanced functionality.

### Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Creating a PromptTemplate Object](#creating-a-prompttemplate-object)
- [Using partial_variables](#using-partial_variables)
- [Load Prompt Templates from YAML Files](#load-prompt-templates-from-yaml-files)
- [ChatPromptTemplate](#chatprompttemplate)
- [MessagePlaceholder](#messageplaceholder)

### References
- [LangChain Documentation : Prompts](https://python.langchain.com/api_reference/core/prompts.html#)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can check out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

<pre class="custom">
    [notice] A new release of pip is available: 24.1 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
</pre>

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
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        # "OPENAI_API_KEY": "",
        # "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Prompt-Template",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

Let's setup ```ChatOpenAI``` with ```gpt-4o``` model.

```python
from langchain_openai import ChatOpenAI

# Load the model
llm = ChatOpenAI(model_name="gpt-4o")
```

## Creating a ```PromptTemplate``` Object

There are two ways to create a ```PromptTemplate``` object.
- 1. Using the ```from_template()``` method
- 2. Creating a ```PromptTemplate``` object and a prompt all at once

### Method 1. Using the ```from_template()``` method

- Define template with variable as ```{variable}``` .

```python
from langchain_core.prompts import PromptTemplate

# Define template. In this case, {country} is a variable
template = "What is the capital of {country}?"

# Create a `PromptTemplate` object using the `from_template` method
prompt = PromptTemplate.from_template(template)
prompt
```




<pre class="custom">PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?')</pre>



You can complete the prompt by assigning a value to the variable ```country``` .

```python
# Create prompt. Assign value to the variable using `format` method
prompt = prompt.format(country="United States of America")
prompt
```




<pre class="custom">'What is the capital of United States of America?'</pre>



```python
# Define template
template = "What is the capital of {country}?"

# Create a `PromptTemplate` object using the `from_template` method
prompt = PromptTemplate.from_template(template)

# Create chain
chain = prompt | llm
```

```python
# Replace the country variable with a value of your choice
chain.invoke("United States of America").content
```




<pre class="custom">'The capital of the United States of America is Washington, D.C.'</pre>



### Method 2. Creating a ```PromptTemplate``` object and a prompt all at once

Explicitly specify ```input_variables``` for additional validation.

Otherwise, a mismatch between such variables and the variables within the template string can raise an exception in instantiation.

```python
# Define template
template = "What is the capital of {country}?"

# Create a prompt template with `PromptTemplate` object
prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
)
prompt
```




<pre class="custom">PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?')</pre>



```python
# Create prompt
prompt.format(country="United States of America")
```




<pre class="custom">'What is the capital of United States of America?'</pre>



```python
# Define template
template = "What are the capitals of {country1} and {country2}, respectively?"

# Create a prompt template with `PromptTemplate` object
prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={
        "country2": "United States of America"  # Pass `partial_variables` in dictionary form
    },
)
prompt
```




<pre class="custom">PromptTemplate(input_variables=['country1'], input_types={}, partial_variables={'country2': 'United States of America'}, template='What are the capitals of {country1} and {country2}, respectively?')</pre>



```python
prompt.format(country1="South Korea")
```




<pre class="custom">'What are the capitals of South Korea and United States of America, respectively?'</pre>



```python
prompt_partial = prompt.partial(country2="India")
prompt_partial
```




<pre class="custom">PromptTemplate(input_variables=['country1'], input_types={}, partial_variables={'country2': 'India'}, template='What are the capitals of {country1} and {country2}, respectively?')</pre>



```python
prompt_partial.format(country1="South Korea")
```




<pre class="custom">'What are the capitals of South Korea and India, respectively?'</pre>



```python
chain = prompt_partial | llm
```

```python
chain.invoke("United States of America").content
```




<pre class="custom">'The capital of the United States of America is Washington, D.C., and the capital of India is New Delhi.'</pre>



```python
chain.invoke({"country1": "United States of America", "country2": "India"}).content
```




<pre class="custom">'The capital of the United States of America is Washington, D.C., and the capital of India is New Delhi.'</pre>



## Using ```partial_variables```

Using ```partial_variables``` , you can partially apply functions.  This is particularly useful when there are **common variables** to be shared.

Common examples are **date or time**.

Suppose you want to specify the current date in your prompt, hardcoding the date into the prompt or passing it along with other input variables may not be practical. In this case, using a function that returns the current date to modify the prompt partially is much more convenient.

```python
from datetime import datetime

# Print the current date
datetime.now().strftime("%B %d")
```




<pre class="custom">'January 14'</pre>



```python
# Define function to return the current date
def get_today():
    return datetime.now().strftime("%B %d")
```

```python
prompt = PromptTemplate(
    template="Today's date is {today}. Please list {n} celebrities whose birthday is today. Please specify their date of birth.",
    input_variables=["n"],
    partial_variables={
        "today": get_today  # Pass `partial_variables` in dictionary form
    },
)
```

```python
# Create prompt
prompt.format(n=3)
```




<pre class="custom">"Today's date is January 14. Please list 3 celebrities whose birthday is today. Please specify their date of birth."</pre>



```python
# Create chain
chain = prompt | llm
```

```python
# Invoke chain and check the result
print(chain.invoke(3).content)
```

<pre class="custom">Here are three celebrities born on January 14:
    
    1. **Dave Grohl** - Born on January 14, 1969.
    2. **LL Cool J** - Born on January 14, 1968.
    3. **Jason Bateman** - Born on January 14, 1969.
</pre>

```python
# Invoke chain and check the result
print(chain.invoke({"today": "Jan 02", "n": 3}).content)
```

<pre class="custom">Here are three celebrities born on January 2:
    
    1. **Cuba Gooding Jr.** - Born on January 2, 1968.
    2. **Taye Diggs** - Born on January 2, 1971.
    3. **Kate Bosworth** - Born on January 2, 1983.
</pre>

## Load Prompt Templates from YAML Files

You can manage prompt templates in seperate yaml files and load using ```load_prompt``` .

```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("prompts/fruit_color.yaml", encoding="utf-8")
prompt
```




<pre class="custom">PromptTemplate(input_variables=['fruit'], input_types={}, partial_variables={}, template='What is the color of {fruit}?')</pre>



```python
prompt.format(fruit="an apple")
```




<pre class="custom">'What is the color of an apple?'</pre>



```python
prompt2 = load_prompt("prompts/capital.yaml")
print(prompt2.format(country="United States of America"))
```

<pre class="custom">Please provide information about the capital city of United States of America.
    Summarize the characteristics of the capital in the following format, within 300 words.
    ----
    [Format]
    1. Area
    2. Population
    3. Historical Sites
    4. Regional Products
    
    #Answer:
    
</pre>

## ```ChatPromptTemplate```

```ChatPromptTemplate``` can be used to include a conversation history as a prompt.

Messages are structured as tuples in the format (```role``` , ```message``` ) and are created as a list.

**role**
- ```system``` : A system setup message, typically used for global settings-related prompts.
- ```human``` : A user input message.
- ```ai``` : An AI response message.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
chat_prompt
```




<pre class="custom">ChatPromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['country'], input_types={}, partial_variables={}, template='What is the capital of {country}?'), additional_kwargs={})])</pre>



```python
chat_prompt.format(country="United States of America")
```




<pre class="custom">'Human: What is the capital of United States of America?'</pre>



```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "You are a friendly AI assistant. Your name is {name}."),
        ("human", "Nice to meet you!"),
        ("ai", "Hello! How can I assist you?"),
        ("human", "{user_input}"),
    ]
)

# Create chat messages
messages = chat_template.format_messages(name="Teddy", user_input="What is your name?")
messages
```




<pre class="custom">[SystemMessage(content='You are a friendly AI assistant. Your name is Teddy.', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='Nice to meet you!', additional_kwargs={}, response_metadata={}),
     AIMessage(content='Hello! How can I assist you?', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='What is your name?', additional_kwargs={}, response_metadata={})]</pre>



You can directly invoke LLM using the messages created above.

```python
llm.invoke(messages).content
```




<pre class="custom">'My name is Teddy. How can I help you today?'</pre>



You can also create a chain to execute.

```python
chain = chat_template | llm
```

```python
chain.invoke({"name": "Teddy", "user_input": "What is your name?"}).content
```




<pre class="custom">'My name is Teddy. How can I help you today?'</pre>



## ```MessagePlaceholder```

```LangChain``` also provides a ```MessagePlaceholder``` , which provides complete control over rendering messages during formatting.

This can be useful if you’re unsure which roles to use in a message prompt template or if you want to insert a list of messages during formatting.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "Summarize the conversation so far in {word_count} words."),
    ]
)
chat_prompt
```




<pre class="custom">ChatPromptTemplate(input_variables=['conversation', 'word_count'], input_types={'conversation': list[typing.Annotated[typing.Union[typing.Annotated[langchain_core.messages.ai.AIMessage, Tag(tag='ai')], typing.Annotated[langchain_core.messages.human.HumanMessage, Tag(tag='human')], typing.Annotated[langchain_core.messages.chat.ChatMessage, Tag(tag='chat')], typing.Annotated[langchain_core.messages.system.SystemMessage, Tag(tag='system')], typing.Annotated[langchain_core.messages.function.FunctionMessage, Tag(tag='function')], typing.Annotated[langchain_core.messages.tool.ToolMessage, Tag(tag='tool')], typing.Annotated[langchain_core.messages.ai.AIMessageChunk, Tag(tag='AIMessageChunk')], typing.Annotated[langchain_core.messages.human.HumanMessageChunk, Tag(tag='HumanMessageChunk')], typing.Annotated[langchain_core.messages.chat.ChatMessageChunk, Tag(tag='ChatMessageChunk')], typing.Annotated[langchain_core.messages.system.SystemMessageChunk, Tag(tag='SystemMessageChunk')], typing.Annotated[langchain_core.messages.function.FunctionMessageChunk, Tag(tag='FunctionMessageChunk')], typing.Annotated[langchain_core.messages.tool.ToolMessageChunk, Tag(tag='ToolMessageChunk')]], FieldInfo(annotation=NoneType, required=True, discriminator=Discriminator(discriminator=<function _get_type at 0x0000020B148D7E20>, custom_error_type=None, custom_error_message=None, custom_error_context=None))]]}, partial_variables={}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.'), additional_kwargs={}), MessagesPlaceholder(variable_name='conversation'), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['word_count'], input_types={}, partial_variables={}, template='Summarize the conversation so far in {word_count} words.'), additional_kwargs={})])</pre>



You can use ```MessagesPlaceholder``` to add the conversation message list.

```python
formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "Hello! I’m Teddy. Nice to meet you."),
        ("ai", "Nice to meet you! I look forward to working with you."),
    ],
)

print(formatted_chat_prompt)
```

<pre class="custom">System: You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.
    Human: Hello! I’m Teddy. Nice to meet you.
    AI: Nice to meet you! I look forward to working with you.
    Human: Summarize the conversation so far in 5 words.
</pre>

```python
# Create chain
chain = chat_prompt | llm | StrOutputParser()
```

```python
# Invoke chain and check the result
chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            (
                "human",
                "Hello! I'm Teddy. Nice to meet you.",
            ),
            ("ai", "Nice to meet you! I look forward to working with you."),
        ],
    }
)
```




<pre class="custom">'Teddy introduces himself, exchanges greetings.'</pre>


