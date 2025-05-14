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

# PydanticOutputParser

- Author: [Jaeho Kim](https://github.com/Jae-hoya)
- Peer Review: [Donghak Lee](https://github.com/stsr1284), [frimer](https://github.com/brian604)
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/01-PydanticOutputParser.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/01-PydanticOutputParser.ipynb)

## Overview


This tutorial covers how to perform ```PydanticOutputParser``` using ```pydantic```.

The ```PydanticOutputParser``` is a class that helps transform the output of a language model into **structured information** . This class can **provide the information you need in a clear and organized form** instead of a simple text response.


By utilizing this class, you can transform the output of your language model to fit a specific data model, making it easier to process and utilize the information

### Main Method

A ```PydanticOutputParser``` primarily requires the implementation of **two core methods**.


1. **```get_format_instructions()```**: Provide instructions that define the format of the information that the language model should output. 
For example, you can return instructions as a string that describes the fields of data that the language model should output and how they should be formatted. 
These instructions are very important for the language model to structure the output and transform it to fit your specific data model.

2. **```parse()```**: Takes the output of the language model (assumed to be a string) and analyzes and transforms it into a specific structure. 
Use a tool like Pydantic to validate the input string against a predefined schema and transform it into a data structure that follows that schema.


### Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [A Chain with the Basic StrOutputParser](#a-chain-with-the-basic-stroutputparser)
- [Using the PydanticOutputParser](#using-the-pydanticoutputparser)

### References

- [Pydantic Official Document](https://docs.pydantic.dev/latest/)
---


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setups, useful functions, and utilities for tutorials. 
- You can check out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_openai",
        "pydantic",
        "itertools",
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
        "LANGCHAIN_PROJECT": "01-PydanticOuputParser",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

Environment variables have been set successfully.
You can alternatively set API keys, such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
# Import the required libraries
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
```

Below is an example of an email conversation stored in the variable ```email_conversation``` .



```python
email_conversation = """
From: John (John@bikecorporation.me)
To: Kim (Kim@teddyinternational.me)
Subject: “ZENESIS” bike distribution cooperation and meeting schedule proposal
Dear Mr. Kim,

I am John, Senior Executive Director at Bike Corporation. I recently learned about your new bicycle model, "ZENESIS," through your press release. Bike Corporation is a company that leads innovation and quality in the field of bicycle manufacturing and distribution, with long-time experience and expertise in this field.

We would like to request a detailed brochure for the ZENESIS model. In particular, we need information on technical specifications, battery performance, and design aspects. This information will help us further refine our proposed distribution strategy and marketing plan.

Additionally, to discuss the possibilities for collaboration in more detail, I propose a meeting next Tuesday, January 15th, at 10:00 AM. Would it be possible to meet at your office to have this discussion?

Thank you.

Best regards,
John
Senior Executive Director
Bike Corporation
"""
```

Let's create a chain that utilizes the basic ```StrOutputParser```.

```python
from itertools import chain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "Please extract the important parts of the following email.\n\n{email_conversation}"
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

chain = prompt | llm | StrOutputParser()

answer = chain.stream({"email_conversation": email_conversation})


#  A function for real-time output (streaming)
def stream_response(response, return_output=False):
    """
    Streams the response from the AI model, processing and printing each chunk.

    This function iterates over each item in the 'response' iterable. If an item is an instance of AIMessageChunk, it extracts and prints the content.
    If the item is a string, it prints the string directly.
    Optionally, the function can return the concatenated string of all response chunks.

    Args:
    - response (iterable): An iterable of response chunks, which can be AIMessageChunk objects or strings.
    - return_output (bool, optional): If True, the function returns the concatenated response string. The default is False.

    Returns:
    - str: If ```return_output``` is True, the concatenated response string. Otherwise, nothing is returned.
    """
    answer = ""
    for token in response:
        if isinstance(token, AIMessageChunk):
            answer += token.content
            print(token.content, end="", flush=True)
        elif isinstance(token, str):
            answer += token
            print(token, end="", flush=True)
    if return_output:
        return answer


output = stream_response(answer, return_output=True)
```

<pre class="custom">**Important Parts of the Email:**
    
    - **Sender:** John (Senior Executive Director, Bike Corporation)
    - **Recipient:** Kim (Teddy International)
    - **Subject:** ZENESIS bike distribution cooperation and meeting schedule proposal
    - **Request:** Detailed brochure for the ZENESIS model, specifically information on:
      - Technical specifications
      - Battery performance
      - Design aspects
    - **Purpose:** To refine distribution strategy and marketing plan for ZENESIS.
    - **Proposed Meeting:** 
      - Date: Tuesday, January 15th
      - Time: 10:00 AM
      - Location: Kim's office
    - **Closing:** Thank you and best regards.</pre>

```python
answer = chain.invoke({"email_conversation": email_conversation})
print(answer)
```

<pre class="custom">**Important Parts of the Email:**
    
    - **Sender:** John (Senior Executive Director, Bike Corporation)
    - **Recipient:** Kim (Teddy International)
    - **Subject:** ZENESIS bike distribution cooperation and meeting schedule proposal
    - **Request:** Detailed brochure for the ZENESIS model, specifically information on:
      - Technical specifications
      - Battery performance
      - Design aspects
    - **Purpose:** To refine distribution strategy and marketing plan for ZENESIS.
    - **Proposed Meeting:** 
      - Date: Tuesday, January 15th
      - Time: 10:00 AM
      - Location: Kim's office
    - **Closing:** Thank you and best regards.
</pre>

```python
print(output)
```

<pre class="custom">**Important Parts of the Email:**
    
    - **Sender:** John (Senior Executive Director, Bike Corporation)
    - **Recipient:** Kim (Teddy International)
    - **Subject:** ZENESIS bike distribution cooperation and meeting schedule proposal
    - **Request:** Detailed brochure for the ZENESIS model, specifically information on:
      - Technical specifications
      - Battery performance
      - Design aspects
    - **Purpose:** To refine distribution strategy and marketing plan for ZENESIS.
    - **Proposed Meeting:** 
      - Date: Tuesday, January 15th
      - Time: 10:00 AM
      - Location: Kim's office
    - **Closing:** Thank you and best regards.
</pre>

## Using the ```PydanticOutputParser```
When provided with email content like the one above, we can parse the email information using the class defined in the ```Pydantic``` style below.

For reference, the ```description``` inside the ```Field``` serves as guidance for extracting key information from text-based responses. LLMs rely on this description to extract the required information. Therefore, it is crucial that this description is accurate and clear.

```python
class EmailSummary(BaseModel):
    person: str = Field(description="The sender of the email")
    email: str = Field(description="The email address of the sender")
    subject: str = Field(description="The subject of the email")
    summary: str = Field(description="A summary of the email content")
    date: str = Field(
        description="The meeting date and time mentioned in the email content"
    )


# Create an output parser using the PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=EmailSummary)
```

```python
# Print the instructions.
print(parser.get_format_instructions())
```

<pre class="custom">The output should be formatted as a JSON instance that conforms to the JSON schema below.
    
    As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
    the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
    
    Here is the output schema:
    ```
    {"properties": {"person": {"description": "The sender of the email", "title": "Person", "type": "string"}, "email": {"description": "The email address of the sender", "title": "Email", "type": "string"}, "subject": {"description": "The subject of the email", "title": "Subject", "type": "string"}, "summary": {"description": "A summary of the email content", "title": "Summary", "type": "string"}, "date": {"description": "The meeting date and time mentioned in the email content", "title": "Date", "type": "string"}}, "required": ["person", "email", "subject", "summary", "date"]}
    ```
</pre>

Let's define a prompt including the following variables:

1. ```question```: Receives the user's question.
2. ```email_conversation```: Inputs the content of the email conversation.
3. ```format```: Specifies the format.


```python
prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. 

QUESTION:
{question}

EMAIL CONVERSATION:
{email_conversation}

FORMAT:
{format}
"""
)


# Apply the partial method and fill in the ```format``` variable in advance
prompt = prompt.partial(format=parser.get_format_instructions())
```

Next, create a chain.

```python
chain = prompt | llm
```

Execute the chain and review the results.

```python
# Execute the chain and print the result.
response = chain.stream(
    {
        "email_conversation": email_conversation,
        "question": "Extract the main content of the email.",
    }
)

# The result is provided in JSON format.
output = stream_response(response, return_output=True)
```

<pre class="custom">```json
    {
      "person": "John",
      "email": "John@bikecorporation.me",
      "subject": "ZENESIS bike distribution cooperation and meeting schedule proposal",
      "summary": "John from Bike Corporation requests a detailed brochure for the ZENESIS bike model, including technical specifications, battery performance, and design aspects. He also proposes a meeting on January 15th at 10:00 AM to discuss collaboration possibilities.",
      "date": "January 15th, 10:00 AM"
    }
    ```</pre>

Finally, use the parser to parse the results and convert them into an ```EmailSummary``` object.

```python
structured_output = parser.parse(output)
print(structured_output)
```

<pre class="custom">person='John' email='John@bikecorporation.me' subject='ZENESIS bike distribution cooperation and meeting schedule proposal' summary='John from Bike Corporation requests a detailed brochure for the ZENESIS bike model, including technical specifications, battery performance, and design aspects. He also proposes a meeting on January 15th at 10:00 AM to discuss collaboration possibilities.' date='January 15th, 10:00 AM'
</pre>

### Creating a Chain Including the ```PydanticOutputParser```

You can generate the output as a Pydantic object that you define.

```python
# Reconstruct the entire chain by adding an output parser
chain = prompt | llm | parser
```

```python
# Execute the chain and print the results
response = chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": "Extract the main content of the email.",
    }
)

# The results are output in the form of an EmailSummary object
print(response)
```

<pre class="custom">person='John' email='John@bikecorporation.me' subject='ZENESIS bike distribution cooperation and meeting schedule proposal' summary='John from Bike Corporation requests a detailed brochure for the ZENESIS bike model, including technical specifications, battery performance, and design aspects. He also proposes a meeting on January 15th at 10:00 AM to discuss collaboration possibilities.' date='January 15th, 10:00 AM'
</pre>

### ```with_structured_output()```

By adding an output parser using ```.with_structured_output(Pydantic)```, you can convert the output into a Pydantic object.

```python
llm_with_structured = ChatOpenAI(
    temperature=0, model_name="gpt-4o"
).with_structured_output(EmailSummary)
```

```python
# Call the ```invoke()``` method and print the result
answer = llm_with_structured.invoke(email_conversation)
answer
```




<pre class="custom">EmailSummary(person='John', email='John@bikecorporation.me', subject='“ZENESIS” bike distribution cooperation and meeting schedule proposal', summary='John, Senior Executive Director at Bike Corporation, is interested in the ZENESIS bicycle model and requests a detailed brochure including technical specifications, battery performance, and design aspects. He proposes a meeting to discuss potential collaboration.', date='Tuesday, January 15th, at 10:00 AM')</pre>



**Note**

One thing to note is that the ```.with_structured_output()``` method does not support the ```stream()``` method.
