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

# ConversationBufferMemory

- Author: [Sungchul Kim](https://github.com/rlatjcj)
- Peer Review : [Shinar12](https://github.com/Shinar12), [BAEM1N](https://github.com/BAEM1N), [YellowGangneng](https://github.com/YellowGangneng)
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/01-ConversationBufferMemory.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/05-Memory/01-ConversationBufferMemory.ipynb)
## Overview

This tutorial introduces ```ConversationBufferMemory```, a memory class that stores conversation history in a buffer.

Typically, no additional processing is required. Sometimes, however, it may be necessary when the conversation history exceeds the model's context window.

In this tutorial, we will learn how to use ```ConversationBufferMemory``` to store and retrieve conversation history.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Extracting messages as strings](#extracting-messages-as-strings)
- [Extracting messages as HumanMessage and AIMessage objects](#extracting-messages-as-humanmessage-and-aimessage-objects)
- [Applying to a Chain](#applying-to-a-chain)

### References
- [LangChain: ConversationBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer.ConversationBufferMemory.html)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "ConversationBufferMemory",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it.

**[Note]**  
- This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



## Extracting messages as strings

After storing conversation messages, this memory allows you to extract messages into a variable.


```python
from langchain.memory import ConversationBufferMemory
```

```python
memory = ConversationBufferMemory()
memory
```

<pre class="custom">/tmp/ipykernel_188575/2223904900.py:1: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
      memory = ConversationBufferMemory()
</pre>




    ConversationBufferMemory(chat_memory=InMemoryChatMessageHistory(messages=[]))



You can use the ```save_context(inputs, outputs)``` method to save conversation records.

- This method accepts two arguments, ```inputs``` and ```outputs```.
- ```inputs``` stores the user's question, and ```outputs``` stores the AI's answer.
- The conversation record is stored internally under the ```history``` key.
- You can then use the ```load_memory_variables``` method to retrieve and inspect the saved conversation history.


```python
# inputs: dictionary(key: "human" or "ai", value: question)
# outputs: dictionary(key: "ai" or "human", value: answer)
memory.save_context(
    inputs={
        "human": "Hello, I want to open a bank account remotely. How do I start?",
    },
    outputs={
        "ai": "Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification."
    },
)
```

```python
memory
```




<pre class="custom">ConversationBufferMemory(chat_memory=InMemoryChatMessageHistory(messages=[HumanMessage(content='Hello, I want to open a bank account remotely. How do I start?', additional_kwargs={}, response_metadata={}), AIMessage(content="Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification.", additional_kwargs={}, response_metadata={})]))</pre>



The ```load_memory_variables({})``` method of the memory object returns the complete message history as a string.

```python
# Check the message history stored in the 'history' key.
print(memory.load_memory_variables({})["history"])
```

<pre class="custom">Human: Hello, I want to open a bank account remotely. How do I start?
    AI: Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification.
</pre>

```python
memory.save_context(
    inputs={
        "human": "Yes, I've prepared my ID for identity verification. What should I do next?"
    },
    outputs={
        "ai": "Thank you. Please upload the front and back of your ID clearly. We will proceed with the identity verification process next."
    },
)
```

```python
# Check the message history stored in the 'history' key.
print(memory.load_memory_variables({})["history"])
```

<pre class="custom">Human: Hello, I want to open a bank account remotely. How do I start?
    AI: Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification.
    Human: Yes, I've prepared my ID for identity verification. What should I do next?
    AI: Thank you. Please upload the front and back of your ID clearly. We will proceed with the identity verification process next.
</pre>

```python
# Save 2 conversations.
memory.save_context(
    inputs={
        "human": "I uploaded the photo. How do I proceed with identity verification?"
    },
    outputs={
        "ai": "We have confirmed the photo you uploaded. Please proceed with identity verification through your mobile phone. Please enter the verification number sent by text."
    },
)
memory.save_context(
    inputs={
        "human": "I entered the verification number. How do I open an account now?"
    },
    outputs={
        "ai": "Identity verification has been completed. Please select the type of account you want and enter the necessary information. You can choose the type of deposit, currency, etc."
    },
)
```

```python
# Check the conversation history stored in the 'history' key.
print(memory.load_memory_variables({})["history"])
```

<pre class="custom">Human: Hello, I want to open a bank account remotely. How do I start?
    AI: Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification.
    Human: Yes, I've prepared my ID for identity verification. What should I do next?
    AI: Thank you. Please upload the front and back of your ID clearly. We will proceed with the identity verification process next.
    Human: I uploaded the photo. How do I proceed with identity verification?
    AI: We have confirmed the photo you uploaded. Please proceed with identity verification through your mobile phone. Please enter the verification number sent by text.
    Human: I entered the verification number. How do I open an account now?
    AI: Identity verification has been completed. Please select the type of account you want and enter the necessary information. You can choose the type of deposit, currency, etc.
</pre>

```python
# Save 2 more conversations.
memory.save_context(
    inputs={
        "human": "I've entered all the information. What's the next step?",
    },
    outputs={
        "ai": "I've confirmed the information you've entered. The account opening process is almost complete. Please agree to the terms of use and confirm the account opening."
    },
)
memory.save_context(
    inputs={
        "human": "I've completed all the steps. Has the account been opened?",
    },
    outputs={
        "ai": "Yes, the account has been opened. Your account number and related information have been sent to the email you registered. If you need additional help, please contact us at any time. Thank you!"
    },
)
```

```python
# Check the conversation history stored in the 'history' key.
print(memory.load_memory_variables({})["history"])
```

<pre class="custom">Human: Hello, I want to open a bank account remotely. How do I start?
    AI: Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification.
    Human: Yes, I've prepared my ID for identity verification. What should I do next?
    AI: Thank you. Please upload the front and back of your ID clearly. We will proceed with the identity verification process next.
    Human: I uploaded the photo. How do I proceed with identity verification?
    AI: We have confirmed the photo you uploaded. Please proceed with identity verification through your mobile phone. Please enter the verification number sent by text.
    Human: I entered the verification number. How do I open an account now?
    AI: Identity verification has been completed. Please select the type of account you want and enter the necessary information. You can choose the type of deposit, currency, etc.
    Human: I've entered all the information. What's the next step?
    AI: I've confirmed the information you've entered. The account opening process is almost complete. Please agree to the terms of use and confirm the account opening.
    Human: I've completed all the steps. Has the account been opened?
    AI: Yes, the account has been opened. Your account number and related information have been sent to the email you registered. If you need additional help, please contact us at any time. Thank you!
</pre>

## Extracting messages as ```HumanMessage``` and ```AIMessage``` objects

Setting ```return_messages=True``` returns ```HumanMessage``` and ```AIMessage``` objects.

```python
memory = ConversationBufferMemory(return_messages=True)

memory.save_context(
    inputs={
        "human": "Hello, I want to open a bank account remotely. How do I start?",
    },
    outputs={
        "ai": "Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification.",
    },
)

memory.save_context(
    inputs={
        "human": "Yes, I've prepared my ID for identity verification. What should I do next?"
    },
    outputs={
        "ai": "Thank you. Please upload the front and back of your ID clearly. We will proceed with the identity verification process next."
    },
)

memory.save_context(
    inputs={
        "human": "I uploaded the photo. How do I proceed with identity verification?"
    },
    outputs={
        "ai": "We have confirmed the photo you uploaded. Please proceed with identity verification through your mobile phone. Please enter the verification number sent by text."
    },
)
```

```python
# Check the conversation history stored in the 'history' key.
memory.load_memory_variables({})["history"]
```




<pre class="custom">[HumanMessage(content='Hello, I want to open a bank account remotely. How do I start?', additional_kwargs={}, response_metadata={}),
     AIMessage(content="Hello! I'm glad you want to open an account. First, please prepare your ID for identity verification.", additional_kwargs={}, response_metadata={}),
     HumanMessage(content="Yes, I've prepared my ID for identity verification. What should I do next?", additional_kwargs={}, response_metadata={}),
     AIMessage(content='Thank you. Please upload the front and back of your ID clearly. We will proceed with the identity verification process next.', additional_kwargs={}, response_metadata={}),
     HumanMessage(content='I uploaded the photo. How do I proceed with identity verification?', additional_kwargs={}, response_metadata={}),
     AIMessage(content='We have confirmed the photo you uploaded. Please proceed with identity verification through your mobile phone. Please enter the verification number sent by text.', additional_kwargs={}, response_metadata={})]</pre>



## Applying to a Chain

Let's apply ```ConversationBufferMemory``` to a ```ConversationChain```.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# Create an LLM model.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create a ConversationChain.
conversation = ConversationChain(
    # Use ConversationBufferMemory.
    llm=llm,
    memory=ConversationBufferMemory(),
)
```

<pre class="custom">/tmp/ipykernel_188575/3549519840.py:8: LangChainDeprecationWarning: The class `ConversationChain` was deprecated in LangChain 0.2.7 and will be removed in 1.0. Use :meth:`~RunnableWithMessageHistory: https://python.langchain.com/v0.2/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html` instead.
      conversation = ConversationChain(
</pre>

Proceed with the conversation using the ```ConversationChain```.

```python
# Start the conversation.
response = conversation.predict(
    input="Hello, I want to open a bank account remotely. How do I start?"
)
print(response)
```

<pre class="custom">Hello again! Opening a bank account remotely is a convenient process, and I’m happy to guide you through it. Here’s a detailed step-by-step guide to help you get started:
    
    1. **Research and Choose a Bank**: Start by researching different banks to find one that suits your needs. Consider factors like account fees, interest rates, customer service, and any special features they offer. Online reviews and comparison websites can be helpful in making your decision.
    
    2. **Visit the Bank’s Website**: Once you’ve chosen a bank, head to their official website. Look for the section dedicated to personal banking or account opening. Most banks have a clear pathway for opening accounts online.
    
    3. **Select the Type of Account**: Decide on the type of account you want to open. Common options include checking accounts, savings accounts, or a combination of both. Some banks also offer specialized accounts like student accounts or high-yield savings accounts.
    
    4. **Prepare Required Documents**: Gather the necessary documents and information. Typically, you’ll need:
       - A government-issued ID (such as a passport or driver’s license)
       - Social Security Number or Tax Identification Number
       - Proof of address (like a utility bill or lease agreement)
       - Employment and income information
    
    5. **Fill Out the Application**: Complete the online application form by entering your personal details, selecting your account preferences, and agreeing to the bank’s terms and conditions.
    
    6. **Verify Your Identity**: The bank will likely require you to verify your identity. This might involve uploading a photo of your ID, answering security questions, or participating in a video call with a bank representative.
    
    7. **Fund Your Account**: You’ll need to make an initial deposit to activate your account. This can usually be done via a transfer from another bank account, a credit card, or a check.
    
    8. **Receive Confirmation**: Once your application is approved and your account is funded, you’ll receive confirmation from the bank. This might include your account number, online banking login details, and information on how to manage your account.
    
    9. **Set Up Online Banking**: If you haven’t already, set up online banking to manage your account. This will allow you to check your balance, transfer funds, pay bills, and more.
    
    If you have any specific questions about a particular bank or need further assistance, feel free to ask!
</pre>

Verify if the system remembers the previous conversation.

```python
# Send a request to summarize the previous conversation in bullet points.
response = conversation.predict(input="Summarize the previous answer in bullet points.")
print(response)
```

<pre class="custom">Certainly! Here’s a summarized version of the steps to open a bank account remotely:
    
    - **Research and Choose a Bank**: Consider fees, interest rates, customer service, and special features.
    - **Visit the Bank’s Website**: Navigate to the personal banking or account opening section.
    - **Select the Type of Account**: Choose between checking, savings, or specialized accounts.
    - **Prepare Required Documents**: Gather ID, Social Security Number, proof of address, and income information.
    - **Fill Out the Application**: Enter personal details and agree to terms and conditions.
    - **Verify Your Identity**: Upload ID, answer security questions, or join a video call.
    - **Fund Your Account**: Make an initial deposit via transfer, credit card, or check.
    - **Receive Confirmation**: Get account number and online banking details.
    - **Set Up Online Banking**: Manage your account online for transactions and bill payments.
</pre>
