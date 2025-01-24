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

# Structured Output Chain 

- Author: [JeongHo Shin](https://github.com/ThePurpleCollar)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial demonstrates how to implement Structured output generation using LangChain and OpenAI's language models.

We'll build a quiz generation system that creates multiple-choice questions with consistent formatting and structure.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Implementing Structured Output Chain](#implementing-structured-output-chain)
- [Invoking Generation Chain](#invoking-generation-chain)

### References

----

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.


**[Note]**

The langchain-opentutorial is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
Check out the  [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain-anthropic",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
    ],
    verbose=False,
    upgrade=False,
)
```

You can set API keys in a `.env` file or set them manually.

[Note] If you’re not using the `.env` file, no worries! Just enter the keys directly in the cell below, and you’re good to go.

```python
from dotenv import load_dotenv
from langchain_opentutorial import set_env

# Attempt to load environment variables from a .env file; if unsuccessful, set them manually.
if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "Structured-Output-Chain", 
        }
    )
```

## Implementing Structured Output Chain

This tutorial walks you through the process of generating 4-option multiple-choice quizzes for a given topic.

The `Quiz` class defines the structure of the quiz, including the question, difficulty level, and four answer options.

A `ChatOpenAI` instance leverages the **GPT-4o** model for natural language processing, while a `ChatPromptTemplate` specifies the conversational instructions for generating the quizzes dynamically.

```python
# Import required modules and libraries
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Define the Quiz class - Represents the structure of a 4-option multiple-choice quiz
class Quiz(BaseModel):
    """Extracts information for a 4-option multiple-choice quiz"""

    question: str = Field(..., description="The quiz question")  # Quiz question
    level: str = Field(
        ..., description="The difficulty level of the quiz (easy, medium, hard)"
    )
    options: List[str] = Field(..., description="The 4 answer options for the quiz")  # Answer options


# Set up the GPT-4o model with appropriate parameters
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Define a prompt template to guide the model in generating quizzes
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a world-famous quizzer and generate quizzes in structured formats.",
        ),
        (
            "human",
            "Please create a 4-option multiple-choice quiz related to the topic provided below. "
            "If possible, base the question on existing trivia but do not directly include details from the topic in the question. "
            "\nTOPIC:\n{topic}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

# Create a structured output model to match the Quiz class structure
llm_with_structured_output = llm.with_structured_output(Quiz)

# Combine the prompt and the structured output model into a single chain
chain = prompt | llm_with_structured_output
```

## Invoking Generation Chain

In this section, we demonstrate how to invoke the **Structured Output Chain** to generate quizzes dynamically. The chain combines a prompt template and a structured output model to ensure the output adheres to the desired Quiz structure.

```python
# Request the generation of a quiz based on a given topic
generated_quiz = chain.invoke({"topic": "Korean Food"})
```

```python
# Print the generated quiz
print(f"{generated_quiz.question} (Difficulty: {generated_quiz.level})\n")
for i, opt in enumerate(generated_quiz.options):
    print(f"{i+1}) {opt}")
```

<pre class="custom">Which of the following is a traditional Korean dish made by fermenting vegetables, primarily napa cabbage and Korean radishes, with a variety of seasonings? (Difficulty: medium)
    
    1) Kimchi
    2) Sushi
    3) Tacos
    4) Paella
</pre>
