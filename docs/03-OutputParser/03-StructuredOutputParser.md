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

# Structured Output Parser

- Author: [Yoolim Han](https://github.com/hohosznta)
- Peer Review: [ranian963](https://github.com/ranian963), [asummerz](https://github.com/asummerz)
- Proofread : [BokyungisaGod](https://github.com/BokyungisaGod)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/03-StructuredOutputParser.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/03-StructuredOutputParser.ipynb)
## Overview

The ```StructuredOutputParser``` is a valuable tool for formatting Large Language Model (LLM) responses into dictionary structures, enabling the return of multiple fields as key/value pairs. 
While Pydantic and JSON parsers offer robust capabilities, the ```StructuredOutputParser``` is particularly effective for less powerful models, such as local models with fewer parameters. It is especially beneficial for models with lower intelligence compared to advanced models like GPT or Claude. 
By utilizing the ```StructuredOutputParser```, developers can maintain data integrity and consistency across various LLM applications, even when operating with models that have reduced parameter counts.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Implementing the StructuredOutputParser](#implementing-the-structuredoutputparser)

### References

- [LangChain ChatOpenAI API reference](https://python.langchain.com/docs/integrations/chat/openai/)
- [LangChain Structured output parser](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.structured.StructuredOutputParser.html#langchain.output_parsers.structured.StructuredOutputParser)
---

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup along with useful functions and utilities for tutorials. 
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
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "03-StructuredOutputParser",
    }
)

```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)

```




<pre class="custom">False</pre>



## Implementing the ```StructuredOutputParser```

### Using ```ResponseSchema``` with ```StructuredOutputParser```
*   Define a response schema using the ```ResponseSchema``` class to include the answer to the user's question and a ```description``` of the source (website) used.

*   Initialize ```StructuredOutputParser``` with ```response_schemas``` to structure the output according to the defined response schema.

**[Note]**
When using local models, Pydantic parsers may frequently fail to work properly. In such cases, using ```StructuredOutputParser``` can be a good alternative solution.

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Response to the user's question
response_schemas = [
    ResponseSchema(name="answer", description="Answer to the user's question"),
    ResponseSchema(
        name="source",
        description="The `source` used to answer the user's question, which should be a `website URL`.",
    ),
]
# Initialize the structured output parser based on the response schemas
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

```

### Embedding Response Schemas into Prompts 

Create a ```PromptTemplate``` to format user questions and embed parsing instructions for structured outputs.

```python
from langchain_core.prompts import PromptTemplate
# Parse the format instructions.
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    # Set up the template to answer the user's question as well as possible.
    template="answer the user's question as well as possible.\n{format_instructions}\n{question}",
    # Use 'question' as the input variable.
    input_variables=["question"],
    # Use 'format_instructions' as a partial variable.
    partial_variables={"format_instructions": format_instructions},
)
```

### Integrating with ```ChatOpenAI``` and Running the ```Chain```

Combine the ```PromptTemplate``` , ```ChatOpenAI``` model , and ```StructuredOutputParser``` into a ```chain``` . Finally, run the ```chain``` with a specific ```question``` to produce results.

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)  # Initialize the ChatOpenAI model

chain = prompt | model | output_parser  # Connect the prompt, model, and output parser

# Ask the question, "What is the largest desert in the world?"
chain.invoke({"question": "What is the largest desert in the world?"})
```




<pre class="custom">{'answer': 'The largest desert in the world is the Antarctic Desert.',
     'source': 'https://www.worldatlas.com/articles/what-is-the-largest-desert-in-the-world.html'}</pre>



### Using Streamed Outputs

Use the ```chain.stream``` method to receive a streaming response to the ```question``` , "How many players are on a soccer team?"

```python
for s in chain.stream({"question": "How many players are on a soccer team?"}):
    # Stream the output
    print(s)
```

<pre class="custom">{'answer': 'A standard soccer team consists of 11 players on the field at a time.', 'source': 'https://www.fifa.com/who-we-are/news/what-are-the-rules-of-football-2040008'}
</pre>
