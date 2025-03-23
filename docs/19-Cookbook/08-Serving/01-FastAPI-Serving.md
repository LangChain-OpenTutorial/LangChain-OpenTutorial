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

# FastAPI Serving

- Author: [Donghak Lee](https://github.com/stsr1284)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial is about FastAPI Serving.
FastAPI is one of the python web frameworks that supports asynchronous programming and is very fast.

In this tutorial, we will implement the following FastAPI examples.
- Implement different types of parameters
- Declare an input/output data model
- Serve a langchain with FastAPI

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is FastAPI](#what-is-fastapi)
- [FastAPI Fast Tutorial](#fastapi-fast-tutorial)
- [FastAPI Serving of LangChain](#fastapi-serving-of-langchain)

### References

- [FastAPI](https://fastapi.tiangolo.com/)
- [langchain_reference](https://python.langchain.com/api_reference/index.html#)
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
        "uvicorn",
        "fastapi",
        "pydantic",
        "typing",
        "pydantic",
        "langchain_openai",
        "langchain_core",
        "langchain_community",
        "langchain_chroma",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 24.3.1 -> 25.0.1
    [notice] To update, run: pip install --upgrade pip
</pre>

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "FastAPI-Serving",
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




<pre class="custom">False</pre>



## What is FastAPI
FastAPI is a modern, high-performance web framework for building APIs with Python, based on standard Python type hints.

Key features include:

- Speed: Built on Starlette and Pydantic, it is fully compatible with these tools and delivers extremely high performance—on par with NodeJS and Go—making it one of the fastest Python frameworks available.
- Fast coding: Increases feature development speed by approximately 200% to 300%.
- Fewer bugs: Reduces human (developer) errors by around 40%.
- Intuitive: Offers excellent editor support with autocomplete everywhere, reducing debugging time.
- Easy: Designed to be simple to use and learn, cutting down on the time needed to read documentation.
- Robust: Provides production-ready code along with automatically generated interactive documentation.
- Standards-based: Built on open, fully compatible standards for APIs, such as OpenAPI (formerly known as Swagger) and JSON Schema.

### FastAPI Features
Key Features:

- Supports asynchronous programming.
- Provides automatically updating interactive API documentation (Swagger UI), allowing you to interact with your API directly.
- Boosts coding speed with excellent editor support through autocomplete and type checking.
- Seamlessly integrates security and authentication, enabling use without compromising your database or data models while incorporating numerous security features—including those from Starlette.
- Automatically handles dependency injection, making it easy to use.
- Built on Starlette and Pydantic, ensuring full compatibility.

### How to run a server

You can find the API documentation in the `/docs` path and interact with it directly via the `Try it out` button.

To spin up a live server, you can copy the code to a `.py` file and run it by typing `uvicorn [file name]:[FastAPI instance] --reload` in a shell.

For this tutorial, we'll temporarily run the server from the `.ipynb` file with the following code
```python
import uvicorn
import nest_asynci

nest_asyncio.apply()
uvicorn.run(app)
```

## FastAPI Fast Tutorial
Quickly learn how to communicate with the API via FastAPI.
- Create an instance of FastAPI with `FastAPI()`.
- Define a path operation decorator to communicate with the path by setting the HTTP Method on the path.

### How to run code
When you run the code block, it's loading infinitely, which means the server is running.

We recommend testing the API at `http://127.0.0.1:8000/docs`

```python
import uvicorn
import nest_asyncio
from fastapi import FastAPI

app = FastAPI()  ## create FastAPI instance


# FastAPI decorators are used to set routing paths
@app.get("/")
def read_root():
    return "hello"


nest_asyncio.apply()
uvicorn.run(app)
```

<pre class="custom">INFO:     Started server process [26086]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
</pre>

    INFO:     127.0.0.1:54044 - "GET / HTTP/1.1" 200 OK
    

    INFO:     Shutting down
    INFO:     Waiting for application shutdown.
    INFO:     Application shutdown complete.
    INFO:     Finished server process [26086]
    

### Define Path Parameters

- You can set parameters on a path and use them as variables inside a function by setting the arguments of the function.
- You can declare the type of the path parameters in your function using Python's standard type annotations.
- FastAPI will automatically ‘parse’ the request to validate the type of the data.



```python
app = FastAPI()  # create FastAPI instance


# Declare route parameters by adding parameters to the route.
@app.get("/chat/{chat_id}")
def read_chat(chat_id: int):  # Pass the path parameter as a parameter of the function.
    return {"chat_id": chat_id}


nest_asyncio.apply()
uvicorn.run(app)
```

<pre class="custom">INFO:     Started server process [26086]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
</pre>

    INFO:     127.0.0.1:54093 - "GET / HTTP/1.1" 404 Not Found
    INFO:     127.0.0.1:54094 - "GET /chat/123 HTTP/1.1" 200 OK
    INFO:     127.0.0.1:54094 - "GET /chat/hello HTTP/1.1" 422 Unprocessable Entity
    

    INFO:     Shutting down
    INFO:     Waiting for application shutdown.
    INFO:     Application shutdown complete.
    INFO:     Finished server process [26086]
    

### Define Query Parameters
- If you declare a function parameter other than as part of a path parameter, FastAPI automatically interprets it as a query parameter.
- Query parameters can be declared as optional parameters by setting their default value to `None`.


```python
app = FastAPI()


# Declare the path parameter and the query parameter.
@app.get("/chat/{chat_id}")
def read_item(chat_id: int, item_id: int, q: str | None = None):
    # item_id, q is the query parameter, and q is an optional parameter.
    return {"chat_id": chat_id, "item_id": item_id, "q": q}


nest_asyncio.apply()
uvicorn.run(app)
```

<pre class="custom">INFO:     Started server process [26086]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Shutting down
    INFO:     Waiting for application shutdown.
    INFO:     Application shutdown complete.
    INFO:     Finished server process [26086]
</pre>

### Define Request Model
- It can be defined using the `Pydantic` model.
- Request is the data sent from the client to the API. Response is the data that the API sends back to the client.
- You can declare the request body, path, and query parameters together.

**Note:** It is not recommended to include a body in a `GET` request.

```python
from pydantic import BaseModel

app = FastAPI()


# Define an Item class that is the Request Model.
class Item(BaseModel):
    name: str
    description: str | None = None  # Optionally set it by declaring a default value.
    price: float
    tax: float | None = None


@app.post("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: str | None = None):
    result = {"item_id": item_id, **item.model_dump()}
    # if q exists, add q to result
    if q:
        result.update({"q": q})
    # add price_with_tax if tax exists
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        result.update({"price_with_tax": price_with_tax})
    return result


nest_asyncio.apply()
uvicorn.run(app)
```

<pre class="custom">INFO:     Started server process [26086]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Shutting down
    INFO:     Waiting for application shutdown.
    INFO:     Application shutdown complete.
    INFO:     Finished server process [26086]
</pre>

### Define Response Model

You can define the return type by adding the `response_model` parameter to the path operation decorator.

This allows you to exclude sensitive data received from the input model from the output.

FastAPI provides the following features when setting the output model
- Converting output data to type declarations
- Data validation
- Add JSON schema to the response in the Swagger UI

```python
from typing import Any

app = FastAPI()


class PostIn(BaseModel):
    postId: str
    password: str
    description: str | None = None  # Optionally set it by declaring a default value.
    content: str


class PostOut(BaseModel):
    postId: str
    description: str | None = None  # Optionally set it by declaring a default value.
    content: str


@app.post("/posts", response_model=PostOut)
async def create_Post(post: PostIn) -> Any:
    return post


nest_asyncio.apply()
uvicorn.run(app)
```

<pre class="custom">INFO:     Started server process [26086]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Shutting down
    INFO:     Waiting for application shutdown.
    INFO:     Application shutdown complete.
    INFO:     Finished server process [26086]
</pre>

## FastAPI Serving of LangChain
- Try serving a langchain with the fastAPI.
- Use what you have learnt above.
- Implement stream output in the fastAPI.

```python
from typing import List
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

DB_PATH = "../data/chroma_db"

load_dotenv()


# Define the chat output data structure.
class ChatReturnType(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")


# Defines the chat stream output data structure.
class ChatReturnStreamType(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")


# Define the Add contents input data type.
class AddContentsInType(BaseModel):
    content: List[str]
    source: List[dict]


# Define the Add contents output data type.
class AddContentsOutType(BaseModel):
    content: List[str]
    source: List[dict]
    id: List[str]


chroma = Chroma(
    collection_name="FastApiServing",
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(),
)

retriever = chroma.as_retriever(
    search_kwargs={
        "k": 4,
    }
)

parser = JsonOutputParser(pydantic_object=ChatReturnType)

prompt = ChatPromptTemplate(
    [
        ("system", "You are a friendly AI assistant. Answer questions concisely.’"),
        (
            "system",
            "Answer the question based only on the following context: {context}",
        ),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | JsonOutputParser()
)

app = FastAPI()


@app.post("/invoke", response_model=ChatReturnType)
def sync_chat(message: str):
    response = chain.invoke(message)
    return response


@app.post("/ainvoke", response_model=ChatReturnType)
async def async_chat(message: str):
    response = await chain.ainvoke(message)
    return response


@app.post("/stream", response_model=ChatReturnStreamType)
def sync_stream_chat(message: str):
    def event_stream():
        try:
            for chunk in chain.stream(message):
                if len(chunk) > 0:
                    yield f"{chunk}"
        except Exception as e:
            yield f"data: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/astream", response_model=ChatReturnStreamType)
async def async_stream_chat(message: str):
    async def event_stream():
        try:
            async for chunk in chain.astream(message):
                if len(chunk) > 0:
                    yield f"{chunk}"
        except Exception as e:
            yield f"data: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/add-contents", response_model=AddContentsOutType)
async def add_content(input: AddContentsInType):
    id = chroma.add_texts(input.content, metadatas=input.source)
    output = input.model_copy(update={"id": id})
    return output


@app.post("/async-add-contents", response_model=AddContentsOutType)
async def async_add_content(input: AddContentsInType):
    id = await chroma.aadd_texts(input.content, metadatas=input.source)
    output = input.model_copy(update={"id": id})
    return output


nest_asyncio.apply()
uvicorn.run(app)
```

<pre class="custom">INFO:     Started server process [26086]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
</pre>

    INFO:     127.0.0.1:56950 - "POST /add-contents HTTP/1.1" 200 OK
    

    INFO:     Shutting down
    INFO:     Waiting for application shutdown.
    INFO:     Application shutdown complete.
    INFO:     Finished server process [26086]
    
