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

# Self-querying
- Author: [Hye-yoon Jeong](https://github.com/Hye-yoonJeong)
- Peer Review: 
- Proofread : [Juni Lee](https://www.linkedin.com/in/ee-juni)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/08-SelfQueryRetriever.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/10-Retriever/08-SelfQueryRetriever.ipynb)
## Overview

```SelfQueryRetriever``` is a retriever equipped with the capability to generate and resolve queries autonomously.

```SelfQueryRetriever``` converts the natural language input provided by the user into a **structured query** using a **query-constructing LLM chain** . This **structured query** is then used to retrieve documents from the **vector store** .

Through this process, ```SelfQueryRetriever``` goes beyond merely comparing the user's input query with the content of stored documents semantically, and **extracts filters on the metadata** from the user's query and executes those filters to retrieve relevant documents.

The list of **self-querying retrievers** supported by LangChain can be found [here](https://python.langchain.com/docs/integrations/retrievers/self_query).

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Sample Data](#sample-data)
- [SelfQueryRetriever](#selfqueryretriever)
- [Query Constructor Chain](#query-constructor-chain)

### References
- [LangChain Documentation: Self-querying retrievers](https://python.langchain.com/docs/integrations/retrievers/self_query)
- [LangChain cookbook: Building hotel room search with self-querying retrieval](https://github.com/langchain-ai/langchain/blob/master/cookbook/self_query_hotel_search.ipynb)
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
        "langchain_openai",
        "langchain_chroma",
        "langchain_community",
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
        "LANGCHAIN_PROJECT": "Self-querying",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
%pip install lark
```

<pre class="custom">Requirement already satisfied: lark in c:\users\hyj89\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-arohchi8-py3.11\lib\site-packages (1.2.2)
    Note: you may need to restart the kernel to use updated packages.
</pre>

    
    [notice] A new release of pip is available: 24.1 -> 24.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    

## Sample Data

Let's build a vector store that enables similarity search based on the descriptions and metadata of some cosmetic products.

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Generate sample data with the descriptions and metadata of cosmetic products.
docs = [
    Document(
        page_content="A hyaluronic acid serum packed with moisture, delivering hydration deep into the skin.",
        metadata={"year": 2024, "category": "Skincare", "user_rating": 4.7},
    ),
    Document(
        page_content="A matte-finish foundation with 24-hour wear, covering pores and providing a natural skin appearance.",
        metadata={"year": 2023, "category": "Makeup", "user_rating": 4.5},
    ),
    Document(
        page_content="A hypoallergenic cleansing oil made with plant-based ingredients, gently removes makeup and impurities.",
        metadata={"year": 2023, "category": "Cleansers", "user_rating": 4.8},
    ),
    Document(
        page_content="A brightening cream with vitamin C, brightens dull skin tones for a radiant complexion.",
        metadata={"year": 2023, "category": "Skincare", "user_rating": 4.6},
    ),
    Document(
        page_content="A long-lasting lipstick with vivid color payoff and a moisturizing texture for all-day comfort.",
        metadata={"year": 2024, "category": "Makeup", "user_rating": 4.4},
    ),
    Document(
        page_content="A tone-up sunscreen with SPF50+/PA++++, offering high UV protection and keeping the skin safe.",
        metadata={"year": 2024, "category": "Sunscreen", "user_rating": 4.9},
    ),
]

# Build a vector store
vectorstore = Chroma.from_documents(
    docs, OpenAIEmbeddings(model="text-embedding-3-small")
)
```

## SelfQueryRetriever

To instantiate the ```retriever``` , you need to define **metadata fields** and **a brief description of the document contents** in advance using the ```AttributeInfo``` class.

In this example, the metadata for cosmetic products is defined as follows:

- ```category``` : String type, represents the category of the cosmetic product and takes one of the following values: ['Skincare', 'Makeup', 'Cleansers', 'Sunscreen'].
- ```year``` : Integer type, represents the year the cosmetic product was released.
- ```user_rating``` : Float type, represents the user rating in the range of 1 to 5.

```python
from langchain.chains.query_constructor.schema import AttributeInfo

# Generate metadata field
metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The category of the cosmetic product. One of ['Skincare', 'Makeup', 'Cleansers', 'Sunscreen']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the cosmetic product was released",
        type="integer",
    ),
    AttributeInfo(
        name="user_rating",
        description="A user rating for the cosmetic product, ranging from 1 to 5",
        type="float",
    ),
]
```

Create ```retriever``` object with ```SelfQueryRetriever.from_llm``` method.

- ```llm```: Large language model
- ```vectorstore```: Vector store
- ```document_contents```: Description of the contents of the documents
- ```metadata_field_info```: Metadata field information

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

# Define the LLM to use
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief summary of a cosmetic product",
    metadata_field_info=metadata_field_info,
)
```

Now, let's test this ```SelfQueryRetriever``` with some example queries.

```python
# Self-query retrieval
retriever.invoke("Please recommend products with a rating of 4.8 or higher.")
```




<pre class="custom">[Document(metadata={'category': 'Cleansers', 'user_rating': 4.8, 'year': 2023}, page_content='A hypoallergenic cleansing oil made with plant-based ingredients, gently removes makeup and impurities.'),
     Document(metadata={'category': 'Sunscreen', 'user_rating': 4.9, 'year': 2024}, page_content='A tone-up sunscreen with SPF50+/PA++++, offering high UV protection and keeping the skin safe.')]</pre>



```python
# Self-query retrieval
retriever.invoke("Please recommend products released in 2023.")
```




<pre class="custom">[Document(metadata={'category': 'Cleansers', 'user_rating': 4.8, 'year': 2023}, page_content='A hypoallergenic cleansing oil made with plant-based ingredients, gently removes makeup and impurities.'),
     Document(metadata={'category': 'Skincare', 'user_rating': 4.6, 'year': 2023}, page_content='A brightening cream with vitamin C, brightens dull skin tones for a radiant complexion.'),
     Document(metadata={'category': 'Makeup', 'user_rating': 4.5, 'year': 2023}, page_content='A matte-finish foundation with 24-hour wear, covering pores and providing a natural skin appearance.')]</pre>



```python
# Self-query retrieval
retriever.invoke("Please recommend products in the Sunscreen category.")
```




<pre class="custom">[Document(metadata={'category': 'Sunscreen', 'user_rating': 4.9, 'year': 2024}, page_content='A tone-up sunscreen with SPF50+/PA++++, offering high UV protection and keeping the skin safe.')]</pre>



```SelfQueryRetriever``` can also be used to retrieve items with two or more conditions.

```python
# Self-query retrieval
retriever.invoke(
    "Please recommend products in the 'Makeup' category with a rating of 4.5 or higher."
)
```




<pre class="custom">[Document(metadata={'category': 'Makeup', 'user_rating': 4.5, 'year': 2023}, page_content='A matte-finish foundation with 24-hour wear, covering pores and providing a natural skin appearance.')]</pre>



You can also specify **the number of documents to retrieve** using the argument ```k``` when using ```SelfQueryRetriever``` .

This can be done by passing ```enable_limit=True``` to the constructor.

```python
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief summary of a cosmetic product",
    metadata_field_info=metadata_field_info,
    enable_limit=True,  # Enable to limit the search result.
    search_kwargs={"k": 2},  # Limit the number of retrieved documents to 2.
)
```

There are 3 products released in 2023, but by setting the value of ```k``` to 2, only 2 products are retrieved.

```python
# Self-query retrieval
retriever.invoke("Please recommend products released in 2023.")
```




<pre class="custom">[Document(metadata={'category': 'Cleansers', 'user_rating': 4.8, 'year': 2023}, page_content='A hypoallergenic cleansing oil made with plant-based ingredients, gently removes makeup and impurities.'),
     Document(metadata={'category': 'Skincare', 'user_rating': 4.6, 'year': 2023}, page_content='A brightening cream with vitamin C, brightens dull skin tones for a radiant complexion.')]</pre>



However, you can also limit the number of search results by directly specifying the number of search results in the query without explicitly specifying ```search_kwargs``` in the code.

```python
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief summary of a cosmetic product",
    metadata_field_info=metadata_field_info,
    enable_limit=True,  # Enable to limit the search result.
)

# Self-query retrieval
retriever.invoke("Please recommend one product released in 2023.")
```




<pre class="custom">[Document(metadata={'category': 'Cleansers', 'user_rating': 4.8, 'year': 2023}, page_content='A hypoallergenic cleansing oil made with plant-based ingredients, gently removes makeup and impurities.')]</pre>



```python
# Self-query retrieval
retriever.invoke("Please recommend 2 products released in 2023.")
```




<pre class="custom">[Document(metadata={'category': 'Cleansers', 'user_rating': 4.8, 'year': 2023}, page_content='A hypoallergenic cleansing oil made with plant-based ingredients, gently removes makeup and impurities.'),
     Document(metadata={'category': 'Skincare', 'user_rating': 4.6, 'year': 2023}, page_content='A brightening cream with vitamin C, brightens dull skin tones for a radiant complexion.')]</pre>



## Query Constructor Chain

To see what happens internally and to have more custom control, we can construct a ```retriever``` from scratch.

First, we need to create a ```query_constructor``` chain that generates structured queries. Here, we use the ```get_query_constructor_prompt``` function to retrieve the prompt that helps constructing queries.

```python
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

# Retrieve the query constructor prompt using the document content description and metadata field information.
prompt = get_query_constructor_prompt(
    "Brief summary of a cosmetic product",  # Document content description
    metadata_field_info,  # Metadata field information
)

# Create StructuredQueryOutputParser
output_parser = StructuredQueryOutputParser.from_components()

# Create query_constructor chain
query_constructor = prompt | llm | output_parser
```

To check the content of the prompt, use the ```prompt.format``` method to pass the string ```"dummy question"``` to the ```query``` parameter and print the result.

```python
# Print prompt
print(prompt.format(query="dummy question"))
```

<pre class="custom">Your goal is to structure the user's query to match the request schema provided below.
    
    << Structured Request Schema >>
    When responding use a markdown code snippet with a JSON object formatted in the following schema:
    
    ```json
    {
        "query": string \ text string to compare to document contents
        "filter": string \ logical condition statement for filtering documents
    }
    ```
    
    The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
    
    A logical condition statement is composed of one or more comparison and logical operation statements.
    
    A comparison statement takes the form: `comp(attr, val)`:
    - `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
    - `attr` (string):  name of attribute to apply the comparison to
    - `val` (string): is the comparison value
    
    A logical operation statement takes the form `op(statement1, statement2, ...)`:
    - `op` (and | or | not): logical operator
    - `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to
    
    Make sure that you only use the comparators and logical operators listed above and no others.
    Make sure that filters only refer to attributes that exist in the data source.
    Make sure that filters only use the attributed names with its function names if there are functions applied on them.
    Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
    Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
    Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
    
    << Example 1. >>
    Data Source:
    ```json
    {
        "content": "Lyrics of a song",
        "attributes": {
            "artist": {
                "type": "string",
                "description": "Name of the song artist"
            },
            "length": {
                "type": "integer",
                "description": "Length of the song in seconds"
            },
            "genre": {
                "type": "string",
                "description": "The song genre, one of "pop", "rock" or "rap""
            }
        }
    }
    ```
    
    User Query:
    What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre
    
    Structured Request:
    ```json
    {
        "query": "teenager love",
        "filter": "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180), eq(\"genre\", \"pop\"))"
    }
    ```
    
    
    << Example 2. >>
    Data Source:
    ```json
    {
        "content": "Lyrics of a song",
        "attributes": {
            "artist": {
                "type": "string",
                "description": "Name of the song artist"
            },
            "length": {
                "type": "integer",
                "description": "Length of the song in seconds"
            },
            "genre": {
                "type": "string",
                "description": "The song genre, one of "pop", "rock" or "rap""
            }
        }
    }
    ```
    
    User Query:
    What are songs that were not published on Spotify
    
    Structured Request:
    ```json
    {
        "query": "",
        "filter": "NO_FILTER"
    }
    ```
    
    
    << Example 3. >>
    Data Source:
    ```json
    {
        "content": "Brief summary of a cosmetic product",
        "attributes": {
        "category": {
            "description": "The category of the cosmetic product. One of ['Skincare', 'Makeup', 'Cleansers', 'Sunscreen']",
            "type": "string"
        },
        "year": {
            "description": "The year the cosmetic product was released",
            "type": "integer"
        },
        "user_rating": {
            "description": "A user rating for the cosmetic product, ranging from 1 to 5",
            "type": "float"
        }
    }
    }
    ```
    
    User Query:
    dummy question
    
    Structured Request:
    
</pre>

Call the ```query_constructor.invoke``` method to process the given query.

```python
query_output = query_constructor.invoke(
    {
        # Call the query constructor to generate a query.
        "query": "Please recommend skincare products released in 2023 with a rating of 4.5 or higher."
    }
)
```

```python
# Print query
query_output.filter.arguments
```




<pre class="custom">[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='category', value='Skincare'),
     Comparison(comparator=<Comparator.GTE: 'gte'>, attribute='year', value=2023),
     Comparison(comparator=<Comparator.GTE: 'gte'>, attribute='user_rating', value=4.5)]</pre>



The core component of a ```SelfQueryRetriever``` is the **query constructor** . To build an effective retrieval system, it is essential to ensure that the **query constructor** is well defined.

To achieve this, you need to adjust the **prompt**, **examples within the prompt**, and **attribute descriptions** .

### Structured Query Translator

You can also create a structured query using the **structured query translator** .

**Structured query translator** converts a query into metadata filters compatible with the syntax of the vector store with ```StructuredQuery``` object.

```python
from langchain.retrievers.self_query.chroma import ChromaTranslator

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,  # The query_constructor chain created in the previous step.
    vectorstore=vectorstore,  # Specify the VectorStore
    structured_query_translator=ChromaTranslator(),  # Query translator
)
```

Use the ```retriever.invoke``` method to generate an answer for the given question.

```python
retriever.invoke(
    "Please recommend skincare products released in 2023 with a rating of 4.5 or higher."
)
```




<pre class="custom">[Document(metadata={'category': 'Skincare', 'user_rating': 4.7, 'year': 2024}, page_content='A hyaluronic acid serum packed with moisture, delivering hydration deep into the skin.'),
     Document(metadata={'category': 'Skincare', 'user_rating': 4.6, 'year': 2023}, page_content='A brightening cream with vitamin C, brightens dull skin tones for a radiant complexion.')]</pre>


