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

# Neo4j Vector Index

- Author: [Jongho](https://github.com/XaviereKU)
- Design: 
- Peer Review: 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview
Neo4j is a Graph database backed by vector store and can be deployed locally or on cloud.

In this tutorial we utilize its ability to store vectors only, and deal with its real ability, Graph database, later.

To encode data into vector, we use ```OpenAIEmbedding```, but you can use any embedding you want.

Furthermore, you need to note that you should read about ```Cypher```, declarative query language for Neo4j, to fully utilize Neo4j.

We use some Cypher queries but will not go deeply. You can visit Cypher official document web site in References.

For more information, visit [Neo4j](https://neo4j.com/).

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Setup Neo4j](#setup-neo4j)
	- [Getting started with Aura](#getting-started-with-aura)
	- [Getting started with Docker](#getting-started-with-docker)
- [Credentials](#credentials)
- [Initialization](#initialization)
	- [List Indexes](#list-indexs)
	- [Create Index](#create-index)
	- [Delete Index](#delete-index)
	- [Select Embedding model](#select-embeddings-model)
	- [Data Preprocessing](#data-preprocessing)
- [Manage vector store](#manage-vector-store)
	- [Add items to vector store](#add-items-to-vector-store)
	- [Delete items from vector store](#delete-items-from-vector-store)
	- [Scroll items from vector store](#scroll-items-from-vector-store)
	- [(Advanced)Scroll items with query](#advanced-scroll-items-with-query)
- [Similarity search](#similarity-search)

### References

- [Cypher](https://neo4j.com/docs/cypher-manual/current/introduction/)
- [Neo4j Docker Installation](https://hub.docker.com/_/neo4j)
- [Neo4j Official Installation guide](https://neo4j.com/docs/operations-manual/current/installation/)
- [Neo4j Python SDK document](https://neo4j.com/docs/api/python-driver/current/index.html)
- [Neo4j document](https://neo4j.com/docs/)
- [Langchain Neo4j document](https://python.langchain.com/docs/integrations/vectorstores/neo4jvector/)

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.
- We built ```Neo4jDB``` class from Python SDK of ```Neo4j```. Langchain also supports neo4j vector store class but it lacks some methods like delete. Look neo4j_interface.py in utils

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Pip install necessary package
%pip install -qU neo4j
```

<pre class="custom">Note: you may need to restart the kernel to use updated packages.
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
        "neo4j",
        "nltk",
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
        "OPENAI_API_KEY": "Your OpenAI API Key",
        "LANGCHAIN_API_KEY": "Your LangChain API Key",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Neo4j",
        "NEO4J_URI": "Your Neo4j Aura URI",
        "NEO4J_USERNAME": "Your Neo4j Aura username",
        "NEO4J_PASSWORD": "Your Neo4j Aura password",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">False</pre>



## Setup Neo4j
We have two options to start with. Cloud or local deployment.

In this tutorial, we will user Cloud service, called ```Aura``` provided by ```Neo4j```.

But we will also describe how to deploy ```Neo4j``` with docker.

### Getting started with Aura
You can create a new **Neo4j Aura** account at [Neo4j](https://neo4j.com/) offical website.

Visit web site and click Get Started Free at top right.

If you done signing in, you will se a button, **Create instance** and after that you will see your username and password.

To get your API Key, click **Download and continue** to download a txt file which contains API key to connect your **NEO4j Aura** .

### Getting started with Docker
We now describe how to run ```Neo4j``` using docker.

To run Neo4j container, we use the following command.
```
docker run \
    -itd \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --env=NEO4J_AUTH=none \
    --name neo4j \
    neo4j
```

You can visit **Neo4j Docker installation** reference to check more detailed information.

**[NOTE]**
* ```Neo4j``` also supports macOS, windows and Linux native deployment. Visit **Neo4j Official Installation guide** reference for more detail.
* ```Neo4j``` community edition only supports one database.

## Credentials
Now, if you successfully create your own account for Aura, you will get your ```NEO4J_URI```, ```NEO4J_USERNAME```, ```NEO4J_USERPASSWORD```.

Add it to environmental variable above or add it to your ```.env``` file.

```python
import os
import time
from utils.neo4j_interface import Neo4jDB

# set uri, username, password
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

client = Neo4jDB(uri=uri, username=username, password=password)
```

<pre class="custom">Connected to Neo4j database
    Connection info
    URI=neo4j+s://3ed1167e.databases.neo4j.io
    username=neo4j
    Neo4j version is above 5.23
</pre>

Once we established connection to Aura ```Neo4j``` database, connection info using ```get_api_key``` method.

```python
# get connection info
client.get_api_key()
```

## Initialization
If you are succesfully connected to **Neo4j Aura**, there are some basic indexes already created.

But, in this tutorial we will create a new indexand will add items(nodes) to it.

To do this, we now look how to manage indexes.

To manage indexes, we will see how to:
* List indexes
* Create new index
* Delete index

### List Indexs
Before create a new index, let's check indexes already in the ```Neo4j``` database

```python
# get name list of indexes
names = client.list_indexes()

print(names)
```

<pre class="custom">['index_343aff4e', 'index_f7700477']
</pre>

### Create Index

Now we will create a new index.

This can be done by calling `create_index` method, which will return an object connected to newly created index.

If an index exists with the same name, the method will print out notification.

When we create a new index, we must provide embedding object or dimension of vector, and ```metric``` to use for similarity search.

In this tutorial we will pass `OpenAIEmbeddings` when we create a new index.


**[ NOTE ]**
- If you pass dimension of vector instead of embedding object, this must match the dimension of embeded vector of your choice of embedding model.
- An embedding object must have ```embed_query``` and ```embed_documents``` methods.
- ```metric``` is used to set distance method for similarity search. ```Neo4j``` supports **cosine** and **euclidean** .

```python
# Initialize OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# set index_name and node_label
index_name = "tutorial_index"
node_label = "tutorial_node"

# create a new index
index = client.create_index(
    embedding=embeddings, index_name=index_name, node_label=node_label
)

if isinstance(index, Neo4jDB):
    print("Index creation was successful")

# check name list of indexes
names = client.list_indexes()
print(names)
```

<pre class="custom">Created index information
    Index name: tutorial_index
    Node label: tutorial_node
    Similarity metric: COSINE
    Embedding dimension: 1536
    Embedding node property: embedding
    Text node property: text
    
    Index creation was successful
    ['index_343aff4e', 'index_f7700477', 'tutorial_index']
</pre>

### Delete Index

We can delete specific index by calling `delete_index` method.

Delete ```tutorial_index``` we created above and then create it again to use later.

```python
# delete index
client.delete_index("tutorial_index")

# print name list of indexes
names = client.list_indexes()
if "tutorial_index" not in names:
    print(f"Index deleted succesfully ")
    print(names)

# recreate the tutorial_index
index = client.create_index(
    embedding=embeddings, index_name="tutorial_index", node_label="tutorial_node"
)
```

<pre class="custom">Index deleted succesfully 
    ['index_343aff4e', 'index_f7700477']
    Created index information
    Index name: tutorial_index
    Node label: tutorial_node
    Similarity metric: COSINE
    Embedding dimension: 1536
    Embedding node property: embedding
    Text node property: text
    
</pre>

### Select Embeddings model

We also can change embedding model.

In this subsection we use ```text-embedding-3-large``` model to create a new index with it

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings_large = OpenAIEmbeddings(model="text-embedding-3-large")
```

```python
# create new index
index2 = client.create_index(
    embedding=embeddings_large,
    index_name="tutorial_index_2",
    node_label="tutorial_node_2",
)
```

<pre class="custom">Created index information
    Index name: tutorial_index_2
    Node label: tutorial_node_2
    Similarity metric: COSINE
    Embedding dimension: 3072
    Embedding node property: embedding
    Text node property: text
    
</pre>

### Data Preprocessing

Below is the preprocessing process for general documents.

- Need to extract **metadata** from documents
- Filter documents by minimum length.
  
- Determine whether to use ```basename``` or not. Default is ```False```.
  - ```basename``` denotes the last value of the filepath.
  - For example, **document.pdf** will be the ```basename``` for the filepath **./data/document.pdf** .

```python
# This is a long document we can split up.
data_path = "./data/the_little_prince.txt"
with open(data_path, encoding="utf8") as f:
    raw_text = f.read()
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# define text splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# split raw text by splitter.
split_docs = text_splitter.create_documents([raw_text])

# print one of documents to check its structure
print(split_docs[0])
```

<pre class="custom">page_content='The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)'
</pre>

Now we preprocess splited document to extract author, page and source metadata while fit the data to store it into `Neo4j`

```python
# preprocess raw documents
processed_docs = client.preprocess_documents(
    split_docs=split_docs,
    metadata_keys=["source", "page", "author"],
    min_length=5,
    use_basename=True,
    source=data_path,
    author="Saiot-Exupery",
)

# print one of preprocessed document to chekc its structure
print(processed_docs[0])
```

<pre class="custom">page_content='The Little Prince
    Written By Antoine de Saiot-Exupery (1900〜1944)' metadata={'source': 'the_little_prince.txt', 'page': 1, 'author': 'Saiot-Exupery'}
</pre>

## Manage vector store
Once you have created your vector store, we can interact with it by adding and deleting different items.

Also, you can scroll data from the store with filter or with ```Cypher``` query.


### Add items to vector store

We can add items to our vector store by using the ```upsert_documents``` or ```upsert_documents_parallel``` method.

If you pass ids along with documents, then ids will be used, but if you do not pass ids, it will be created based `page_content` using md5 hash function.

Basically, ```upsert_document``` and ```upsert_document_parallel``` methods do upsert not insert, based on **id** of the item.

So if you provided id and want to update data, you must provide the same id that you provided at first upsertion.

We will upsert data to index, tutorial_index, with ```upsert_documents``` method for the first half, and with ```upsert_documents_parallel``` for the second half.

```python
from uuid import uuid4

# make ids for each document
uuids = [str(uuid4()) for _ in range(len(processed_docs))]


# upsert documents
total_number = len(processed_docs)
upsert_result = index.upsert_documents(
    processed_docs[: total_number // 2], ids=uuids[: total_number // 2]
)

# upsert documents parallel
upsert_parallel_result = index.upsert_documents_parallel(
    processed_docs[total_number // 2 :],
    batch_size=32,
    max_workers=8,
    ids=uuids[total_number // 2 :],
)

result = upsert_result + upsert_parallel_result

# check number of ids upserted
print(len(result))

# check manual ids are the same as output ids
print("Manual Ids == Output Ids:", sorted(result) == sorted(uuids))
```


### Delete items from vector store

We can delete nodes by filter or ids with `delete_node` method.


For example, we will delete **the first page**, that is `page` 1, of the little prince, and try to scroll it.

```python
# define filter
filters = {"page": 1, "author": "Saiot-Exupery"}

# call delete_node method
result = index.delete_node(filters=filters)
print(result)
```

<pre class="custom">True
</pre>

```python
# define filter for scroll data
filters = {"page": 1, "author": "Saiot-Exupery"}

# call scroll method
result = index.scroll_nodes(filters=filters)
print(result)
```

<pre class="custom">Scroll nodes by filter
    []
</pre>

As you can see, we successfully deleted a node which satisfies the given filter.

To make sure only 1 data deleted, let's check the total number of nodes in index `vector`

```python
# scroll vector index
result = index.scroll_nodes(limit=None)
print("The number of nodes in vector: {}".format(len(result)))
```

<pre class="custom">The number of nodes in vector: 1358
</pre>

```python
sorted(result, key=lambda x: x["metadata"]["page"])[0]
```




<pre class="custom">{'id': '8f9ed6b2-4fc5-4c23-a32b-d53acc72a68a',
     'metadata': {'author': 'Saiot-Exupery',
      'text': '[ Antoine de Saiot-Exupery ]',
      'source': 'the_little_prince.txt',
      'page': 2}}</pre>



Now delete 5 items using ```ids```.

```python
# delete item by ids
ids = uuids[1:6]

# call delete_node method
result = index.delete_node(ids=ids)
print(result)
```

<pre class="custom">True
</pre>

```python
# scroll vector index
result = index.scroll_nodes(limit=None)
print("The number of nodes in vector: {}".format(len(result)))
```

<pre class="custom">The number of nodes in vector: 1353
</pre>

### Scroll items from vector store
You can scroll items(nodes) in store by calling ```scroll_nodes``` method with filters or ids.

If you are you scroll by filter and you passed keys and values, those will be treated as **MUST** condition, which means the nodes that match all the conditions will be returned.

```python
# define scroll filter
filters = {"author": "Saiot-Exupery", "page": 10}

# get nodes
result = index.scroll_nodes(filters=filters)
print(result)
```

<pre class="custom">Scroll nodes by filter
    [{'id': '8fcae3d1-8d41-4010-9458-6324a87c6cb4', 'metadata': {'author': 'Saiot-Exupery', 'text': 'learned to fly a plane. Five years later, he would leave the military in order to begin flying air', 'source': 'the_little_prince.txt', 'page': 10}}]
</pre>

```python
# get nodes by ids
result = index.scroll_nodes(ids=uuids[11])
print(result)
```

<pre class="custom">Scroll nodes by ids
    [{'id': '9f4790f0-6f1b-428c-87c7-dbc3b909852a', 'metadata': {'author': 'Saiot-Exupery', 'text': 'For Saint-Exupéry, it was a grand adventure - one with dangers lurking at every corner. Flying his', 'source': 'the_little_prince.txt', 'page': 12}}]
</pre>

### (Advanced) Scroll items with query
Provided method, ```scroll_nodes``` only support **AND** condition for multiple (key, value) pairs.

But if you use ```Cypher```, more complicated condition can be used to scroll items.

```python
# create cypher query
query = "MATCH (n) WHERE n.page IN [10,11,12] AND n.author='Saiot-Exupery' RETURN n.page, n.author, n.text"

# scroll items with query
result = index.scroll_nodes(query=query)

for item in result:
    print(item)
```

<pre class="custom">Scroll nodes by query
    {'n.page': 10, 'n.author': 'Saiot-Exupery', 'n.text': 'learned to fly a plane. Five years later, he would leave the military in order to begin flying air'}
    {'n.page': 11, 'n.author': 'Saiot-Exupery', 'n.text': 'to begin flying air mail between remote settlements in the Sahara desert.'}
    {'n.page': 12, 'n.author': 'Saiot-Exupery', 'n.text': 'For Saint-Exupéry, it was a grand adventure - one with dangers lurking at every corner. Flying his'}
</pre>

## Similarity search
As ```Neo4j``` supports vector database, you can also do similarity search.

The similarity is calculated by the metric you set when you created the index to search on.

In this tutorial we will search items on **tutorial_index** , which has metric **cosine** .

To do search, we call ```search``` method.

You can pass the raw text(to ```query``` paramter), or embeded vector of the text(to ```embeded_query``` paramter) when calling ```search```.

```python
# do search. top_k is the number of documents in the result
res_with_text = index.search(query="Does the little prince have a friend?", top_k=5)

# print out top 2 results
print("RESULT BY RAW QUERY")
for i in range(2):
    print(res_with_text[i])

# embed query
embeded_query = embeddings.embed_query("Does the little prince have a friend?")

# do search with embeded vector value
res_with_embed = index.search(embeded_query=embeded_query, top_k=5)

# print out top 2 results
print()
print("RESULT BY EMBEDED QUERY")
for i in range(2):
    print(res_with_embed[i])
```

<pre class="custom">RESULT BY RAW QUERY
    {'text': '"My friend the fox--" the little prince said to me.', 'metadata': {'id': '70d75baa-3bed-4751-b0cf-98157e190756', 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt', 'page': 1087, 'embedding': None}, 'score': 0.947}
    {'text': 'And the little prince asked himself:', 'metadata': {'id': '9e779e02-1d2b-4252-a8f4-78bae7866af5', 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt', 'page': 492, 'embedding': None}, 'score': 0.946}
    
    RESULT BY EMBEDED QUERY
    {'text': '"My friend the fox--" the little prince said to me.', 'metadata': {'id': '70d75baa-3bed-4751-b0cf-98157e190756', 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt', 'page': 1087, 'embedding': None}, 'score': 0.947}
    {'text': 'And the little prince asked himself:', 'metadata': {'id': '9e779e02-1d2b-4252-a8f4-78bae7866af5', 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt', 'page': 492, 'embedding': None}, 'score': 0.946}
</pre>

That's it!

You can now do the basics of how to use Neo4j.

If you want to do more advanced tasks, please refer to `Neo4j` official API documents and official Python SDK of `Neo4j` API documents.
