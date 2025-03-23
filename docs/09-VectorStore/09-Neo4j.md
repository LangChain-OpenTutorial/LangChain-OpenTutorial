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
```Neo4j``` is a graph database backed by vector store and can be deployed locally or on cloud.

In this tutorial we utilize its ability to store vectors only, and deal with its real ability, graph database, later.

To encode data into vector, we use ```OpenAIEmbedding```, but you can use any embedding you want.

Furthermore, you need to note that you should read about ```Cypher```, declarative query language for ```Neo4j```, to fully utilize ```Neo4j```.

We use some Cypher queries but will not go deeply. You can visit Cypher official document web site in References.

For more information, visit [Neo4j](https://neo4j.com/).

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Setup Neo4j](#setup-neo4j)
- [Credentials](#credentials)
- [Initialization](#initialization)
- [Manage vector store](#manage-vector-store)
- [Similarity search](#similarity-search)

### References

- [Cypher](https://neo4j.com/docs/cypher-manual/current/introduction/)
- [Neo4j Docker Installation](https://hub.docker.com/_/neo4j)
- [Neo4j Official Installation guide](https://neo4j.com/docs/operations-manual/current/installation/)
- [Neo4j Python SDK document](https://neo4j.com/docs/api/python-driver/current/index.html)
- [Neo4j document](https://neo4j.com/docs/)
- [Langchain Neo4j document](https://python.langchain.com/docs/integrations/vectorstores/neo4jvector/)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.
- We built the ```Neo4jDocumentManager``` class from Python SDK of ```Neo4j```. LangChain supports ```Neo4j``` vector store class but it lacks some methods like ```delete```. You can check these methods in neo4j_interface.py in utils directory.

```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install necessary package
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
        "OPENAI_API_KEY": "Your OPENAI API KEY",
        "LANGCHAIN_API_KEY": "Your LangChain API KEY",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "Neo4j",
        "NEO4J_URI": "Your Neo4j URI",
        "NEO4J_USERNAME": "Your Neo4j username",
        "NEO4J_PASSWORD": "Your Neo4j password",
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
We have two options to start with: cloud or local deployment.

In this tutorial, we will use the cloud service called ```Aura```, provided by ```Neo4j```.

We will also describe how to deploy ```Neo4j``` using ```Docker```.

### Getting started with Aura
You can create a new **Neo4j Aura** account on the [Neo4j](https://neo4j.com/) official website.

Visit the website and click "Get Started" Free at the top right.

Once you have signed in, you will see a button, **Create instance**, and after that, you will see your username and password.

To get your API key, click **Download and continue** to download a .txt file that contains the API key to connect your **NEO4j Aura** .

### Getting started with Docker
Here is the description for how to run ```Neo4j``` using ```Docker```.

To run **Neo4j container** , use the following command.
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
* ```Neo4j``` also supports native deployment on macOS, Windows and Linux. Visit the **Neo4j Official Installation guide** reference for more details.
* The ```Neo4j community edition``` only supports one database.

## Credentials
Now, if you successfully create your own account for Aura, you will get your ```NEO4J_URI```, ```NEO4J_USERNAME```, ```NEO4J_USERPASSWORD```.

Add it to environmental variable above or your ```.env``` file.

```python
import os
import time
import neo4j

# set uri, username, password
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

# Connect to Neo4j client
client = neo4j.GraphDatabase.driver(uri=uri, auth=(username, password))
```

## Initialization
If you are successfully connected to **Neo4j Aura**, some basic indexes are already created.

But, in this tutorial we will create a new index and add items(nodes) to it.

To do this, we now look at how to manage indexes.

To manage indexes, we will see how to:
* List indexes
* Create a new index
* Delete an index

### Define ```Neo4jIndexManager```

**Neo4j** uses **Cypher** , which is similar to an SQL query.

So, when you try to list indexes you have, you need to use **Cypher** . 

But as a tutorial, to make it easier, we defined a class to manager indexes.

```python
from utils.neo4j_interface import Neo4jIndexManager

indexManger = Neo4jIndexManager(client)
```

### List Indexs
Before create a new index, let's check indexes already in the ```Neo4j``` database

```python
# get name list of indexes
names = indexManger.list_indexes()

print(names)
```

<pre class="custom">['index_343aff4e', 'index_f7700477']
</pre>

### Create Index

Now, we will create a new index.

This can be done by calling the ```create_index``` method, which will return an object connected to the newly created index.

If an index exists with the same name, the method will print out a notification.

When creating a new index, we must provide an embedding object or the dimension of vector, along with a ```metric``` to use for similarity search.

If the index created successfully or already exists, the ```create_index``` method will return a ```Neo4jDocumentManager``` object that can add, delete, search or scroll through items in the index.

In this tutorial we will pass ```OpenAIEmbeddings``` when creating a new index.


**[ NOTE ]**
- If you pass the dimension of a vector instead of an embedding object, it must match the dimension of the embeded vector of the embedding model that you choose.
- An embedding object must have ```embed_query``` and ```embed_documents``` methods.
- The ```metric``` parameter is used to set distance metric for similarity search. ```Neo4j``` supports **cosine** and **euclidean** distance.

```python
# Initialize OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# set index_name and node_label
index_name = "tutorial_index"
node_label = "tutorial_node"

# create a new index
try:
    tutorial_index = indexManger.create_index(
        embeddings, index_name=index_name, metric="cosine", node_label=node_label
    )
except Exception as e:
    print("Index creation failed due to")
    print(type(e))
    print(str(e))

# check name list of indexes
names = indexManger.list_indexes()
print()
print(f"Indexes in database: {names}")
```

<pre class="custom">Created index information
    ('Index name: tutorial_index', 'Node label: tutorial_node', 'Similarity metric: COSINE', 'Embedding dimension: 1536', 'Embedding node property: embedding', 'Text node property: text')
    Index creation successful. Return Neo4jDBManager object.
    
    Indexes in database: ['index_343aff4e', 'index_f7700477', 'tutorial_index']
</pre>

### Delete Index

We can delete a specific index by calling the ```delete_index``` method.

Delete ```tutorial_index``` that we created above, and then recreate for later use.

```python
# delete index
indexManger.delete_index("tutorial_index")

# print name list of indexes
names = indexManger.list_indexes()
if "tutorial_index" not in names:
    print("Index deleted succesfully ")
    print(f"Indexes in database: {names}")
    print()

# recreate the tutorial_index
tutorial_index = indexManger.create_index(
    embedding=embeddings, index_name="tutorial_index", node_label="tutorial_node"
)
```

<pre class="custom">Index deleted succesfully 
    Indexes in database: ['index_343aff4e', 'index_f7700477']
    
    Created index information
    ('Index name: tutorial_index', 'Node label: tutorial_node', 'Similarity metric: COSINE', 'Embedding dimension: 1536', 'Embedding node property: embedding', 'Text node property: text')
    Index creation successful. Return Neo4jDBManager object.
</pre>

### Select Embedding model

We can also change embedding model.

In this subsection, we will use ```text-embedding-3-large``` model to create a new index.

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
tutorial_index_2 = indexManger.create_index(
    embedding=embeddings_large,
    index_name="tutorial_index_2",
    metric="cosine",
    node_label="tutorial_node_2",
)
```

<pre class="custom">Created index information
    ('Index name: tutorial_index_2', 'Node label: tutorial_node_2', 'Similarity metric: COSINE', 'Embedding dimension: 3072', 'Embedding node property: embedding', 'Text node property: text')
    Index creation successful. Return Neo4jDBManager object.
</pre>

### Data Preprocessing

The following describes the preprocessing process for general documents.

- Extract **metadata** from documents.
- Filter documents by minimum length.
  
- Determine whether to use ```basename```. The default is ```False```.
  - The ```basename``` denotes the last value of the filepath.
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

Now we preprocess split document to extract author, page, and source metadata while formatting the data to store it into ```Neo4j```

```python
# define document preprocessor
def preprocess_documents(
    split_docs, metadata_keys, min_length, use_basename=False, **kwargs
):
    metadata = kwargs

    if use_basename:
        assert metadata.get("source", None) is not None, "source must be provided"
        metadata["source"] = metadata["source"].split("/")[-1]

    result_docs = []
    for idx, doc in enumerate(split_docs):
        if len(doc.page_content) < min_length:
            continue
        for k in metadata_keys:
            doc.metadata.update({k: metadata.get(k, "")})
        doc.metadata.update({"page": idx + 1})
        result_docs.append(doc)

    return result_docs
```

```python
# preprocess raw documents
processed_docs = preprocess_documents(
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
Once you have created your vector store, you can interact with it by adding and deleting different items.

Also, you can scroll through data from the store using a filter or a ```Cypher``` query.

### Connect to index
To add, delete, search, or scroll items, we need to initialize an object that is connected to the index we are operating on.

We will connect to ```tutorial_index```. Recall that we used basic ```OpenAIEmbedding``` as a embedding function, and thus we need to pass it when we initialize ```index_manager``` object.

Remember that we can also get ```Neo4jDocumentManager``` object when creating an index, but this time we call it directly to get a ```Neo4jDocumentManager``` object.

```python
# import Neo4jDocumentManager
from utils.neo4j_interface import Neo4jDocumentManager

# connect to tutorial_index
index_manager = Neo4jDocumentManager(
    client=client, index_name="tutorial_index", embedding=embeddings
)
```


### Add items to vector store

We can add items to our vector store by using the ```upsert_documents``` or ```upsert_documents_parallel``` method.

If you pass IDs along with documents, then IDs will be used. However if you do not pass IDs, they will be generated based ```page_content``` using **MD5** hash function.

Basically, ```upsert_document``` and ```upsert_document_parallel``` methods perform an upsert, not insert, based on **ID** of the item.

So if you provided an ID and want to update the data, you must use the same id that you provided at first upsertion.

We will upsert data to the index, ```tutorial_index```, using the ```upsert_documents``` method for the first half, and with ```upsert_documents_parallel``` for the second half.

```python
from uuid import uuid4

# get texts and metadatas from processed documents
texts = [p.page_content for p in processed_docs]
metadatas = [p.metadata for p in processed_docs]

# make manual ids for each processed documents
uuids = [str(uuid4()) for _ in range(len(processed_docs))]

# Get total number of documents
total_number = len(processed_docs)
print("Number of documents:", total_number)
```

<pre class="custom">Number of documents: 1359
</pre>

```python
%%time
# upsert documents
upsert_result = index_manager.upsert(
    
    texts=texts[:total_number//2], metadatas=metadatas[:total_number//2], ids=uuids[:total_number//2]
)
```

<pre class="custom">CPU times: total: 3.88 s
    Wall time: 7.31 s
</pre>

```python
%%time
# upsert documents parallel
upsert_parallel_result = index_manager.upsert_parallel(
    texts = texts[total_number//2 :],
    metadatas = metadatas[total_number//2:],
    ids = uuids[total_number//2:],
    batch_size=32,
    max_workers=8
)
```

<pre class="custom">CPU times: total: 4.47 s
    Wall time: 6.01 s
</pre>

```python
result = upsert_result + upsert_parallel_result

# check number of ids upserted
print(len(result))

# check manual ids are the same as output ids
print("Manual Ids == Output Ids:", sorted(result) == sorted(uuids))
```

<pre class="custom">1359
    Manual Ids == Output Ids: True
</pre>

### Scroll items from vector store
Since we have added some items to our first vector store, named ```tutorial_index``` , we can scroll items from the vector store.

This can be done by calling the ```scroll``` method.

When we scroll items from the vector store, we can pass ```ids``` or ```filters``` to get items that we want, or just call ```scroll``` to get ```k```(*default: 10*) items.

We can get embedded vector values of each items by set ```include_embedding``` True.

Also, by setting ```meta_keys```, we can get metadata that we want. If not set, all metadats, except embeddings, will be returned.

```python
# Do scroll without ids or filters
result1 = tutorial_index.scroll()

# print the number of items scrolled and first item that returned.
print(f"Number of items scrolled: {len(result1)}")
print(result1[0])
```

<pre class="custom">Number of items scrolled: 10
    {'id': '92eaae3a-ff0b-4a87-a823-1c512edbaf77', 'author': 'Saiot-Exupery', 'text': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', 'source': 'the_little_prince.txt', 'page': 1}
</pre>

```python
# Do scroll with filter
result2 = tutorial_index.scroll(filters={"page": [1, 2, 3]})

# print the number of items scrolled and all items that returned.
print(f"Number of items scrolled: {len(result2)}")
for r in result2:
    print(r)
```

<pre class="custom">Number of items scrolled: 3
    {'id': '92eaae3a-ff0b-4a87-a823-1c512edbaf77', 'author': 'Saiot-Exupery', 'text': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', 'source': 'the_little_prince.txt', 'page': 1}
    {'id': '7bea13ca-a5f8-4e03-888e-24018d7e72b5', 'author': 'Saiot-Exupery', 'text': '[ Antoine de Saiot-Exupery ]', 'source': 'the_little_prince.txt', 'page': 2}
    {'id': '40a3d3b0-2052-42b6-b870-089d9519ef96', 'author': 'Saiot-Exupery', 'text': 'Over the past century, the thrill of flying has inspired some to perform remarkable feats of', 'source': 'the_little_prince.txt', 'page': 3}
</pre>

```python
# Do scroll with ids
result3 = tutorial_index.scroll(ids=uuids[:3])

# print the number of items scrolled and all items that returned.
print(f"Number of items scrolled: {len(result3)}")
for r in result3:
    print(r)
```

<pre class="custom">Number of items scrolled: 3
    {'id': '92eaae3a-ff0b-4a87-a823-1c512edbaf77', 'author': 'Saiot-Exupery', 'text': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', 'source': 'the_little_prince.txt', 'page': 1}
    {'id': '7bea13ca-a5f8-4e03-888e-24018d7e72b5', 'author': 'Saiot-Exupery', 'text': '[ Antoine de Saiot-Exupery ]', 'source': 'the_little_prince.txt', 'page': 2}
    {'id': '40a3d3b0-2052-42b6-b870-089d9519ef96', 'author': 'Saiot-Exupery', 'text': 'Over the past century, the thrill of flying has inspired some to perform remarkable feats of', 'source': 'the_little_prince.txt', 'page': 3}
</pre>

```python
# Do scroll with selected meta keys and only 3 items.
result4 = tutorial_index.scroll(meta_keys=["page"], k=3)

# print the number of items scrolled and all items that returned.
print(f"Number of items scrolled: {len(result4)}")
for r in result4:
    print(r)
```

<pre class="custom">Number of items scrolled: 3
    {'page': 1}
    {'page': 2}
    {'page': 3}
</pre>

### Delete items from vector store

We can delete nodes using filter or IDs with the ```delete_node``` method.


For example, we will delete **the first page** (```page``` 1) of the little prince and then try to scroll it.

```python
# define filter
filters = {"page": 1, "author": "Saiot-Exupery"}

# call delete_node method
result = tutorial_index.delete(filters=filters)
print(result)
```

<pre class="custom">True
</pre>

```python
# Check if item is deleted
result = tutorial_index.scroll(filters={"page": 1, "author": "Saiot-Exupery"})

print(len(result))
```

<pre class="custom">0
</pre>

Now you can delete 5 items using ```ids```.

```python
# delete item by ids
ids = uuids[1:6]

# call delete_node method
result = tutorial_index.delete(ids=ids)
print(result)
```

<pre class="custom">True
</pre>

```python
# Check if items are deleted
result = tutorial_index.scroll(ids=uuids[1:6])

print(len(result))
```

<pre class="custom">0
</pre>

## Similarity search
Since ```Neo4j``` supports a vector database, you can also do similarity search.

**Similarity** is calculated by the metric you set when you creating the index to search.

In this tutorial we will search items on ```tutorial_index``` , which use the **cosine** metric.

To do search, we call the ```search``` method.

```python
# do search. top_k is the number of documents in the result
res_with_text = tutorial_index.search(
    query="Does the little prince have a friend?", top_k=5
)

# print out top 2 results
print("RESULT BY RAW QUERY")
for i in range(2):
    print(res_with_text[i])
```

<pre class="custom">RESULT BY RAW QUERY
    {'text': '"My friend the fox--" the little prince said to me.', 'metadata': {'id': 'adf282b0-3efc-418c-8f9d-a48ae8052ba8', 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt', 'page': 1087, 'embedding': None}, 'score': 0.947}
    {'text': 'And the little prince asked himself:', 'metadata': {'id': 'd058a8a1-6440-4dff-837b-75976b71dc76', 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt', 'page': 492, 'embedding': None}, 'score': 0.946}
</pre>

That's all!

You now know the basics of using ```Neo4j```.

If you want to do more advanced tasks, please refer to the official ```Neo4j```  API documents and official Python SDK of ```Neo4j``` API documents.
