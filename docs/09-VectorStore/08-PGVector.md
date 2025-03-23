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

# PGVector

- Author: [Min-su Jung](https://github.com/effort-type), [Joonha Jeon](https://github.com/realjoonha), [Jongho Lee](https://github.com/XaviereKU)
- Design: 
- Peer Review : 
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/07-PGVector.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/09-VectorStore/07-PGVector.ipynb)

## Overview  

[```PGVector```](https://github.com/pgvector/pgvector) is an open-source extension for PostgreSQL that allows you to store and search vector data alongside your regular database information.

This notebook shows how to use functionality related to ```PGVector```, implementing LangChain vectorstore abstraction using postgres as the backend and utilizing the pgvector extension.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [What is PGVector?](#what-is-pgvector)
- [Initialization](#initialization)
- [Manage vector store](#manage-vector-store)
- [Similarity search](#similarity-search)

### References

- [langchain-postgres](https://github.com/langchain-ai/langchain-postgres/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Install pgvector on Windows](https://dev.to/mehmetakar/install-pgvector-on-windows-6gl)
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
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_openai",
        "pgvector",
        "psycopg",
        "psycopg-binary",
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
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "PGVector",
        "OPENAI_API_KEY": "",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## What is PGVector?

`PGVector` is a ```PostgreSQL``` extension that enables vector similarity search directly within your ```PostgreSQL``` database, making it ideal for AI applications, semantic search, and recommendation systems.

This is particularly valuable for who already use ```PostgreSQL``` who want to add vector search capabilities without managing separate infrastructure or learning new query languages.

**Features** :
1. Native ```PostgreSQL``` integration with standard SQL queries
2. Multiple similarity search methods including L2, Inner Product, Cosine
3. Several indexing options including HNSW and IVFFlat
4. Support for up to 2,000 dimensions per vector
5. ACID compliance inherited from ```PostgreSQL```

**Advantages** :

1. Free and open-source
2. Easy integration with existing ```PostgreSQL``` databases
3. Full SQL functionality and transactional support
4. No additional infrastructure needed
5. Supports hybrid searches combining vector and traditional SQL queries

**Disadvantages** :
1. Performance limitations with very large datasets (billions of vectors)
2. Limited to single-node deployment
3. Memory-intensive for large vector dimensions
4. Requires manual optimization for best performance
5. Less specialized features compared to dedicated vector databases

### Set up PGVector

If you are using Windows and have installed postgresql for Windows, you are required to install **vector** extension for postgresql. The following may help [Install pgvector on Windows](https://dev.to/mehmetakar/install-pgvector-on-windows-6gl).

But in this tutorial, we will use ```Docker``` container. If you are using Mac or Windows, check [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/) or [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install).

If you are using ```Docker``` desktop, you can easily set up `PGVector` by running the following command that spins up a ```Docker``` container:

```bash
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
```

For more detailed instructions, please refer to [the official documentation](https://github.com/pgvector/pgvector) 

** [ NOTE ] **
* If you want to maintain the stored data even after container being deleted, you must mount volume like below:
```bash
docker run --name pgvector-container -v {/mount/path}:/var/lib/postgresql/data -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
```


## Initialization

If you are successfully running the pgvector container, you can use ```pgVectorIndexManager``` from ```pgvector_interface``` in utils directory to handle collections.

To initialize ```pgVectorIndexManager``` you can pass full connection string or pass each parameter separately.

```python
from utils.pgvector_interface import pgVectorIndexManager

# Setup connection infomation
conn_str = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Initialize pgVectorIndexManaer
index_manager = pgVectorIndexManager(connection=conn_str)
```

When you initialize ```pgVectorIndexManager```, the procedure will automatically create two tables
**langchain_pg_collection** and **langchain_pg_embedding.**

* langchain_pg_collection
    * Stores **names** of the collections.
    * Distinguish collection by uuid and name.
* langchain_pg_embedding
    * Stores actual data.
    
So, when you create a new collection and insert data to the collection, the data will be stored in **langchain_pg_embedding** table.

As you can see below, the uuid column in langchain_pg_collection table matched with collection_id column in langchain_pg_embedding table.

![pgVector Entity Relation](./img/08-pgvector-entityRelation.png)


![pgVector Collection](./img/08-pgvector-collection.png)


![pgVector Data](./img/08-pgvector-data.png)

## Create collection
Now we can create collection with ```index_manager```.

To create collection, you need to pass **embedding** model and **collection_name** when calling the ```create_index``` method.

In this tutorial we will use ```text-embedding-3-large``` model of OpenAI.

If creation is successful, the method will return ```pgVectorDocumentManager``` class that can handle actual data.

In this tutorial we will create an collection with name **langchain_opentutorial.**

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

```python
# create new collection
col_manager = index_manager.create_index(
    collection_name="langchain_opentutorial", embedding=embeddings
)
```

### List collections

As we have created a new collection, we will call the ```list_indexes``` method to check if the collection is created.

```python
# check collections
indexes = index_manager.list_indexes()
print(indexes)
```

<pre class="custom">['langchain_opentutorial']
</pre>

### Delete collections

We can also delete collection by calling the ```delete_index``` method by pass the name of the collection to delete.

We delete **langchain_opentutorial** collection, and then create it again.

```python
# delete collection
index_manager.delete_index("langchain_opentutorial")

# check collections
indexes = index_manager.list_indexes()
print(indexes)

# Create again
col_manager_tmp1 = index_manager.create_index(
    collection_name="langchain_opentutorial", embedding=embeddings
)
```

<pre class="custom">[]
</pre>

## Get collection
As we said, when you create a new collection by calling the ```create_index``` method, this will automatically return ```pgVectorDocumentManager``` instance.

But if you want to re-use already created collection, you can call the ```get_index``` method with name of the collection and embedding model you used to create the collection to get manager.

```python
# Get collection
col_manager_tmp2 = index_manager.get_index(
    embedding=embeddings, collection_name="langchain_opentutorial"
)
```

## Manage vector store

Once you have created your vector store, we can interact with it by adding and deleting different items.

### Filtering
The ```pgVector``` support following filtering operations.

| Operator | Meaning/Category        |
|----------|-------------------------|
| \$eq      | Equality (==)           |
| \$ne      | Inequality (!=)         |
| \$lt      | Less than (&lt;)           |
| \$lte     | Less than or equal (&lt;=) |
| \$gt      | Greater than (>)        |
| \$gte     | Greater than or equal (>=) |
| \$in      | Special Cased (in)      |
| \$nin     | Special Cased (not in)  |
| \$between | Special Cased (between) |
| \$like    | Text (like)             |
| \$ilike   | Text (case-insensitive like) |
| \$and     | Logical (and)           |
| \$or      | Logical (or)            |

Filter can be used with ```scroll```, ```delete```, and ```search``` methods.

To apply filter, we create a dictionary and pass it to ```filter``` parameter like the following
```python
{"page": {"$between": [10,20]}}
```

### Connect to index
To add, delete, search items, we need to initialize an object which connected to the index we operate on.

We will connect to **langchain_opentutorial** . Recall that we used basic ```OpenAIEmbedding``` as a embedding function, and thus we need to pass it when we initialize ```index_manager``` object.

Remember that we also can get ```pgVectorDocumentManager``` object when we create an index with ```pgVectorIndexManager``` object or ```pgVectorIndexManager.get_index``` method, but this time we call it directly to get an ```pgVectorDocumentManager``` object.

```python
from utils.pgvector_interface import pgVectorDocumentManager

# Get document manager
col_manager = pgVectorDocumentManager(
    embedding=embeddings,
    connection_info=conn_str,
    collection_name="langchain_opentutorial",
)
```

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
from uuid import uuid4

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
        doc.metadata.update({"page": idx + 1, "id": str(uuid4())})
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
    Written By Antoine de Saiot-Exupery (1900〜1944)' metadata={'source': 'the_little_prince.txt', 'page': 1, 'author': 'Saiot-Exupery', 'id': 'cc23e228-2540-4e5c-8eb3-be6df7a3bf77'}
</pre>

### Add items to vector store

We can add items to our vector store by using the ```upsert``` or ```upsert_parallel``` method.

If you pass ids along with documents, then ids will be used, but if you do not pass ids, it will be created based `page_content` using md5 hash function.

Basically, ```upsert``` and ```upsert_parallel``` methods do upsert not insert, based on **id** of the item.

So if you provided id and want to update data, you must provide the same id that you provided at first upsertion.

We will upsert data to collection, **langchain_opentutorial** , with ```upsert``` method for the first half, and with ```upsert_parallel``` for the second half.

```python
# Gather uuids, texts, metadatas
uuids = [doc.metadata["id"] for doc in processed_docs]
texts = [doc.page_content for doc in processed_docs]
metadatas = [doc.metadata for doc in processed_docs]

# Get total number of documents
total_number = len(processed_docs)
print("Number of documents:", total_number)
```

<pre class="custom">Number of documents: 1359
</pre>

```python
%%time
# upsert documents
upsert_result = col_manager.upsert(
    
    texts=texts[:total_number//2], metadatas=metadatas[:total_number//2], ids=uuids[:total_number//2]
)
```

<pre class="custom">CPU times: user 1.57 s, sys: 140 ms, total: 1.71 s
    Wall time: 5.46 s
</pre>

```python
%%time
# upsert documents parallel
upsert_parallel_result = col_manager.upsert_parallel(
    texts = texts[total_number//2 :],
    metadatas = metadatas[total_number//2:],
    ids = uuids[total_number//2:],
    batch_size=32,
    max_workers=8
)
```

<pre class="custom">CPU times: user 1.79 s, sys: 82.9 ms, total: 1.88 s
    Wall time: 4.96 s
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

**[ NOTE ]**

As we have only one table, **langchain_pg_embedding** to store data, we have only one column **cmetadata** to store metadata for each document.

The **cmetadata** column is jsonb type, and thus if you want to update the metadata, you should provide not only the new metadata key-value you want to update, but with all the metadata already stored.

### Scroll items from vector store
As we have added some items to our first vector store, named **langchain_opentutorial** , we can scroll items from the vector store.

This can be done by calling ```scroll``` method.

When we scroll items from the vector store we can pass ```ids``` or ```filter``` to get items that we want, or just call ```scroll``` to get ```k```(*default 10*) items.

We can get embedded vector values of each items by set ```include_embedding``` True.

```python
# Do scroll without ids or filter
scroll_result = col_manager.scroll()

# print the number of items scrolled and first item that returned.
print(f"Number of items scrolled: {len(scroll_result)}")
print(scroll_result[0])
```

<pre class="custom">Number of items scrolled: 10
    {'content': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', 'metadata': {'id': 'cc23e228-2540-4e5c-8eb3-be6df7a3bf77', 'page': 1, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None}
</pre>

```python
# Do scroll with filter
scroll_result = col_manager.scroll(filter={"page": {"$in": [1, 2, 3]}})

# print the number of items scrolled and all items that returned.
print(f"Number of items scrolled: {len(scroll_result)}")
for r in scroll_result:
    print(r)
```

<pre class="custom">Number of items scrolled: 3
    {'content': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', 'metadata': {'id': 'cc23e228-2540-4e5c-8eb3-be6df7a3bf77', 'page': 1, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None}
    {'content': '[ Antoine de Saiot-Exupery ]', 'metadata': {'id': 'd4bf8981-2af4-4288-8aaf-6586381973c4', 'page': 2, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None}
    {'content': 'Over the past century, the thrill of flying has inspired some to perform remarkable feats of', 'metadata': {'id': '31dc52cf-530b-449c-a3db-ec64d9e1a10c', 'page': 3, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None}
</pre>

```python
# Do scroll with ids
scroll_result = col_manager.scroll(ids=uuids[:3])

# print the number of items scrolled and all items that returned.
print(f"Number of items scrolled: {len(scroll_result)}")
for r in scroll_result:
    print(r)
```

<pre class="custom">Number of items scrolled: 3
    {'content': 'The Little Prince\nWritten By Antoine de Saiot-Exupery (1900〜1944)', 'metadata': {'id': 'cc23e228-2540-4e5c-8eb3-be6df7a3bf77', 'page': 1, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None}
    {'content': '[ Antoine de Saiot-Exupery ]', 'metadata': {'id': 'd4bf8981-2af4-4288-8aaf-6586381973c4', 'page': 2, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None}
    {'content': 'Over the past century, the thrill of flying has inspired some to perform remarkable feats of', 'metadata': {'id': '31dc52cf-530b-449c-a3db-ec64d9e1a10c', 'page': 3, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None}
</pre>

### Delete items from vector store

We can delete items by filter or ids with ```delete``` method.


For example, we will delete **the first page**, that is ```page``` 1, of the little prince, and try to scroll it.

```python
# delete an item
col_manager.delete(filter={"page": {"$eq": 1}})

# check if it remains in DB.
print(col_manager.scroll(filter={"page": {"$eq": 1}}))
```

<pre class="custom">Delete done successfully
    []
</pre>

Now we delete 5 items using ```ids```.

```python
# delete item by ids
ids = uuids[1:6]

# call delete_node method
col_manager.delete(ids=ids)

# check if it remains in DB.
print(col_manager.scroll(ids=ids))
```

<pre class="custom">Delete done successfully
    []
</pre>

## Similarity search

As a vector store, ```pgVector``` support similarity search with various distance metric, **l2** , **inner** (max inner product), **cosine** .

By default, distance strategy is set to **cosine.** 

Similarity search can be done by calling the ```search``` method.

You can set the number of retrieved documents by passing ```k```(*default to 4*).

```python
results = col_manager.search(query="Does the little prince have a friend?", k=5)
for doc in results:
    print(doc)
```

<pre class="custom">{'content': '"My friend the fox--" the little prince said to me.', 'metadata': {'id': 'b02aaaa0-9352-403a-8924-cfff4973b926', 'page': 1087, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.631413271508214}
    {'content': '"No," said the little prince. "I am looking for friends. What does that mean-- ‘tame‘?"', 'metadata': {'id': '48adae15-36ba-4384-8762-0ef3f0ac33a3', 'page': 958, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.6050397117589812}
    {'content': 'the little prince returns to his planet', 'metadata': {'id': '4ed37f54-5619-4fc9-912b-4a37fb5a5625', 'page': 1202, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.5846221199406966}
    {'content': 'midst of the Sahara where he meets a tiny prince from another world traveling the universe in order', 'metadata': {'id': '28b44d4b-cf4e-4cb9-983b-7fb3ec735609', 'page': 25, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.5682375512406654}
    {'content': '[ Chapter 2 ]\n- the narrator crashes in the desert and makes the acquaintance of the little prince', 'metadata': {'id': '2a4e0184-bc2c-4558-8eaa-63a1a13da3a0', 'page': 85, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.555493427632688}
</pre>

### Similarity search with filters

You can also do similarity search with filter as we have done in ```scroll``` or ```delete```.

```python
# search with filter
result_with_filter = col_manager.search(
    "Does the little prince have a friend?",
    filter={"page": {"$between": [100, 110]}},
    k=5,
)

for doc in result_with_filter:
    print(doc)
```

<pre class="custom">{'content': 'inhabited region. And yet my little man seemed neither to be straying uncertainly among the sands,', 'metadata': {'id': '1be69712-f0f4-4728-b6f2-d4cf12cddfdb', 'page': 107, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.23158187113240447}
    {'content': 'Nothing about him gave any suggestion of a child lost in the middle of the desert, a thousand miles', 'metadata': {'id': 'df4ece8c-dcb6-400e-9d8e-0eb5820a5c4e', 'page': 109, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.18018012822748797}
    {'content': 'among the sands, nor to be fainting from fatigue or hunger or thirst or fear. Nothing about him', 'metadata': {'id': '71b4297c-3b76-43cb-be6a-afca5f59388d', 'page': 108, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.17715921622781305}
    {'content': 'less charming than its model.', 'metadata': {'id': '507267bc-7076-42f7-ad7c-ed1f835663f2', 'page': 100, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.16131896837723747}
    {'content': 'a thousand miles from any human habitation. When at last I was able to speak, I said to him:', 'metadata': {'id': '524af6ff-1370-4c20-ad94-1b37e45fe0c5', 'page': 110, 'author': 'Saiot-Exupery', 'source': 'the_little_prince.txt'}, 'embedding': None, 'score': 0.15769872390077566}
</pre>
