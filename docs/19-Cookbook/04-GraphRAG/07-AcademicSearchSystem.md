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

# Academic Search System

- Author: [Heeah Kim](https://github.com/yellowGangneng)
- Peer Review: [Yongdam Kim](https://github.com/dancing-with-coffee), [syshin0116](https://github.com/syshin0116)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/07-AcademicSearchSystem.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/07-AcademicSearchSystem.ipynb)

## Overview

This tutorial involves loading an open academic publication dataset called *OpenAlex* into a Graph DB named *Neo4J*.

Then, utilizing an LLM to generate <U>Cypher queries</U>, which are essentially queries for the Graph DB,
and using the data obtained from these Cypher queries to produce appropriate answers to inquiries,
we will build an *Academic Search System*.

Before we dive into the tutorial, let's understand what **GraphRAG** is and why we should use it!

**GraphRAG** refers to the RAG we already know well, but extended to include <U>not only vectors but also a **knowledge graph** in the search path.</U>

But what are the advantages of using this **GraphRAG** that we need to explore?
The reasons are as follows.

1. You can obtain more accurate and higher quality results.
    - According to Microsoft, using **GraphRAG** allowed them to obtain more relevant contexts, which led to better answers. It also made it easier to trace the grounds for those answers. 
    - Additionally, it required 26~97% fewer tokens, resulting in cost savings and scalability benefits.

2. It enhances data comprehension.
    - When looking at vectors represented by numerous numbers, it is nearly impossible for a human to conceptually and intuitively understand them.
    <br><center><img src='./assets/07-academic-search-system-01.png' alt='vector data' style="width:35%; height:35%"></center>
      <center style="color:gray">It seems impossible to understand...</center>
    <br>However, graphs are highly intuitive. They make it much easier to understand the relationships between data.
    <br><center><img src='./assets/07-academic-search-system-02.png' alt='graph data' style="width:50%; height:50%"></center>
      <center style="color:gray">It looks much clearer now.</center>
    <br>By exploring such intuitive graphs, you can gain new insights.

3. Management becomes easier in terms of tracking, explaining, and access control.
    - Using graphs, you can trace why certain data was selected or why errors occurred. This traceability can be used to explain the results.
    - Additionally, you can assign data permissions within the knowledge graph, enhancing security and privacy protection.

Knowing what **GraphRAG** is makes you want to use it even more, doesn't it?
Now, let's embark on creating an **Academic Search System** together!

### Table of Contents

- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Preliminary Task: Running Neo4j Using Docker](#preliminary-task-running-neo4j-using-docker)
- [Prepare the Data](#prepare-the-data)
- [Let's make the Academic Search System](#lets-make-the-academic-search-system)

### References

- [Create a Neo4j GraphRAG Workflow Using LangChain and LangGraph](https://neo4j.com/developer-blog/neo4j-graphrag-workflow-langchain-langgraph/)
- [The GraphRAG Manifesto: Adding Knowledge to GenAI](https://neo4j.com/blog/graphrag-manifesto/)
- [Graph-Based-Literature-Review-Tool](https://github.com/vtmike2015/Graph-Based-Literature-Review-Tool/tree/main)
- [Neo4j](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
- [OpenAlex](https://docs.openalex.org/)
- [GraphRAG : Neo4j DB와 LangChain 결합을 통한 질의응답 구현하기 (Kaggle CSV 데이터 적용하기)](https://uoahvu.tistory.com/entry/GraphRAG-Neo4j-DB%EC%99%80-LangChain-%EA%B2%B0%ED%95%A9%EC%9D%84-%ED%86%B5%ED%95%9C-%EC%A7%88%EC%9D%98%EC%9D%91%EB%8B%B5-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-Kaggle-CSV-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0)
----

<div style="color:gray">
Of course, Graph RAG does not come without its disadvantages.

1. It is quite challenging to construct.
2. It can be inefficient when dealing with unstructured data.
3. etc ...

Therefore, one must exercise caution when applying it in a production environment.

However, in this tutorial, we will focus solely on the topic of **Academic Search System**.
</div>

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
        "langchain-neo4j",
        "langchain",
        "langchain_openai",
        "langchain_core",
        "langgraph",
        "pyalex",
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
        "LANGCHAIN_PROJECT": "Academic Search System",
        "NEO4J_USERNAME": "",
        "NEO4J_PASSWORD": "",
        "NEO4J_URI": "",
    }
)
```

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```

## Preliminary Task: Running Neo4j Using Docker

Before we get into the main tutorial, we need to perform some pre-tasks.<br>
Specifically, we need to launch the Graph DB, Neo4j, using Docker!

Since our goal is not to study Docker, we will skip the detailed explanation about Docker and share the Docker Compose code declared to launch the Neo4j container.
Please modify it according to your environment!

[Official Site : Getting started with Neo4j in Docker](https://neo4j.com/docs/operations-manual/current/docker/introduction/)

*docker-compose.yml*

```yaml
services:
  neo4j:
    container_name: neo4j-boot
    image: neo4j:5.22.0
    ports:
      - 7474:7474	# for browser console
      - 7687:7687	# for db
    volumes:
      - {your volume path}:/data
      - {your volume path}:/conf
      - {your volume path}:/plugins
      - {your volume path}:/logs # These files specify the volumes to maintain the basic Neo4j configuration and data.
      # We will convert and save the OpenAlex data into JSON format within {your volume path}/json_data,
      # and then load the data into our Neo4j database through the /import folder, which is mounted with this path.
      - {your volume path}:/import 
    environment:
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_use__neo4j__config=true
      - NEO4J_AUTH={your admin ID}/{your admin PW}
      - NEO4J_PLUGINS=["apoc"]
```

After launching the container using the above files (or in your own unique way), 

**https://localhost:7474**

if you access the local server on port 7474,
<br><center><img src='./assets/07-academic-search-system-03.png' alt='vector data' style="width:50%; height:50%"></center>
      <center style="color:gray">neo4j Browser Console</center>

Ta-da! You will be able to see the following Neo4j screen, 
<br>and you will be ready to fully enjoy the tutorial.

Shall we dive into the main content now?

## Prepare the Data

Let's prepare the Data. As mentioned earlier, we will use the **OpenAlex**, an open academic publication dataset.
OpenAlex data describes academic entities and how these entities are interconnected. The data properties provided include:
- `Works`
- `Authors`
- `Scores`
- `Institutions`
- `Topics`
- `Publishers`
- `Funders`

Among these, we will focus on handling the following data properties: `Works`, `Authors`, `Institutions`, `Topics`

### Data Structure

Before we look at the structure of the data we will create, let's briefly understand what Nodes and Relationships in a GraphDB are.

GraphDB is composed of **Nodes** and **Relationships**.

- **Node**: Refers to an individual entity. A node can have zero or more labels that define it.
- **Relationship**: Refers to the connection between a source node and a target node. Relationships always have a direction and a type.

Both nodes and relationships can have key-value pair properties.

For more detailed information about Graph properties, please refer to the [official website](https://neo4j.com/docs/getting-started/graph-database/).

Let us now explore the nodes and relationships of the data we will construct.

**Node**
- `Works`: These are academic documents such as journal articles, books, datasets, and theses.
    - `display_name`: The title of the academic document.
    - `cited_by_count`: The number of times the document has been cited.
    - `language`: The language in which the document is written.
    - `publication_year`: The year the document was published.
    - `type`: The type of the document.
    - `license`: The license under which the document is published.
    - `url`: The URL where the document is available.
<br>
<br>
- `Authors`: Information about the authors who wrote the academic documents.
  - `display_name`: The name of the author.
  - `orcid`: The author's ORCID. (ORCID is a global and unique ID for authors.)
  - `works_count`: The number of documents the author has worked on.
<br>
<br>
- `Topics`: Subjects related to the documents.
  - `display_name`: The title of the topic.
  - `description`: A description of the topic.
<br>
<br>
- `Institutions`: The institutions to which the authors were affiliated. It is included in the Authors data.
  - `display_name`: The name of the institution.
  - `ror`: The ROR (Research Organization Registry) ID of the institution.
  - `country_code`: The country where the institution is located.

**Relationship**
- `Works` <- `WROTE` - `Authors`: The relationship between an author and the documents they have written.
  - `author_position` The author's position in the authorship list (e.g., first author, second author).

- `Works` - `ASSOCIATION` -> `Topics`: The relationship between documents and topics.
  - `score` The relevance score indicating how strongly the document is related to the topic.

- `Authors` - `AFFILIATED` -> `Institutions`: The relationship between authors and the institutions with which they were affiliated.
  - `years` The years during which the author was affiliated with the institution.

<br>For more detailed information about **OpenAlex** Entities, please refer to the [official website](https://docs.openalex.org/api-entities/entities-overview).

The above structure utilizes only a small portion of the data, and you could certainly develop a more logical 
and comprehensive structure for nodes and relationships based on your own rationale.

However, for the purposes of this tutorial, we will proceed with the structure as outlined above.

```python
from pyalex import Works, Authors, Topics, Institutions
import os

json_path = "{your volume path}"  # "The local folder path to be mounted with /imports when running the Neo4j container
os.makedirs(json_path, exist_ok=True)
```

```python
from neo4j import GraphDatabase

# Connect with Neo4j
url = os.environ["NEO4J_URI"]
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(url, auth=(username, password))
```

```python
import json


# Storing json data extracted from OpenAlex
# name : display_name of each entity
# entity : Works / Authors / Institutions / Topics
# page : JSON formatted data obtained through pyAlex
def make_file(name, entity, page):
    file = os.path.join(json_path, f"{name.replace(' ', '_')}_{entity}.json")
    out_file = open(file, "w")
    json.dump(page, out_file, indent=6)

    out_file.close()

    print("Now Downloading " + file)
```

```python
concept_id = (
    "C154945302"  # The Concept ID of Artificial Intelligence scholarly literature data
)

# Extract Works data
works_pager = (
    Works().filter(concept={"id": {concept_id}}).paginate(per_page=1, n_max=10)
)
for works_page in works_pager:
    try:
        make_file(works_page[0]["display_name"], "work", works_page[0])
        # Extract Topics data related to Works
        topics = works_page[0]["topics"]
        for topic in topics:
            try:
                topic_pager = Topics().filter(
                    display_name={"search": topic["display_name"]}
                )
                make_file(topic["display_name"], "topic", topic_pager.get()[0])
            except Exception as e:
                print(
                    "An error occurred while creating the ",
                    topic["display_name"],
                    " file. : ",
                    e,
                )

        # Extract Authors data related to Works
        for authorships in works_page[0]["authorships"]:
            try:
                author_pager = Authors().filter(
                    display_name={"search": {authorships["author"]["display_name"]}}
                )
                make_file(
                    authorships["author"]["display_name"],
                    "author",
                    author_pager.get()[0],
                )

            except Exception as e:
                print(
                    "An error occurred while creating the ",
                    authorships["author"]["display_name"],
                    " file. : ",
                    e,
                )
    except Exception as e:
        print(
            "An error occurred while creating the ",
            works_page[0]["display_name"],
            " file. : ",
            e,
        )
```

Let's create a graph based on the JSON file we downloaded.

```python
from glob import glob

# Retrieve the list of downloaded JSON files per node
# GSince the data to be used in the graph is accessed through the /imports folder within the container, local paths are excluded.
work_files = [i.split("\\")[-1] for i in glob(json_path + "*_work.json")]
author_files = [i.split("\\")[-1] for i in glob(json_path + "*_author.json")]
institution_files = [i.split("\\")[-1] for i in glob(json_path + "*_institution.json")]
topic_files = [i.split("\\")[-1] for i in glob(json_path + "*_topic.json")]
```

Now let's build a graph using Cypher. Before that, let me briefly talk about Cypher.
<br>Cypher is Neo4j’s declarative query language, allowing users to unlock the full potential of property graph databases.
<br>For more detailed information about Neo4j Cyphers, please refer to the [official website](https://neo4j.com/docs/cypher-manual/current/introduction/).

As always, a single line of code is often easier to understand than ten lines of explanation. 
<br>Let's use Cypher to insert JSON data.

Let's analyze the Cypher declared above, line by line. We will omit explanations for duplicated forms of code.

- `CALL apoc.load.json('"+ file+ "')`
  - Read the JSON files. At this time, the **APOC** module is required. In the case of the docker compose provided above, it will be automatically installed through `NEO4J_PLUGINS=['apoc']`.
<br>
<br>
- `YIELD value`
  - Returns the `value` obtained by reading the JSON file.
<br>
<br>
- `UNWIND value.authorships as authorships`
  - By separating the `authorships` list within the `value` object, each item is individually processed as `authorships` objects with an alias assigned through `as`.
<br>
<br>
- `WITH value, authorships, author, topics, field, domain`
  - Variables obtained through `YIELD` and `UNWIND` are passed to the next part of the query, making them available for subsequent operations.
<br>
<br>
- `MERGE (w:Works {id: value.id}) \
  SET w.display_name = coalesce(value.display_name, '')\
  ...`
  - The `MERGE` clause is used to match or create a node with the `Works` label that has a unique `id` property matching `value.id`. If a node with the corresponding `id` already exists, it matches that node; otherwise, it creates a new node.
  - The `SET` clause updates the `display_name` property of the `Works` node. The `coalesce` function ensures that `value.display_name` is replaced with an empty string (`''`) if it is `null`.
<br>
<br>
- `MERGE (a)-[:WROTE{author_position: authorships.author_position}]->(w)`
    - The `MERGE` clause is used to match or create a relationship between nodes `a` and `w`. Just like with nodes, if the same relationship already exists, it matches or creates it, and if it does not exist, it creates a new relationship.
    - This relationship has the `WROTE` label and includes the `author_position` property. This property is set to the value of `authorships.author_position`.

The **nodes** and **relationships** for `Authors` and `Topics` will be constructed in a similar manner, so the explanation will be omitted.

```python
for file in work_files:
    print("File being imported: " + file)
    work_node_creation = (
        "CALL apoc.load.json('"
        + file
        + "') \
        YIELD value \
        UNWIND value.authorships as authorships \
        UNWIND authorships.author as author \
        UNWIND value.topics as topics \
        UNWIND topics.field as field \
        UNWIND topics.domain as domain \
        WITH value, authorships, author, topics, field, domain \
        MERGE (w:Works {id: value.id}) \
        SET w.display_name = coalesce(value.display_name, ''), \
        w.cited_by_count = coalesce(value.cited_by_count, ''), \
        w.is_paratext = coalesce(value.is_paratext, ''), \
        w.language = coalesce(value.language, ''), \
        w.publication_year = coalesce(value.publication_year, ''), \
        w.type = coalesce(value.type, ''), \
        w.license = coalesce(value.license, ''), \
        w.url = coalesce(value.url, '')\
        MERGE (a:Authors {id: author.id})\
        SET a.display_name = coalesce(author.display_name, ''),\
        a.orcid = coalesce(author.orcid, '')\
        MERGE (a)-[:WROTE{author_position: authorships.author_position}]->(w)\
        MERGE (t:Topics {id:topics.id}) \
        SET t.display_name = coalesce(topics.display_name, '')\
        MERGE (w)-[:ASSOCIATION{score: topics.score}]->(t)"
    )
    driver.execute_query(work_node_creation)
    print("File - " + file + " import complete")

print("All works imported")
```

```python
for file in author_files:
    print("File being imported: " + file)
    author_node_creation = (
        "CALL apoc.load.json('"
        + file
        + "') \
        YIELD value \
        UNWIND value.affiliations as affiliations \
        UNWIND affiliations.institution as institution \
        UNWIND affiliations.years as years \
        WITH value, affiliations, institution, years \
        MERGE (a:Authors {id: value.id})\
        SET a.display_name = coalesce(value.display_name, ''),\
        a.orcid = coalesce(value.orcid, ''),\
        a.works_count = coalesce(value.works_count, '')\
        MERGE (i:Institutions {id:institution.id}) \
        SET i.display_name = coalesce(institution.display_name, ''), \
        i.ror = coalesce(institution.ror, ''), \
        i.country_code = coalesce(institution.country_code, '') \
        FOREACH (year IN years |\
        MERGE (a)-[:AFFILIATED{year: years}]->(i))"
    )
    driver.execute_query(author_node_creation)
    print("File - " + file + " import complete")

print("All authors imported")
```

```python
for file in topic_files:
    print("File being imported: " + file)
    topic_node_creation = (
        "CALL apoc.load.json('"
        + file
        + "') \
        YIELD value \
        UNWIND value.field as field \
        UNWIND value.domain as domain \
        WITH value, field, domain \
        MERGE (t:Topics {id: value.id})\
        SET t.display_name = coalesce(value.display_name, '')"
    )
    driver.execute_query(topic_node_creation)
    print("File - " + file + " import complete")

print("All topics imported")
```

The graph with all the data will have the following structure.

<br><center><img src='./assets/07-academic-search-system-04.png' alt='our graph' style="width:50%; height:50%"></center>
      <center style="color:gray">The graph we built!</center>

Now, let us integrate the generated graph with the LLM to build a Q&A system.

## Let's make the Academic Search System

### Using the default QA chain

First, let's make use of the default QA Chain provided by langchain.<br>
`GraphCypherQAChain` is a function that *generates Cypher queries* and *facilitates question-answering about graphs* 
<br>by having a pre-declared chain, making it convenient to use.

<br><center><img src='./assets/07-academic-search-system-05.png' alt='GraphCypherQAChain' style="width:50%; height:50%"></center>
      <center style="color:gray">source : [Langchain](https://python.langchain.com/v0.1/docs/use_cases/graph/)</center>

As can be seen from the above picture, the model operates the LLM once to generate a Cypher query, 
<br>then runs the GraphDB with the generated query, and operates the LLM once again to generate an appropriate response 
<br>to the user's query based on the executed results.

Let's implement a simple QA service using the `GraphCypherQAChain` function.

```python
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
```

```python
# Declaring graphs and LLM models

graph = Neo4jGraph(
    os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)
```

```python
# (Optional) Declaration of Prompts for Generating Cypher Queries and for QA

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)
```

```python
CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.

If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""
CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)
```

```python
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    qa_prompt=CYPHER_QA_PROMPT,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
)
```

```python
chain.invoke("Who is the author with the most publications?")
```

<pre class="custom">
    
    > Entering new GraphCypherQAChain chain...
    Generated Cypher:
    cypher
    MATCH (a:Authors)-[:WROTE]->(w:Works)
    RETURN a.display_name, COUNT(w) AS publication_count
    ORDER BY publication_count DESC
    LIMIT 1
    
    Full Context:
    [{'a.display_name': 'Geoffrey E. Hinton', 'publication_count': 2}]
    
    > Finished chain.
</pre>




    {'query': 'Who is the author with the most publications?',
     'result': 'The author with the most publications is Geoffrey E. Hinton, who has a total of 2 publications.'}



**However**, there is one issue with this function, 
<br>which is that it directly inserts the information from the query into the Cypher.

In other words, <U>you need to have precise information about the data to get the desired answer.</U>

```python
chain.invoke("What literature is available related to CNN?")
```

<pre class="custom">
    
    > Entering new GraphCypherQAChain chain...
    Generated Cypher:
    cypher
    MATCH (t:Topics {display_name: 'CNN'})<-[:ASSOCIATION]-(w:Works)
    RETURN w
    
    Full Context:
    []
    
    > Finished chain.
</pre>




    {'query': 'What literature is available related to CNN?',
     'result': "I'm sorry, but I don't know the answer."}



Therefore, instead of relying on the predefined QA chain, 
<br>we should create our own custom chain using **LangGraph**.

### Using the LangGraph chain we built 

Upon receiving a query, if it pertains to a node or the relationships between nodes, <br>
we plan to first extract related data by performing a semantic search using Embedding Vectors of specific properties. <br>
Then, we will construct a graph to utilize the extracted data for the query.

```python
from langchain.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated
from typing_extensions import TypedDict
```

```python
# Define State


class State(TypedDict):
    messages: Annotated[list, add_messages]
```

```python
# Define Tools


@tool
def node_vector_search_tool(
    node_label: Annotated[
        str,
        "This is the node label that requires vector search (one of Works, Authors, Institutions, Topics).",
    ],
    query: Annotated[str, "User's query."],
):
    """
    Retrieve the node information most similar to the user's query.

    Input:
    - node_label: The label of the node. The types include Works, Authors, Institutions, and Topics.
      - Works: Node containing information about academic literature.
      - Authors: Node containing information about authors.
      - Institutions: Node containing information about institutions.
      - Topics: Node containing information about topics related to the literature.
    - query: The user's query.

    Output:
    - The node with the highest similarity is returned.
    """
    node_vector_index = Neo4jVector.from_existing_graph(
        embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        index_name="name_vector",
        node_label=node_label,
        text_node_properties=(
            ["display_name"]
            if node_label != "Topics"
            else ["display_name", "description"]
        ),
        embedding_node_property="embedding_vectors",
    )

    result = node_vector_index.similarity_search(query, k=1)

    return result


@tool
def relationship_vector_search_tool(
    relationship_property: Annotated[
        str,
        "This is the relationship property that requires vector search (one of year, score, author_position).",
    ],
    query: Annotated[str, "User's query."],
):
    """
    Retrieve the relationship information most similar to the user's query.

    Input:
    - relationship_property: The property of the relationship. The types include year, score, and author_position.
      - year: Information about the relationship between Authors and Institutions, indicating the year an author was affiliated with an institution.
      - score: Information about the relationship between Works and Topics, indicating the percentage of their relevance.
      - author_position: Information about the relationship between Works and Authors, indicating the author's position (order) in the work.

    Output:
    - The relationship with the highest similarity is returned.
    """
    relationship_vector_index = Neo4jVector.from_existing_relationship_index(
        embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        text_node_property=relationship_property,
        embedding_node_property="embedding_vectors",
    )

    result = relationship_vector_index.similarity_search(query, k=1)

    return result


tools = [node_vector_search_tool, relationship_vector_search_tool]
tool_node = ToolNode(tools)
```

```python
# Let's build our graph!

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a question generation agent that generates questions to fetch the optimal results from a Graph DB.\
            If the user's question is about academic literature, authors, topics, or institutions, use a Tool to calculate similarity with Nodes.\
            If the user's question is about the relevance between academic literature and topics, \
            the relationship between academic literature and authors, or the affiliation between authors and institutions, \
            use a Tool to calculate similarity with Relationships to find the related data. \
            Then, append the result of the Tool call to the user's question and output the revised question. \
            If not, pass the user's question as is.\
            Do not add any additional comments and output only the query.",
        ),
        ("user", "{messages}"),
    ]
)

assistant = assistant_prompt | llm.bind_tools(tools)


def chatbot(state: State):
    messages = state["messages"]
    response = assistant.invoke(messages)
    return {"messages": [response]}


def cypherQA(state: State):
    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        qa_prompt=CYPHER_QA_PROMPT,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        allow_dangerous_requests=True,
    )
    response = chain.invoke(state["messages"][-1].content)
    return {"messages": [response["result"]]}


def route_tools(state: State):
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        return "cypherQA"
    return "tools"


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("cypherQA", cypherQA)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools, ["tools", "cypherQA"])
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("cypherQA", END)

app = graph_builder.compile()
```

```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Visualize the compiled StateGraph as a Mermaid diagram
display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
```


    
![png](./img/output_42_0.png)
    


```python
# Let's actually operate the graph.

user_query = "CNN Works"

events = app.stream(
    {"messages": [HumanMessage(content=user_query)]}, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()
```

<pre class="custom">================================ Human Message =================================
    
    CNN Works
    ================================== Ai Message ==================================
    Tool Calls:
      node_vector_search_tool (call_lslW14cJ6XyY26udbv42MGqX)
     Call ID: call_lslW14cJ6XyY26udbv42MGqX
      Args:
        node_label: Works
        query: CNN Works
    ================================= Tool Message =================================
    Name: node_vector_search_tool
    
    [Document(metadata={'title': 'ImageNet Classification with Deep Convolutional Neural Networks', 'cited_by_count': 67974, 'publication_year': 2012, 'language': 'en', 'license': '', 'is_paratext': False, 'type': 'article', 'url': ''}, page_content='\ndisplay_name: ImageNet Classification with Deep Convolutional Neural Networks')]
    ================================== Ai Message ==================================
    
    What are the details of the work titled "ImageNet Classification with Deep Convolutional Neural Networks"?
    
    
    > Entering new GraphCypherQAChain chain...
    Generated Cypher:
    cypher
    MATCH (w:Works {title: "ImageNet Classification with Deep Convolutional Neural Networks"})
    RETURN w
    
    Full Context:
    [{'w': {'embedding_vectors': [-0.01446049939841032, -0.001230438589118421, 0.009549254551529884, -0.007678637281060219, 0.020359598100185394, 0.01800556480884552, -0.0012532082619145513, 1.265742594114272e-05, -0.015315238386392593, 0.002375053009018302, 0.01193831954151392, 0.030826646834611893, 0.0029442950617522, 0.0023575378581881523, -0.0308826956897974, 0.023007888346910477, 0.018734194338321686, 0.014922899194061756, 0.01580566167831421, -0.032732293009757996, -0.008211097680032253, 0.01821574568748474, -0.005734456703066826, -0.025502044707536697, -0.008470322005450726, 0.010095726698637009, 0.017809394747018814, -0.03466596454381943, 0.010137762874364853, -0.011153641156852245, 0.02164871245622635, 0.00020941538969054818, 0.009226975962519646, -0.031275033950805664, 0.0047115725465118885, -0.009857520461082458, -0.005377147812396288, -0.018257781863212585, 0.025249825790524483, 0.016099916771054268, 0.03306858614087105, 0.009381108917295933, -0.016366146504878998, -0.02506766840815544, -0.012358683161437511, 0.010410998947918415, -0.009983629919588566, -0.005748468916863203, -0.0022944833617657423, 0.008190079592168331, 0.021704761311411858, 0.040326859802007675, -0.0059306262992322445, -0.030154066160321236, -0.01785143092274666, 0.015161105431616306, -0.02767392061650753, 0.010298902168869972, 0.0039023731369525194, -0.04077524691820145, 0.018888326361775398, -0.005874577909708023, -0.029565555974841118, 0.005899099167436361, -0.016352135688066483, -0.028066260740160942, 0.01925264298915863, 0.03909379243850708, 0.04004661738872528, -0.012765034101903439, 0.046436138451099396, 0.016632376238703728, 0.008897691033780575, -0.012526827864348888, 0.028752854093909264, -0.005461220629513264, -0.0003154914593324065, 0.01667441427707672, -0.009801472537219524, -0.004165100399404764, 0.011398852802813053, -0.015091044828295708, -0.010950465686619282, 0.004109052009880543, 0.011097593232989311, -0.011959337629377842, -0.016744473949074745, -0.0038638398982584476, 0.0019091502763330936, -0.007496479898691177, 0.059075064957141876, 0.008253133855760098, 0.01909850910305977, 0.0062879351899027824, -0.02737966738641262, 0.015875723212957382, -0.018397903069853783, 0.027575837448239326, -0.004480372648686171, -0.0032770826946944, 0.01268096175044775, 0.010705254040658474, -0.008148043416440487, -0.021928954869508743, -0.018846290186047554, -0.01654830388724804, -0.002438107505440712, 0.01835586689412594, 0.006494613830000162, -0.007013062015175819, -0.00953524187207222, 0.02950950898230076, 0.0030774101614952087, -0.03800084814429283, -0.01228161621838808, 0.0018058109562844038, 0.013122343458235264, -4.3267868022667244e-05, 0.01260389480739832, -0.010600162670016289, 0.005373645108193159, 0.014131215400993824, 0.009969618171453476, -0.018173709511756897, 0.009899557568132877, 0.011468913406133652, -0.0187482051551342, -0.03808492049574852, 0.018608085811138153, -0.011805204674601555, 0.029145194217562675, 0.016394171863794327, 0.01153897400945425, -0.022055065259337425, 0.0034592400770634413, 0.015945782884955406, 0.011209690012037754, 0.0018373382044956088, -0.020513731986284256, 0.02114427648484707, -0.016338123008608818, 0.03786072880029678, -0.0154413478448987, -0.00021773508342448622, -0.01345162745565176, 0.03242402896285057, -0.04102746397256851, 0.0098505150526762, -0.010859386995434761, 0.0012803567806258798, -0.021872907876968384, -0.015497395768761635, -0.008063970133662224, 0.0013407840160652995, -0.0005810647853650153, 0.03500225767493248, 0.006848420016467571, 0.011616040952503681, 0.0017121050041168928, 0.006133802235126495, 0.005664396565407515, 0.011139629408717155, -0.02477341517806053, -0.03144317865371704, 0.040410932153463364, 0.025936419144272804, 0.014614633284509182, -0.002031756332144141, -0.010375969111919403, 0.003457488724961877, 0.014894874766469002, 0.012652937322854996, -0.042961135506629944, 0.0021893924567848444, 0.017893467098474503, -0.010319920256733894, 0.025530068203806877, -0.023834602907299995, -0.03161132335662842, -0.01907048374414444, -0.00840726774185896, 0.012779045850038528, 0.006329971831291914, -0.007685643620789051, 0.0022559501230716705, 0.003376919077709317, 0.03332080319523811, -0.01646423153579235, 0.003940906375646591, -0.005976165644824505, -0.011027532629668713, 0.027786018326878548, 0.0004825421201530844, -0.005779996048659086, -0.6326748728752136, -0.03503027930855751, 0.01675848662853241, -0.021130265668034554, 0.006091765593737364, 0.025516055524349213, -0.013949058018624783, -0.01434139721095562, -0.011062562465667725, 0.04209238663315773, 0.002870731521397829, -0.01625405065715313, -0.03287241607904434, 0.0014371172292158008, -0.0011069568572565913, -0.030041968449950218, -0.0005307087558321655, -0.01102052628993988, -0.010305908508598804, -0.0019266654271632433, 0.0071496800519526005, 0.009051823988556862, -0.017164837568998337, 0.023218069225549698, -0.007118152920156717, -0.007405401207506657, -0.004392797127366066, 0.008141037076711655, 0.01893036440014839, 0.0077416920103132725, -0.04472666233778, 0.015665540471673012, 0.02515174075961113, -0.001524692983366549, 0.04915449023246765, -0.009787460789084435, -0.008694515563547611, 0.004974300041794777, -0.0002957869437523186, 0.06664160639047623, -0.03396536037325859, -0.012022391892969608, 0.0013223930727690458, 0.004750106018036604, -0.021816859021782875, 0.03135910630226135, 0.012800064869225025, 0.003238549456000328, 0.020962119102478027, -0.012652937322854996, 0.0037587489932775497, 0.005072384607046843, 0.008421279489994049, -0.0033348826691508293, 0.022475427016615868, -0.010256865993142128, 0.016394171863794327, -0.006974529009312391, 0.021760810166597366, 0.006102274637669325, -0.008540382608771324, 0.013409591279923916, -0.010873398743569851, -0.02687523141503334, -0.02127038687467575, 0.020836010575294495, -0.008533376269042492, -0.0019704531878232956, -0.008575412444770336, -0.016632376238703728, 0.007114649750292301, -0.0022034046705812216, 0.027743982151150703, -0.00416860356926918, 0.002995088929310441, 0.026328759267926216, 0.0023049924056977034, -0.01260389480739832, -0.001072802348062396, 0.019182581454515457, 0.011658077128231525, -0.010754295624792576, 0.02443712390959263, 0.008309182710945606, -0.006830904632806778, 0.006147813983261585, 0.00420363387092948, 0.002299737883731723, -0.012120476923882961, -0.013521688058972359, 0.029817774891853333, 0.007944867946207523, -0.007195219397544861, -0.03909379243850708, -0.023960711434483528, 0.024843474850058556, -0.017417054623365402, -0.00418261531740427, -0.0065541653893888, -0.007580552715808153, 0.013759894296526909, -0.010144769214093685, 0.0025204287376254797, 0.0005434072227217257, 0.014110197313129902, 0.013038270175457, -0.032227858901023865, 0.04486678168177605, 0.01975707896053791, -0.005713438615202904, -0.013374561443924904, -0.009135897271335125, -0.04399803280830383, 0.011735144071280956, 0.009584284387528896, -0.03096676804125309, 0.015315238386392593, 0.021704761311411858, 0.017515139654278755, 0.008148043416440487, 0.008967751637101173, -0.00886966660618782, 0.014600620605051517, 0.009647339582443237, 0.0026955800130963326, 0.01359875500202179, 0.005170469172298908, -0.024128858000040054, -0.017613224685192108, 0.008939727209508419, 0.021256374195218086, 0.017066752538084984, 0.020485708490014076, -0.007776722311973572, 0.029089145362377167, 0.010551120154559612, 0.024605268612504005, 0.009437157772481441, -0.04108351469039917, -0.01253383420407772, 0.0004707193875219673, 0.002362792380154133, 0.017991552129387856, 0.006228383630514145, -0.02862674556672573, 0.003989948891103268, -0.0014476263895630836, 0.018916351720690727, -0.005033851135522127, 0.013472645543515682, -0.015483384020626545, -0.013255458325147629, -0.0167024377733469, 0.018888326361775398, 0.005058372393250465, -0.033685117959976196, 0.00034439144656062126, -0.02865476906299591, 0.0008179570431821048, -0.02257351204752922, -0.008218104019761086, 0.020177440717816353, -0.016127940267324448, 0.00020733546989504248, -0.02127038687467575, -0.004067015368491411, -0.021942967548966408, 0.043353475630283356, -0.028836926445364952, -0.018972400575876236, 0.009290030226111412, -0.02494155988097191, -0.009388115257024765, 0.016240037977695465, 0.01675848662853241, 0.026300733909010887, -0.01673046126961708, -0.019644981250166893, 0.0043437546119093895, -0.0075174979865550995, 0.023274118080735207, 0.006456080824136734, -0.00336290686391294, -0.004599475767463446, 0.024787425994873047, 0.015861710533499718, 0.019406775012612343, 0.008834636770188808, -0.0310228168964386, 0.004417318385094404, -0.007419413421303034, 0.02233530767261982, 0.0012006628094241023, 0.013479651883244514, 0.015231166034936905, 0.04363371804356575, 0.01996725983917713, -0.018159696832299232, 0.009058830328285694, 0.025516055524349213, 0.008939727209508419, 0.014544572681188583, 0.019168568775057793, 0.006305450573563576, 0.02023348957300186, 0.0028532163705676794, 0.009304042905569077, -0.02366645820438862, -0.014404451474547386, 0.016842558979988098, 0.01705273985862732, 0.0063089532777667046, -0.012449761852622032, -0.00677135307341814, -0.0001980305532924831, 0.02013540454208851, -0.016576329246163368, -0.0027796528302133083, -0.03301253542304039, 0.030826646834611893, 0.021046193316578865, 0.005653887055814266, 0.014530560001730919, 0.0010167538421228528, -0.004988311789929867, -0.003657161258161068, 0.017837418243288994, 0.009233982302248478, 0.01067722961306572, -0.041924238204956055, 0.010389980860054493, 0.030714549124240875, 0.018313830718398094, -0.013521688058972359, -0.004855196923017502, 0.0004821042239200324, 0.02900507301092148, -0.005075887776911259, 0.020864034071564674, -0.014824815094470978, 0.0192246176302433, 0.018664132803678513, 0.016001831740140915, -0.021732786670327187, 0.025880370289087296, 0.0175571758300066, -0.0012645930983126163, 0.023624420166015625, -0.039430081844329834, -0.0009326811996288598, -0.0033821735996752977, 0.00017558927356731147, -0.022965852171182632, -0.0016937140608206391, 0.0026623012963682413, 0.006238893140107393, 0.0003277520590927452, 0.0167024377733469, 0.007601570803672075, 0.01978510245680809, 0.012666949070990086, 0.0022927317768335342, 0.030658502131700516, 0.005016336217522621, 0.021522603929042816, -0.021704761311411858, 0.004038991406559944, -0.02164871245622635, 0.00609526876360178, -0.023148009553551674, -0.017599212005734444, -0.024633293971419334, 0.00808498915284872, -0.026230674237012863, -0.007048092316836119, -0.007923849858343601, 0.006855425890535116, -0.004235161002725363, 0.014040136709809303, -0.00012479537690524012, -0.010621180757880211, -0.021284397691488266, 0.020934095606207848, 8.303271170007065e-05, 0.007657619193196297, -0.007293304428458214, -0.045987751334905624, 0.027239546179771423, -0.022475427016615868, 0.00159825652372092, 0.001186650712043047, 0.018117660656571388, 0.007181207649409771, -0.00028286949964240193, -0.019532883539795876, -0.004119561053812504, 0.03657161444425583, -0.008148043416440487, -0.009240987710654736, -0.011223701760172844, 0.0329284630715847, 0.015034995973110199, -0.0037377309054136276, -0.007664625532925129, 0.051452476531267166, -0.00115687504876405, -0.006508626043796539, 0.003734227968379855, -0.006995547097176313, -0.004144082311540842, 0.008197085931897163, 0.01444648765027523, -0.01752915233373642, -0.005559305660426617, 0.016240037977695465, 0.0008906449074856937, -0.022321294993162155, 0.0007308192434720695, 0.017220886424183846, 0.0003386990283615887, -0.005440202541649342, -0.015034995973110199, -0.01779538206756115, 0.0015316989738494158, 0.0007062980439513922, 0.04231657832860947, -0.004245670046657324, 0.003993452060967684, -0.03096676804125309, -0.008547388017177582, -0.03651556372642517, 0.001051784143783152, 0.005373645108193159, -0.010670223273336887, -0.034806087613105774, 0.01773933321237564, 0.010656211525201797, 0.010347944684326649, 0.006126795895397663, -0.004098542965948582, 0.012715991586446762, 0.007587558589875698, -0.002345277229323983, -0.015665540471673012, -0.004764118231832981, 0.0031544766388833523, 0.02597845532000065, -0.0099205756559968, 0.01726292259991169, -0.03965427726507187, 0.023302143439650536, 0.022937826812267303, 0.012400719337165356, -0.03284439072012901, -0.01654830388724804, 0.04052302986383438, 0.025586117058992386, -0.007461449597030878, -0.013073300942778587, 0.03665568679571152, 0.005839547608047724, 0.03853330761194229, 0.007139171008020639, -0.003413700731471181, 0.015791650861501694, 0.014908887445926666, 0.0077346861362457275, -0.009170927107334137, -0.005184481386095285, -0.018313830718398094, 0.006939498707652092, 0.05128433182835579, 0.0013425354845821857, -0.025431983172893524, 0.014782777987420559, 0.015133081004023552, -0.04764118045568466, -0.01963096857070923, -0.022027039900422096, -0.009065836668014526, -0.018902339041233063, -0.02254548855125904, -0.006904468405991793, -0.017220886424183846, -0.015665540471673012, -0.018580060452222824, 0.0029267799109220505, -0.0034347190521657467, -0.0018460957799106836, -0.0008845145930536091, -0.01277204044163227, -0.004991814959794283, -0.009381108917295933, 0.010859386995434761, -0.013353542424738407, -0.026538940146565437, -0.037356290966272354, -0.00016026353114284575, -0.004158094059675932, 0.008666491135954857, 0.015034995973110199, -0.020331574603915215, 0.0024731378071010113, 0.010074708610773087, 0.0035152886994183064, -0.026202648878097534, -0.018257781863212585, -0.039906494319438934, 0.005527778062969446, 0.0002699520846363157, 0.004553936421871185, -0.002551955869421363, -0.02257351204752922, 0.020163429901003838, 0.030041968449950218, -0.002555459039285779, 0.021844882518053055, -0.0014169748174026608, 0.01468469388782978, 0.010453036054968834, 0.015301226638257504, -0.016772497445344925, 0.01309431903064251, -0.01898641139268875, 0.018257781863212585, -0.010242854245007038, -0.00734234694391489, -0.004557439591735601, -0.0026868225540965796, 0.004525911994278431, -0.004347257781773806, 0.012036404572427273, -0.011658077128231525, 0.021942967548966408, 0.033292777836322784, 0.01036896277219057, -0.00968236941844225, -0.009843508712947369, 0.003888361155986786, 0.041896216571331024, -0.010523096658289433, 0.009913569316267967, -0.014327384531497955, 0.0026640528813004494, 0.013535700738430023, -0.0072582741267979145, 0.026258697733283043, -0.014530560001730919, -0.005363136064261198, 0.032732293009757996, -0.0030126040801405907, -0.041307706385850906, -0.03444177284836769, 0.012996233999729156, -0.010803338140249252, 0.026538940146565437, -0.02257351204752922, -0.04245670139789581, -0.009044818580150604, -0.0007636601221747696, -0.00596915977075696, 0.009913569316267967, 0.0025484529323875904, -0.045483317226171494, -0.00513193616643548, -0.01569356583058834, -0.0007851161644794047, -0.010109739378094673, 0.025558091700077057, -0.03505830466747284, 0.007188213523477316, -0.012071434408426285, 0.004448845516890287, 0.033685117959976196, 0.015595480799674988, 0.010796332731842995, -0.015903746709227562, -0.012863119132816792, -0.017697297036647797, -0.005744966212660074, -0.03769258037209511, 0.025025632232427597, 0.04018673673272133, 0.03772060573101044, 0.04018673673272133, 0.01625405065715313, 0.023386215791106224, -0.0014537565875798464, -0.0011647568317130208, 0.0061723352409899235, 0.0308826956897974, -0.004504893906414509, -0.03816899284720421, -0.01779538206756115, 0.008988769724965096, -0.009983629919588566, 0.04077524691820145, -0.004238663706928492, -0.002975822426378727, -0.013949058018624783, 0.010894416831433773, -0.006329971831291914, -0.026805169880390167, -0.018818266689777374, -0.0006182844517752528, 0.0028864950872957706, 0.0028321982827037573, 0.010088720358908176, -0.01856604777276516, -0.0070340801030397415, 0.0149649353697896, -0.006816892419010401, 0.0019704531878232956, 0.00019704533042386174, 0.00876457616686821, 0.007272286340594292, 0.005142445210367441, 0.008533376269042492, -0.004119561053812504, -0.025936419144272804, 0.0002953490475192666, -0.023456275463104248, -0.0048376815393567085, 0.01771130971610546, 0.0025361923035234213, 0.032788343727588654, 0.0067748562432825565, 0.020093368366360664, -0.006368504837155342, 0.03718814626336098, 0.005265051033347845, -0.011111604981124401, -0.010537108406424522, -0.021102240309119225, -0.011756162159144878, -0.021844882518053055, -0.050499651581048965, 0.007405401207506657, 0.030294187366962433, -0.0023032408207654953, -0.031218985095620155, 0.008204091340303421, -0.02265758626163006, -0.020205466076731682, 0.010810344479978085, 0.008070976473391056, 0.024156881496310234, -0.02108822949230671, 0.02987382374703884, 0.04419420287013054, -0.012568864971399307, -0.001229562796652317, 0.029677653685212135, 0.002532689366489649, -0.012575870379805565, 0.026833195239305496, 0.03287241607904434, 0.010102733038365841, -0.02998591959476471, 0.002891749609261751, -0.0028847435023635626, -0.01183322910219431, -0.02892099879682064, 0.027561824768781662, 0.01919659413397312, 0.02079397439956665, -0.014061154797673225, -0.0076155830174684525, -0.025796297937631607, -0.014544572681188583, -0.0065401531755924225, -0.006133802235126495, -0.019799115136265755, -0.01444648765027523, -0.011342804878950119, 0.006928989663720131, 0.0035363067872822285, 0.017094776034355164, -0.014145227149128914, 0.004448845516890287, -0.0021683743689209223, 0.010109739378094673, -0.004623997025191784, 0.014348402619361877, -0.006760844029486179, 0.02201302908360958, -0.0117771802470088, 0.010957472026348114, 0.029537532478570938, 0.021130265668034554, -0.00604972941800952, -0.0025186771526932716, -0.014866851270198822, 0.03769258037209511, -0.010831362567842007, -0.026707084849476814, -0.010880405083298683, -0.013311506249010563, 0.003061646595597267, -0.013269470073282719, -0.008806612342596054, -0.019644981250166893, -0.007062104530632496, -0.007461449597030878, 0.03158330172300339, -0.017304958775639534, 0.0036221309565007687, 0.01443247590214014, 0.00772067392244935, 0.005398166365921497, -0.023694481700658798, 0.016085904091596603, -0.012701979838311672, -0.025488032028079033, -0.005867572035640478, -0.017893467098474503, 6.35471151326783e-05, -0.014005105942487717, 0.014082172885537148, -0.014285348355770111, 0.021900931373238564, 0.02868279442191124, -0.017276933416724205, 0.017935503274202347, 0.014285348355770111, -0.017332982271909714, -0.04719279333949089, 0.008729546330869198, -0.004490882158279419, -0.012148501351475716, 0.013080306351184845, -0.0008210222003981471, -0.01468469388782978, -0.028570696711540222, -0.014040136709809303, 0.01269497349858284, 0.017248909920454025, -0.011742150411009789, 0.0009335569920949638, 0.007051595486700535, 0.0045679486356675625, -0.016954654827713966, -0.02637079544365406, 0.026538940146565437, 0.0017734079156070948, -0.020359598100185394, 0.010544114746153355, -0.008042952045798302, 0.020948108285665512, -0.04262484610080719, -0.0007006056257523596, 0.0401587150990963, -0.030126040801405907, -0.017375018447637558, -0.009780454449355602, 0.018790243193507195, -0.012715991586446762, -0.000472908781375736, -0.005184481386095285, -0.014362415298819542, -0.015875723212957382, -0.00039649897371418774, -0.0011218447471037507, -0.0013477900065481663, 0.010761301964521408, 0.010502077639102936, -0.00710414070636034, 0.013304500840604305, -0.023736517876386642, -0.039233915507793427, -0.007524504326283932, -0.010214829817414284, -0.030854670330882072, -0.014033130370080471, 0.008806612342596054, 0.02411484532058239, 0.03357302024960518, 0.0049602878279984, -0.02485748752951622, -0.02334417961537838, -0.024002747610211372, 0.0035695855040103197, -0.014061154797673225, 0.004872711841017008, -0.029845798388123512, 0.002551955869421363, 0.0018863806035369635, 0.006715304683893919, -0.006928989663720131, -0.0017900472739711404, -0.023260105401277542, 0.01078232005238533, -0.01957492157816887, 0.016183989122509956, -0.013052282854914665, -0.021550629287958145, -0.02886495180428028, -0.019504860043525696, 0.021284397691488266, -0.005082893650978804, -0.002110574394464493, 0.0019109017448499799, -0.021494580432772636, -0.005538287106901407, -0.0044418396428227425, 0.003121197922155261, -0.014530560001730919, 0.009493205696344376, 0.014600620605051517, -0.017108788713812828, -0.005117923952639103, -0.004494384862482548, 0.022699622437357903, 0.004897233098745346, 0.0024959074798971415, 0.02453520894050598, 0.006844916846603155, 0.00759456492960453, -0.03744036331772804, 0.0081130126491189, -0.010887411423027515, 0.015875723212957382, 0.001019381103105843, 0.01848197542130947, -0.010488065890967846, 0.03707604855298996, 0.006070747505873442, 0.011665083467960358, -0.00928302388638258, 0.006252904888242483, -0.03236797824501991, -0.012407725676894188, 0.0012602143688127398, -0.009815484285354614, -0.00420363387092948, -0.02560012973845005, 0.04102746397256851, -0.049042392522096634, 0.004553936421871185, 0.0036641673650592566, -0.0034452280960977077, -0.01598781906068325, 2.735372254392132e-05, -0.004038991406559944, -0.01329048816114664, -0.010109739378094673, 0.01043902337551117, 3.002204539370723e-05, -0.011426877230405807, 0.008883679285645485, -0.00928302388638258, -0.016800522804260254, 0.012765034101903439, 0.006988540757447481, -0.008183073252439499, -0.008610443212091923, 0.0014441233361139894, -0.019715040922164917, 0.018874315544962883, 0.006144311279058456, 0.1840631067752838, -0.024605268612504005, 0.0011262234766036272, 0.014572597108781338, 0.0028094283770769835, 0.009191945195198059, 0.01446049939841032, 0.003017858602106571, 0.01593177206814289, 0.017248909920454025, -0.0010158781660720706, -0.003618628019466996, -0.008316188119351864, -0.014614633284509182, 0.02950950898230076, 0.006403535138815641, -0.03965427726507187, -0.03309660777449608, -0.013059288263320923, -0.030041968449950218, -0.006736322771757841, -0.018790243193507195, -0.01493691187351942, -0.008421279489994049, 0.022797705605626106, -0.004704566672444344, -0.009276018477976322, 0.00584305077791214, 0.02666504867374897, 0.008379243314266205, -0.030069991946220398, 0.00969638116657734, 0.00011018117947969586, -0.016772497445344925, -0.03760850802063942, -0.030098017305135727, 0.035758908838033676, -0.014040136709809303, -0.0023277620784938335, 0.010796332731842995, 0.004785136319696903, -0.011812210083007812, 0.037216171622276306, -0.023484300822019577, -0.023414239287376404, 0.003291094908490777, -0.02212512493133545, -0.017122801393270493, 0.01869215816259384, 0.0023172530345618725, -0.008204091340303421, 0.007993909530341625, 0.01625405065715313, 0.013801930472254753, -0.017346994951367378, -0.01574961468577385, -0.008645473048090935, -0.006354493089020252, 0.03589903190732002, 0.00416860356926918, -0.0081130126491189, 0.035226449370384216, -0.01102052628993988, 0.008218104019761086, 0.021396495401859283, 0.015315238386392593, -0.012870125472545624, 0.018706168979406357, -0.009829496964812279, -0.0033821735996752977, -0.00668027438223362, -0.021410508081316948, -0.029285313561558723, -0.00808498915284872, -0.038897622376680374, -0.023540347814559937, 0.036823831498622894, 0.029733702540397644, 0.02517976611852646, 0.04161597415804863, -0.0015203141374513507, 0.01167208980768919, -0.010319920256733894, 0.00647359574213624, -0.018201733008027077, -0.030182089656591415, 0.01385097298771143, -0.008981764316558838, 0.0059236204251646996, -0.005783499218523502, -0.004007464274764061, -0.002658798359334469, -0.01521715335547924, -0.01293317973613739, 0.01092944759875536, 0.017935503274202347, 0.029341362416744232, 0.0047115725465118885, 0.00775570422410965, 0.011188671924173832, -0.017571188509464264, 0.05394663289189339, 0.05548796430230141, -0.00827415194362402, -0.02950950898230076, -0.01176316849887371, 0.02384861558675766, 0.012722997926175594, 0.02286776714026928, -0.021410508081316948, -0.01649225689470768, -0.032676246017217636, -0.011153641156852245, -0.0005473481141962111, -0.013752887956798077, 0.01708076521754265, 0.004869209136813879, -0.028584709390997887, -0.004413815215229988, -0.006119790021330118, -0.005075887776911259, -0.029761726036667824, 0.005391160026192665, -0.006992043927311897, -0.014866851270198822, -0.020948108285665512, -0.0059306262992322445, 0.0010114993201568723, 0.007209231611341238, -0.027547812089323997, 0.02034558728337288, -0.011244719848036766, 0.016422195360064507, -0.001105205388739705, 0.0034977735485881567, -0.009072843007743359, -0.004368275869637728, -0.009184939786791801, 0.003545064479112625, -0.005723947659134865, -0.04018673673272133, 0.011132623068988323, -0.008743558079004288, 0.008386248722672462, -0.014628645032644272, -0.011588016524910927, -0.014922899194061756, 0.009591290727257729, 0.002911016345024109, 0.00851235818117857, -0.013633784838020802, -0.005702929571270943, 0.009353084489703178, -0.02007935754954815, 0.021760810166597366, -0.0031825010664761066, -0.014894874766469002, -0.009710393846035004, 6.064617264200933e-05, -0.0013232688652351499, -0.013780912384390831, -0.00693249236792326, 0.008792600594460964, -0.013584742322564125, -0.03926193714141846, -0.009402127005159855, -0.18114858865737915, 0.019673004746437073, 0.02352633699774742, -0.02597845532000065, 0.01002566609531641, 0.008288164623081684, 0.009619315154850483, -0.0049357665702700615, -0.041307706385850906, 0.007839776575565338, 0.01960294507443905, 0.0051879845559597015, -0.019294679164886475, -0.013129348866641521, -0.011756162159144878, -0.007937861606478691, -0.008715533651411533, 0.01919659413397312, 0.03542261943221092, 0.005769487004727125, 0.04971497505903244, -0.03262019529938698, -0.005237027071416378, -0.0037692582700401545, -0.022181173786520958, 0.00912188459187746, -0.009836502373218536, 0.01972905360162258, -0.010775314643979073, -0.010838368907570839, -0.024212930351495743, 0.0048376815393567085, 0.02254548855125904, 0.005895595997571945, -0.0039023731369525194, 0.00977344810962677, 0.0029075131751596928, 0.004382288083434105, -0.0401587150990963, 0.015651529654860497, -0.0018671139841899276, 0.036879878491163254, 0.02485748752951622, -0.017599212005734444, -0.0022524469532072544, -0.01907048374414444, 0.01368983369320631, -0.010761301964521408, 0.034357700496912, -0.02509569376707077, 0.017599212005734444, -0.032760318368673325, 0.013893009163439274, -0.005737959872931242, 0.0324520505964756, 0.01220454927533865, -0.008939727209508419, 0.011545980349183083, 0.00852636992931366, -0.00993458740413189, -0.017431067302823067, -0.023456275463104248, 0.013325518928468227, -0.0017821654910221696, -0.01318539772182703, -0.025431983172893524, -0.01705273985862732, 0.007825764827430248, -0.00808498915284872, 0.009871533140540123, 0.0029513011686503887, -0.03130305930972099, -0.0021911440417170525, -0.011658077128231525, 0.028514647856354713, 0.009297036565840244, -0.007244261913001537, 0.01469870563596487, -0.0014520051190629601, -0.009311048313975334, -0.015133081004023552, 0.043941982090473175, -0.02304992452263832, 0.006200359668582678, 0.013605761341750622, -0.018636109307408333, -0.0037132096476852894, 0.013970076106488705, -0.01410319097340107, -0.009885544888675213, 0.008049958385527134, -0.021760810166597366, 0.00909386109560728, 0.015413323417305946, -0.007951873354613781, 0.0060392203740775585, 0.006463086698204279, 0.0015001717256382108, 0.0012908658245578408, -0.02031756192445755, -0.01827179454267025, 0.007552528288215399, -0.0241708941757679, 0.030210113152861595, 0.011314780451357365, 0.029537532478570938, 0.006326468661427498, 0.022881779819726944, 0.019644981250166893, -0.0039023731369525194, 0.017192861065268517, 0.015034995973110199, 0.01694064401090145, 0.013038270175457, 0.0025904893409460783, 0.01067722961306572, -0.02517976611852646, -0.01518912985920906, 0.012141495011746883, -0.01595979556441307, 0.029649630188941956, 0.012575870379805565, -0.0011726386146619916, 0.01026387233287096, -0.019238630309700966, -0.010600162670016289, -0.06793072074651718, -0.0411115363240242, 0.008035946637392044, 0.01779538206756115, -0.024731377139687538, 0.022111112251877785, 0.003996954765170813, 0.01604386791586876, -0.02520778961479664, 0.01500697247684002, 0.008890685625374317, -0.0053105903789401054, 0.0018671139841899276, -0.013157373294234276, -0.0022226714063435793, -0.01176316849887371, -0.01141987182199955, 0.000446636084234342, -0.009542248211801052, 0.024128858000040054, -0.02209710143506527, -0.01681453548371792, 0.006179341580718756, -0.020065344870090485, -0.003415452316403389, -0.002047519898042083, -0.0154413478448987, 0.025319887325167656, 0.012302634306252003, -0.012751022353768349, 0.010726272128522396, -0.022419380024075508, 0.012078440748155117, 0.00288124056532979, -0.01368983369320631, -0.0004952406161464751, -0.02100415527820587, -0.006743329111486673, 0.003877852112054825, -0.02714146114885807, 0.002679816447198391, 0.015413323417305946, -0.006084759719669819, 0.025782287120819092, -0.0488181971013546, 0.01226760447025299, -0.01628207415342331, 0.015245177783071995, 0.0016595595516264439, -0.02026151493191719, -0.013157373294234276, -0.01217652577906847, -0.012645930983126163, 0.0036501551512628794, 0.013633784838020802, -0.014215287752449512, 0.01102052628993988, -0.015034995973110199, -0.02459125593304634, 0.00630194740369916, 0.0031789978966116905, 0.025768274441361427, -0.01925264298915863, 0.019771089777350426, 0.028724830597639084, -0.0016797019634395838, -0.043941982090473175, -0.017893467098474503, 0.008330200798809528, -0.004543427377939224, 0.003982943017035723, 0.009829496964812279, -0.0063965292647480965, 0.025880370289087296, -0.0351143516600132, -0.030574427917599678, -6.557230517501011e-05, -0.016660401597619057, -0.00031089375261217356, 5.3859057516092435e-05, -0.010074708610773087, -0.01818772219121456, 0.0026237680576741695, -0.013052282854914665, 0.0030633979476988316, -0.008547388017177582, -0.0016210261965170503, -0.009815484285354614, -0.008610443212091923, -0.0093741025775671, -0.004652021452784538, 0.00885565485805273, 0.03598310425877571, -0.01245676726102829, -0.006732820067554712, -0.002741119358688593, -0.00405650632455945, 0.0026080042589455843, -0.00664524408057332, 0.04024278745055199, -0.009009787812829018, 0.006217874586582184, -0.08552993088960648, 0.02384861558675766, -0.02384861558675766, -0.0035783431958407164, 0.015735602006316185, -0.029257290065288544, 0.0039584217593073845, -0.011104598641395569, 0.00943015143275261, -0.0012164264917373657, -0.03662766143679619, 0.014320378191769123, -0.005103911738842726, 0.023512324318289757, -0.00975943636149168, -0.01343761570751667, 0.04808956757187843, 0.001315387082286179, 0.02506766840815544, 0.010018659755587578, -0.012414731085300446, 0.005765984300523996, 0.04514702409505844, -0.0015711081214249134, -0.03561878949403763, -0.01771130971610546, -0.023218069225549698, 0.01720687374472618, -0.0005517269019037485, 0.0027516286354511976, 0.02239135466516018, 0.00018150063988287002, -0.008624454960227013, 0.03659963607788086, -0.003350646235048771, -0.02366645820438862, -0.0013092567678540945, 0.029705677181482315, 0.00723024969920516, 0.016240037977695465, -0.024254966527223587, -0.05019138753414154, 0.00037876490387134254, 0.003228040412068367, -0.01912653259932995, -0.006946504581719637, -0.008981764316558838, -0.018173709511756897, -0.008694515563547611, -0.00772067392244935, 0.0063089532777667046, 0.01996725983917713, 0.009801472537219524, -0.02201302908360958, 0.02419891767203808, -0.0005026845028623939, 0.025754261761903763, -0.0025607135612517595, 0.023176033049821854, -0.0014125960879027843, 0.03301253542304039, -0.0050793904811143875, 0.0432974249124527, 0.010915434919297695, 0.004511900246143341, -0.002919773804023862, -0.014558584429323673, 0.005538287106901407, -0.012050416320562363, -0.028402552008628845, 0.00300734955817461, 0.00022857257863506675, 0.003058143425732851, 0.013066294603049755, 0.04262484610080719, -0.003476755227893591, 0.0035976096987724304, -0.013921033591032028, -0.00639302609488368, 0.026973316445946693, 0.017697297036647797, 0.012113470584154129, -0.010719265788793564, 0.016394171863794327, 0.012239580042660236, -0.002499410416930914, -0.013227433897554874, 0.033685117959976196, 0.01999528333544731, -0.0008297797758132219, -0.02604851685464382, 0.020723914727568626, 0.010537108406424522, 0.010221836157143116, 0.0062704202719032764, 0.0308826956897974, -0.022699622437357903, 0.00014044952695257962, 0.02470335364341736, 0.01675848662853241, 0.031807493418455124, 0.0027989193331450224, -0.017543165013194084, 0.0010044933296740055, -0.03144317865371704, 0.016800522804260254, -5.528216206585057e-05, -0.01877623051404953, -0.004095039796084166, 0.024128858000040054, 0.007139171008020639, -0.016155965626239777, -0.017865443602204323, 0.012057422660291195, -0.030798623338341713, 0.018734194338321686, -0.002299737883731723, -0.01168610155582428, -0.03049035556614399, 0.019911210983991623, 0.01806161180138588, -0.0025064165238291025, 0.046968601644039154, -0.003951415419578552, 0.0042001307010650635, 0.030266162008047104, 0.01577763818204403, -0.04164399579167366, 0.041335731744766235, 0.006512129213660955, -0.005247536115348339, -0.0049252575263381, -0.0086594857275486, -0.004480372648686171, -0.02648289129137993, -0.015623505227267742, 0.010102733038365841, 5.257278826320544e-05, -0.03191959112882614, 0.06596902757883072, -0.003291094908490777, -0.01153897400945425, -0.011609034612774849, -0.017613224685192108, -0.008070976473391056, 0.010481059551239014, 0.0034014403354376554, -0.012926173396408558, 0.009332066401839256, -0.013514681719243526, -0.01625405065715313, -0.033208705484867096, -0.02562815323472023, -0.03553471714258194, -0.013052282854914665, -0.024128858000040054, 0.0047220815904438496, -0.009892551228404045, -0.0037657551001757383, 0.02995789609849453, -0.008610443212091923, 0.03138713166117668, 0.010523096658289433, 0.001968701835721731, -0.0034977735485881567, 0.02331615425646305, 0.00208430178463459, -0.007706661708652973, -0.031247010454535484, -0.001020256895571947, -0.00884164310991764, -0.029761726036667824, -0.01872018165886402, -0.002765640616416931, 0.005237027071416378, 0.02571222558617592, 0.0030126040801405907, -0.008288164623081684, -0.01925264298915863, -0.020387623459100723, 0.026524927467107773, -0.022349318489432335, -0.007650613319128752, 0.01750112883746624, -0.002259453060105443, -0.013311506249010563, -0.029565555974841118, -0.013465640135109425], 'license': '', 'publication_year': 2012, 'cited_by_count': 67974, 'is_paratext': False, 'language': 'en', 'id': 'https://openalex.org/W2163605009', 'type': 'article', 'title': 'ImageNet Classification with Deep Convolutional Neural Networks', 'display_name': 'ImageNet Classification with Deep Convolutional Neural Networks', 'url': ''}}]
    
    > Finished chain.
    ================================ Human Message =================================
    
    The work titled "ImageNet Classification with Deep Convolutional Neural Networks" was published in 2012. It has been cited 67,974 times, indicating its significant impact in the field. The article is written in English and is not considered a paratext. You can find more information about it through its identifier: [OpenAlex ID](https://openalex.org/W2163605009).
</pre>

By acquiring the data most similar to the user's query through semantic search and adjusting the query to fit the acquired data, we were able to obtain the desired answer.

In other words, even without knowing the precise terminology, we have built a QA System that allows us to obtain the desired answers!

This concludes the tutorial for the **Academic Search System**. Thank you for your hard work this time as well!
