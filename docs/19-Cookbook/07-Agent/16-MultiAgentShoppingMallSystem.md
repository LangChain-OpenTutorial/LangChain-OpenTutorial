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

# Multi Agent Shopping Mall System 

- Author: [Heeah Kim](https://github.com/yellowGangneng)
- Peer Review: [Jongcheol Kim](https://github.com/greencode-99), [HeeWung Song](https://github.com/kofsitho87)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/16-MultiAgentShoppingMallSystem.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/16-MultiAgentShoppingMallSystem.ipynb)

## Overview

Hello, everyone!  
In this tutorial, we will build a shopping mall system using **Multi Agent**!  
By the end of this tutorial, you will become the owner of a very impressive shopping mall!  
An impressive owner with several solid **agents** as employees!  

Our shopping mall will be structured with one manager who oversees the purchasing and sales systems,  
a purchasing specialist dedicated to the purchasing system,  
and a sales specialist dedicated to the sales system.

Each role will perform the following tasks:

**Manager**
- Validate Login: Determines whether the user is a legitimate user of our service.
- Assign Tasks: Assigns appropriate tasks to specialists based on user requirements.

**Purchase Specialist**
- Recommend Items: Suggests suitable items based on user needs.
- Purchase Items: Buys the items confirmed through recommendations.
- Cancel Purchase: Cancels the purchase if the item has not been dispatched. If it is in transit, unfortunately, we will need to follow exchange/refund procedures (not covered in this tutorial due to its extensive nature).
- Check Item Status: Shows the status of items purchased by the user. There are four possible statuses: Pre-dispatch, In transit, Delivered, and Cancelled.

**Sales Specialist**
- Check Sales History: For sellers, knowing how many items they have sold is crucial. They check their sales history.
- Restock Inventory: Popular items often sell out quickly. This function allows for restocking.
- Update Item Status: When a buyer places an order, we need to dispatch the item. Once safely delivered to the buyer, the shipment status should be updated. This function is for updating the item status.

Once these three roles are implemented, we will have the following type of service.  

![](./img/16-multi-agent-shopping-mall-system-01.png)

<center style="color:gray">Our Fabulous Service!</center>

Then, let's embark on building our shopping mall service!

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data preparation](#data-preparation)
- [Define State](#define-state)
- [Define Tools](#define-tools)
- [Define Nodes](#define-nodes)
- [Build the Graph](#build-the-graph)
- [Let's run our service](#let's-run-our-service)

### References

- [Kaggle Hub](https://github.com/Kaggle/kagglehub)
- [dbdiagram.io](https://dbdiagram.io/d)
- [Multi-agent supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [Customer Support Langgraph Tutorials](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/customer-support/customer-support.ipynb)
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
        "kaggle",
        "psycopg2-binary"
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
            "LANGCHAIN_PROJECT": "MultiAgentShoppingMallSystem",  # set the project name same as the title
        }
    )
```

## Data Preparation

Let's prepare our shopping mall data.  
First, we will use the `fashion-clothing-products-catalog` available on Kaggle.   
This dataset contains the following information:

- productId
- productName
- productBrand
- Gender
- Price
- Description
- Primary Color

There are two ways to obtain the data.

1. Download the data directly from the site.
   The platform we will be using to download the data is *Kaggle*, which hosts a variety of datasets for data analysis competitions.
   Among these, the dataset we will use can be found at the URL below:
   [Fashion Clothing Products Dataset](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data)
   There is a detailed description of the dataset on the site, along with various insightful materials. Please refer to them for more information!

2. Using code. *Kaggle* provides a api, which allows us to download datasets available on Kaggle very easily.

We will use the second method.

To download the dataset, first log in to [Kaggle](https://www.kaggle.com/),  
then click on your profile picture in the top right corner and select 'Settings' from the menu that appears.

![](./img/16-multi-agent-shopping-mall-system-02.png)


      <center style="color:gray">Kaggle Menu</center>
      
Below 'Settings,' there is an 'API' section where you need to click **Create New Token**.

![](./img/16-multi-agent-shopping-mall-system-03.png)

      <center style="color:gray">Kaggle Settings</center>
      
Then, a file named **kaggle.json** will be created.  
Using this json file, you can authenticate the API and utilize the Kaggle API.  

Once you move the file to the .kaggle folder, you will be ready to use the API. The path is as follows.
- mac : ~/.kaggle
- windows : C:/User/{username}/.kaggle

[Note] : [How to Use Kaggle](https://www.kaggle.com/docs/api?utm_me...)

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from glob import glob
import pandas as pd
import numpy as np

api = KaggleApi()
api.authenticate()

api.dataset_download_files('shivamb/fashion-clothing-products-catalog', unzip=True)
```

<pre class="custom">Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /Users/kimheeah/.kaggle/kaggle.json'
    Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /Users/kimheeah/.kaggle/kaggle.json'
    Dataset URL: https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog
</pre>

```python
data = glob('./*.csv')
data
```




<pre class="custom">['./myntra_products_catalog.csv']</pre>



```python
df = pd.read_csv(data[0])

# Since the `np.int64` type causes errors when loading data into PostgreSQL, convert it to `str` 
# (the database will handle type conversion automatically during insertion).
df = df.map(lambda x: str(x))

df.head()

df.loc[0,'ProductID']
```




<pre class="custom">'10017413'</pre>



```python
# Let's use exactly 100 pieces of data for convenience, as the dataset is quite large.
df = df.iloc[:100]
```

```python
### Since our service will not display images, let's remove unnecessary columns.
df = df.drop(columns=['NumImages'])
```

The data is ready, so we need a database to store it.  
We will use `PostgreSQL`.  
Refer to the code below to set up a `PostgreSQL` database using a Docker container.

However, instead of just `PostgreSQL`, we plan to use an image of `pgvector`,  
which allows for **vector data types** in `PostgreSQL`.

```yaml
services:
  postgres:
    image: ankane/pgvector
    restart: always
    volumes:
      - ./postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
```

Now that the database is up and running, let's insert our prepared data into the DB!

```python
import psycopg2

### Connect to the PostgreSQL database.
conn = psycopg2.connect(host='localhost', dbname='postgres', user='postgres', password='postgres', port=5432)
cursor=conn.cursor()
```

```python
query = "CREATE EXTENSION vector;"

try:
    cursor.execute(query)
    conn.commit()
    print("Init Setting Complete.")
except Exception as e:
    conn.rollback()
    print(f"An error occurred: {e}")
```

The database preparation is now complete!  
So, how should we insert the data into the database for our service?  
In our tutorial, we plan to implement only very simple features, so we have also designed the ERD in a simple way, as shown below.

![](./img/16-multi-agent-shopping-mall-system-04.png)

      <center style="color:gray">Our shopping mall service ERD</center>

Noteworthy in the data configuration is `description_ebd` of `inventory`.  
This is the reason why we use `pgvector`. This data is used when recommending items.  
When a user inputs what kind of item they want,  it compares the features of the item described by the user   
with `description_ebd` using **COSINE distance** and recommends the item with the highest similarity.

In short, a Semantic Search is conducted.

[NOTE] [pgvector github](https://github.com/pgvector/pgvector)

```python
queries = []

### Make CREATE TABLE Queries
queries.append(
    """
    CREATE TABLE Product (
      "id" int8 PRIMARY KEY,
      "name" varchar,
      "brand" varchar,
      "gender" varchar,
      "price" integer,
      "description" text,
      "primary_color" varchar
    );
    """
)
queries.append(
    """
    CREATE TABLE ServiceUser (
      "id" varchar PRIMARY KEY,
      "name" varchar,
      "gender" varchar,
      "type" varchar
      );
    """
)
queries.append(
    """
    CREATE TABLE Inventory (
      "id" int8 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
      "user_id" varchar,
      "product_id" int8,
      "remains" integer,
      "price" integer,
      "description" text,
      description_ebd vector(1536));
    """
)
queries.append(
    """
    CREATE TABLE Purchase (
      "customer_id" varchar,
      "inventory_id" integer,
      "status" varchar);
    """
)

### Make Foreign Key Relationship.
queries.append('ALTER TABLE Inventory ADD FOREIGN KEY ("user_id") REFERENCES ServiceUser ("id");')
queries.append('ALTER TABLE Inventory ADD FOREIGN KEY ("product_id") REFERENCES Product ("id");')
queries.append('ALTER TABLE Purchase ADD FOREIGN KEY ("customer_id") REFERENCES ServiceUser ("id");')
queries.append('ALTER TABLE Purchase ADD FOREIGN KEY ("inventory_id") REFERENCES Inventory ("id");')

try:
    for query in queries:
        cursor.execute(query)
        conn.commit()
    print("Table creation complete.")
except Exception as e:
    conn.rollback()
    print(f"An error occurred: {e}")
```

<pre class="custom">Table creation is complete.
</pre>

```python
### Insert Data
from psycopg2.extras import execute_values

insert_list = [tuple(df.iloc[i]) for i in range(len(df))]

query = f"INSERT INTO Product VALUES %s;"

try:
    execute_values(cursor, query, insert_list)
    conn.commit()
    print("Data insertion completed.")
except Exception as e:
    conn.rollback()
    print(f"An error occurred: {e}")
```

<pre class="custom">Data insert is complete.
</pre>

Let's load dummy data for buyers and sellers.

Creating fictional characters one by one could be one way, but...  
Since we have a reliable tool called GPT, let's actively utilize it!

Let's create 10 dummy data entries each for buyers and sellers.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.environ["OPENAI_API_KEY"],
)

prompt = ChatPromptTemplate.from_template(
   """
    You are an expert in generating PostgreSQL queries. Please review the given table information and column details, 
    and write appropriate queries based on the user's questions.
    
    Do not wrap in MD code blocks, do not insert newline characters with \n, just return the SQL query statement only.

    {db_info}
    
    {user_input}
    """
)

chain = prompt | llm
```

```python
insert_user_queries = chain.invoke({
    "db_info":"""
    # Table information 
    - ServiceUser: A table that contains user information. 
    # Column information 
    - id (varchar): The user's site ID. 
    - name (varchar): The user's name. 
    - gender (varchar): The user's gender. 
    - type (varchar): The user's type. Please just insert TYPE.
    """
    ,"user_input":"""
    # User's Question
    Please return 20 individual insert queries generating dummy data.    
    Please separate each query with \n.
    """}).content

insert_user_queries
```




<pre class="custom">"INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user1', 'pass1', 'John Doe', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user2', 'pass2', 'Jane Smith', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user3', 'pass3', 'Alice Johnson', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user4', 'pass4', 'Bob Brown', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user5', 'pass5', 'Charlie Davis', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user6', 'pass6', 'Daisy Evans', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user7', 'pass7', 'Eva Green', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user8', 'pass8', 'Frank Harris', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user9', 'pass9', 'Grace Lee', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user10', 'pass10', 'Henry Wilson', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user11', 'pass11', 'Ivy Carter', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user12', 'pass12', 'Jack Thompson', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user13', 'pass13', 'Kathy Martinez', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user14', 'pass14', 'Leo Robinson', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user15', 'pass15', 'Mia White', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user16', 'pass16', 'Noah Lewis', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user17', 'pass17', 'Olivia Hall', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user18', 'pass18', 'Paul Young', 'Male', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user19', 'pass19', 'Quinn Allen', 'Female', 'TYPE'); \nINSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user20', 'pass20', 'Ryan King', 'Male', 'TYPE');"</pre>



```python
insert_user_queries = insert_user_queries.split("\n")

insert_user_queries
```




<pre class="custom">["INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user1', 'pass1', 'John Doe', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user2', 'pass2', 'Jane Smith', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user3', 'pass3', 'Alice Johnson', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user4', 'pass4', 'Bob Brown', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user5', 'pass5', 'Charlie Davis', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user6', 'pass6', 'Daisy Evans', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user7', 'pass7', 'Eva Green', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user8', 'pass8', 'Frank Harris', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user9', 'pass9', 'Grace Lee', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user10', 'pass10', 'Henry Wilson', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user11', 'pass11', 'Ivy Carter', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user12', 'pass12', 'Jack Thompson', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user13', 'pass13', 'Kathy Martinez', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user14', 'pass14', 'Leo Robinson', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user15', 'pass15', 'Mia White', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user16', 'pass16', 'Noah Lewis', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user17', 'pass17', 'Olivia Hall', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user18', 'pass18', 'Paul Young', 'Male', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user19', 'pass19', 'Quinn Allen', 'Female', 'TYPE'); ",
     "INSERT INTO ServiceUser (id, pw, name, gender, type) VALUES ('user20', 'pass20', 'Ryan King', 'Male', 'TYPE');"]</pre>



```python
count = 0

try:
    for insert_user_query in insert_user_queries:
        if count<10:
            count += 1
            insert_user_query = insert_user_query.replace('TYPE', 'customer')
        else:
            insert_user_query = insert_user_query.replace('TYPE', 'vendor')
        cursor.execute(insert_user_query)
        conn.commit()
    print("ServiceUser data insertion completed.")
except Exception as e:
    conn.rollback()
    print(f"An error occurred: {e}")
```

<pre class="custom">ServiceUser data insert is complete.
</pre>

This concludes the preparation of buyer and seller data!

Next, let's load information about the items that the seller will sell into a table called `inventory`!

```python
query = "SELECT * FROM ServiceUser Where type='vendor';"

try:
    cursor.execute(query)
    conn.commit()
    vendors= cursor.fetchall()
    print("Vendor Data selection completed.")
except Exception as e:
    conn.rollback()
    print(f"An error occurred: {e}")

print("vendor data : ",vendors[0])
```

<pre class="custom">Vendor Data Selection is complete.
    vendor data :  ('user11', 'pass11', 'Ivy Carter', 'Female', 'vendor')
</pre>

Here, we are going to do something a bit interesting...  
We will be giving character to our dummy data!

Once character is given, we expect to see unique and lively product descriptions!  
Doesn't it sound fun?  
Let's create it together!

```python
personalities = {}

for vendor in vendors:
    personalities[vendor[2]] = llm.invoke(f"""
    The seller's name is {vendor[2]} and the gender is {vendor[3]}, please assign a random personality in one line.
    Please ensure that the given personality is different from the other sellers provided below.
    
    # Personalities generated for sellers so far
    {personalities}
    """).content
```

Shall we check the personalities of our buyers?

```python
personalities
```




<pre class="custom">{'Ivy Carter': 'Ivy Carter is an adventurous and free-spirited individual who thrives on spontaneity and exploration.',
     'Jack Thompson': 'Jack Thompson is a meticulous and detail-oriented planner who values structure and organization in all aspects of life.',
     'Kathy Martinez': 'Kathy Martinez is an empathetic and nurturing soul who finds joy in connecting with others and fostering supportive relationships.',
     'Leo Robinson': 'Leo Robinson is a charismatic and outgoing communicator who loves engaging with people and bringing a sense of enthusiasm to every interaction.',
     'Mia White': 'Mia White is a creative and imaginative thinker who enjoys expressing herself through art and innovative ideas, often inspiring those around her.',
     'Noah Lewis': 'Noah Lewis is a pragmatic and resourceful problem-solver who approaches challenges with a calm demeanor and a focus on practical solutions.',
     'Olivia Hall': 'Olivia Hall is a whimsical and playful dreamer who delights in storytelling and finding magic in the everyday moments of life.',
     'Paul Young': 'Paul Young is a driven and ambitious strategist who is always looking for opportunities to innovate and achieve his goals with determination.',
     'Quinn Allen': 'Quinn Allen is a thoughtful and introspective individual who enjoys deep conversations and finding meaning in every experience, often reflecting on the world around her.',
     'Ryan King': 'Ryan King is a witty and humorous individual who uses laughter as a way to connect with others and lighten any situation.'}</pre>



Quite unique and colorful sellers have been created!  
I look forward to seeing how these unique sellers create product descriptions.

```python
from langchain_openai import OpenAIEmbeddings

# We use the text-embedding-3-small model as the embedding model.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
```

```python
# Execution time approximately 5 to 10 minutes
import re
import json

# Errors occur because apostrophes and single-quoted sentences in the description are not distinguished, so separate handling is required.
def escape_single_quotes(sql_query):
    """
    Replace single quotes that meet specific conditions within an SQL query with two single quotes.
    Condition: Only replace single quotes that are not preceded or followed by (, ), , or space.
    """
    return re.sub(r"(?<![()\',\s])'(?=[^()\',\s]|$)", r"''", sql_query)

ex_print = True

for vendor in vendors:
    count=np.random.randint(5, 10)
    query = f"SELECT * FROM product ORDER BY RANDOM() LIMIT {count};"
    try:
        cursor.execute(query)
        products = cursor.fetchall()
        conn.commit()
        print("Product data selection completed.")
    except Exception as e:
        conn.rollback()
        print(f"An error occurred: {e}")
    for i in range(len(products)):
        try:
            insert_inventory_dict = chain.invoke({
                "db_info":"""
                # Table and Column Information
                
                ### Table Information 1
                - ServiceUser: A table that contains information about sellers.
                
                ### Column Information 1
                - id (varchar): The site ID of the seller.
                - name (varchar): The name of the seller.
                - gender (varchar): The gender of the seller.
                - type (varchar): A column indicating the seller.
                
                ### Table Information 2
                - Product: A table that contains information about products.
                
                ### Column Information 2
                - id (int8): The product ID.
                - name (varchar): The product name.
                - brand (varchar): The product brand.
                - gender (varchar): The product gender classification.
                - price (integer): The cost price of the product.
                - description (text): The product description.
                - primary_color (varchar): The color of the product.
                
                ### Table Information 3
                - Inventory: A table that contains information about products.
                
                ### Column Information 3
                - user_id (varchar): The ID of the seller supplying the product. Refers to the id in the ServiceUser table.
                - product_id (int8): The product ID. Refers to the id in the Product table.
                - remains (integer): The remaining quantity.
                - price (integer): The sale price of the product.
                - description (text): The product description.
                - description_ebd (vector) : Product description embeddings. Please just insert EBD Without single quote.
                """
                ,"user_input":f"""
                # User Question
                Using the given seller DB information and the nature of the product DB information, 
                generate an SQL query to insert data into the Inventory table, 
                which contains information about products to be registered by the seller to the service.
                Seller DB information
                : {"ServiceUser Values ",vendor}
                Seller personality
                : {personalities[vendor[2]]}
                Product DB information
                : {"Product Values ",products[i]}
                At this time, please insert a random value between 1 and 10 for the remains.
                Set the price higher than the cost price based on the registered information, 
                determining the margin according to the seller's nature.
                Modify and add the description based on the registered information, 
                excluding any seller-specific information but reflecting the seller's character and personality.
                Please follow the output format for the response.
                """+
                """
                # Output Format
                {"query" : "{Generated query}", "description" : "{Generated description sentence}"}
                """}).content
            insert_inventory_dict = json.loads(insert_inventory_dict)
            insert_inventory_query = escape_single_quotes(insert_inventory_dict['query'])
            description_ebd = embeddings.embed_query(insert_inventory_dict["description"])
            insert_inventory_query = insert_inventory_query.replace("EBD", "'"+str(description_ebd)+"'")
            cursor.execute(insert_inventory_query)
            conn.commit() 
            print(f"{i+1}th data insertion completed out of {len(products)}th data")
        except Exception as e:
            conn.rollback()
            print(f"An error occurred: {e}")
```

Now all the data preparation is complete! Let's officially start building the service!

## Define State

```python
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, Callable
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition

# When the administrator assigns tasks and it switches to the purchasing expert/sales expert state, 
# it stores how many conversations have taken place during that state. 
# After the expert has completed all possible actions and it returns to the administrator state, 
# it helps to erase all conversation history during that state so that the administrator can refocus on their role. (Refer to pop_dialog_state)
def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class User(BaseModel):
    id: str
    name: str
    gender: str
    type: str

class State(TypedDict):
    user_info: Annotated[User, "User information"]
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[
            Literal[
                "customer",
                "vendor",
            ]
        ],
        update_dialog_stack,
    ]    
```

# Define Tools

```python
@tool
def check_validation(
    config: RunnableConfig):
    """
    Verify the validity by checking if the matching ID exist in the user information table.
    """
    query = f"SELECT * FROM serviceuser WHERE id='{config['configurable']['user_id']}';"
    user = None
    
    try:
        cursor.execute(query)
        row = cursor.fetchone()
        colnames = [desc[0] for desc in cursor.description]

        user_info = dict(zip(colnames, row))
        user = User(**user_info)
        
        conn.commit()
    except Exception as e:
        conn.rollback()

    return user
```

Let's define a tool for administrators.

The tasks that administrators can perform are straightforward:
- Verify whether the login is valid or not.

Once the validity is determined, they can call an expert.

The obtained user information will be stored in the config and can be utilized throughout the service usage.

Once the user is verified as valid, the administrator delegates the task to the expert.

```python
class ToSalesSpecialistAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle sales service."""

    request: str = Field(
        description="Any necessary followup questions the update sales specailist should clarify before proceeding."
    )

class ToPurchaseSpecialistAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle purchase service."""

    request: str = Field(
        description="Any necessary followup questions the update purchase specailist should clarify before proceeding."
    )
```

Let's define tools for sales experts.

As mentioned in the overview, sales experts can perform the following four tasks:
1. Recommend products
2. Purchase products
3. Cancel purchases
4. Check product status

```python
@tool
def get_recommendation(query: Annotated[str, "User's question"], 
                       color:Annotated[str, "The color of the product the user wants. If it does not exist, please fill it with None."], 
                       count:Annotated[int,"Number of recommended products. If it does not exist, please fill it with None."]):
    """
    Recommend products to the user.
    """

    query = f"""SELECT p.name, p.brand, p.gender, i.price, i.description
            FROM inventory i
            JOIN product p on p.id = i.product_id
            WHERE 1=1"""

    if color != 'None' and color is not None:
        query += f" AND REPLACE(lower(p.primary_color), ' ', '') = REPLACE(lower('{color}'), ' ', '')"

    # <=> : Get COSINE distance
    query += " ORDER BY i.description_ebd <=> '"+str(embeddings.embed_query(query))+"'"
    
    if count != 'None' and count is not None:
        query += f" LIMIT {count};"
    else:
        query += f" LIMIT 5;"

    print(query)
    try:
        cursor.execute(query)
        recommendations = cursor.fetchall()
        conn.commit()

        return recommendations
    except Exception as e:
        conn.rollback()
    
        return "The recommendation process encountered an error and has been canceled."

@tool
def purchase_product(inventory_id:Annotated[str, "Inventory product's id"], 
                     config:RunnableConfig):
    """
    Purchase products.
    """
    my_id = config['configurable']['user_id']
    query = f"INSERT INTO purchase({my_id}, {inventory_id}, 'Pre-dispatch')"
    
    try:
        cursor.execute(query)
        conn.commit()
        return "The purchase has been completed."
    except Exception as e:
        conn.rollback()
        return "The purchase was cancelled due to a problem."
    
    

@tool
def cancel_purchase(inventory_id:Annotated[str, "Inventory product's id"], 
                    config:RunnableConfig):
    """
    Canceling the purchase of the product.
    """
    my_id = config['configurable']['user_id']
    query = f"UPDATE purchase SET status='cancelled' WHERE customer_id = '{my_id}' and inventory_id = '{inventory_id}'"
    
    try:
        cursor.execute(query)
        conn.commit()
        return "The purchase cancellation has been completed."
    except Exception as e:
        conn.rollback()
        return "There was an issue with the cancellation, so it was canceled."
    

@tool
def check_item_status(config:RunnableConfig):
    """
    Checking the condition of the purchased product.
    """
    my_id = config['configurable']['user_id']
    query = f"SELECT * FROM purchase WHERE customer_id = '{my_id}'"
    
    try:
        cursor.execute(query)
        purchase_history = cursor.fetchall()

        conn.commit()
        return purchase_history
    except Exception as e:
        conn.rollback()
        return "There was an issue with the data retrieval."
```

Next, let's define the tools of a purchasing expert.

A purchasing expert can perform the following three tasks:
1. Checking sales records
2. Replenishing stock
3. Updating product conditions

```python
@tool 
def check_sales_record(config:RunnableConfig):
    """
    Checking sales records.
    """
    my_id = config['configurable']['user_id']
    query = f"SELECT * FROM purchase p JOIN inventory i ON i.id = p.inventory_id WHERE i.vendor_id = '{my_id}'"
    
    try:
        cursor.execute(query)
        purchase_history = cursor.fetchall()

        conn.commit()
        return purchase_history
    except Exception as e:
        conn.rollback()
        return "There was an issue with the data retrieval."

@tool
def restock(product_id:Annotated[str, "Product's id"], 
            count:Annotated[int, "Number of stocks to replenish."], 
            config:RunnableConfig):
    """
    Replenishing product stock.
    """
    my_id = config['configurable']['user_id']
    query = f"UPDATE inventory SET remains = remains + {count} WHERE user_id = '{my_id}' AND product_id = '{product_id}'"
    
    try:
        cursor.execute(query)
        conn.commit()
        return "Stock replenishment is complete."
    except Exception as e:
        conn.rollback()
        return "There was an issue during stock replenishment, and it has been canceled."

@tool
def update_item_status(inventory_id:Annotated[str, "Inventory product's id"], 
                       status:Annotated[Literal['In transit', 'Delivered', 'Cancelled'], "The product status can be 'In transit', 'Delivered', or 'Cancelled'."]):
    """
    Updating the status of the product.
    The status can be updated to one of the following three:
    - In transit: The product has been shipped from the warehouse.
    - Delivered: The product has been delivered to the customer.
    - Cancelled: The order has been cancelled by the customer.
    """
    query = f"UPDATE purchase SET statue='{status}' FROM inventory WHERE purchase.inventory_id = inventory.id AND purchase.inventory_id = '{inventory_id}'"
    
    try:
        cursor.execute(query)
        conn.commit()
        return "The stock replenishment is complete."
    except Exception as e:
        conn.rollback()
        return "The stock replenishment was canceled due to a problem."
    return
```

The Sales Expert and Purchase Expert will define a common tool called `CompleteOrEscalate`.  
This tool will check whether the current user has completed their task, 
wants to cancel their task,    
or has changed their mind about the current task.  
If the task is finished, it will proceed to return to the administrator.

```python
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the items for more information.",
            },
        }
```

# Define Nodes

Since all the tools have been defined, we need to define the agents that will use these tools!
- Administrator Agent
- Purchase Expert Agent
- Sales Expert Agent
We will define these three types of agents.

Let's define the nodes necessary for these agents.

```python
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an excellent customer support representative for a clothing shopping mall service."
            "Your main role is to perform login verification to ensure that the user is authorized to use the service. \
            Once verified, you delegate tasks to the appropriate expert based on the user's requested service."
            "There are two experts you can delegate tasks to:"
            "1. purchase_specialist : An expert who provides appropriate services to consumers wishing to purchase items."
            "2. sales_specialist: An expert who offers appropriate services to sellers wishing to sell items."
            "Delegate the next action to the experts based on the user's request. If the process is complete, respond with END."
        ),
        ("placeholder", "{messages}"),
    ]
)
primary_assistant_tools = [
    check_validation
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToSalesSpecialistAssistant,
        ToPurchaseSpecialistAssistant,
    ]
)
```

```python
purchase_specialist_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert specialized in customer services."
            "The primary assistant delegates tasks to you when the user wants an item recommendation, a purchase, \
            a purchase cancellation, or to check purchase history."
            "If the user needs help, but you do not have the appropriate tool, 'CompleteOrEscalate' to the host assistant."
            "Do not waste the user's time, and do not create incorrect tools or functions."
        ),
        ("placeholder", "{messages}"),
    ]
)
purchase_specialist_tools = [get_recommendation, purchase_product, cancel_purchase, check_item_status]
purchase_specialist_runnable = purchase_specialist_prompt | llm.bind_tools(
    purchase_specialist_tools + [CompleteOrEscalate]
)
```

```python
sales_specialist_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert specialized in vendor services."
            "The primary assistant delegates tasks to you when the user wants to check item sales history, \
            restock inventory, or update item status."
            "If the user needs help, but you do not have the appropriate tool, 'CompleteOrEscalate' to the host assistant."
            "Do not waste the user's time, and do not create incorrect tools or functions."
        ),
        ("placeholder", "{messages}"),
    ]
)
sales_specialist_tools = [check_sales_record, restock, update_item_status]
sales_specialist_runnable = sales_specialist_prompt | llm.bind_tools(
    sales_specialist_tools + [CompleteOrEscalate]
)
```

```python
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
```

```python
# Validate the user's information. After validation, retrieve the user's information.
def get_user_info_node(state: State):
    return {"user_info": check_validation.invoke({})}
```

```python
# Notifying that you have been delegated a task from the administrator and your current status has become purchase_specialist or sales_specialist.
def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node
```

```python
# When the expert returns to the administrator status after completing all tasks, the conversation history during the expert status is popped.
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }

```

```python
# Graph branching settings for sales_specialist.
def route_sales_specialist(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "sales_specialist_tools"


# Graph branching settings for purchase_specialist.
def route_purchase_specialist(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "purchase_specialist_tools"
```

```python
# Graph branching settings for primary_assistant.
def route_primary_assistant(
    state: State
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToPurchaseSpecialistAssistant.__name__:
            print(state)
            if state['user_info'] is None:
                raise ValueError("Unauthorized user")
            return "enter_purchase_specialist"
        elif tool_calls[0]["name"] == ToSalesSpecialistAssistant.__name__:
            if state['user_info'] is None:
                raise ValueError("Unauthorized user")
            return "enter_sales_specialist"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")
```

```python
# If an error occurs after calling the tool, it returns the cause of the error.
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
```

## Build the Graph

```python
builder = StateGraph(State)

builder.add_node("get_user_info", get_user_info_node)
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_edge(START, "get_user_info")
builder.add_edge("get_user_info", "primary_assistant")


builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "primary_assistant_tools",
        "enter_purchase_specialist",
        "enter_sales_specialist",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")
```




<pre class="custom"><langgraph.graph.state.StateGraph at 0x152f7ba9ad0></pre>



```python
builder.add_node(
    "enter_purchase_specialist",
    create_entry_node("Purchase Specialist", "purchase_specialist"),
)
builder.add_node("purchase_specialist", Assistant(purchase_specialist_runnable))
builder.add_edge("enter_purchase_specialist", "purchase_specialist")

builder.add_node(
    "purchase_specialist_tools",
    create_tool_node_with_fallback(purchase_specialist_tools),
)
builder.add_conditional_edges(
    "purchase_specialist",
    route_purchase_specialist,
    ["purchase_specialist_tools", "leave_skill", END],
)
builder.add_edge("purchase_specialist_tools", "purchase_specialist")
```




<pre class="custom"><langgraph.graph.state.StateGraph at 0x152f7ba9ad0></pre>



```python
builder.add_node(
    "enter_sales_specialist",
    create_entry_node("Sales Specialist", "sales_specialist"),
)
builder.add_node("sales_specialist", Assistant(sales_specialist_runnable))
builder.add_edge("enter_sales_specialist", "sales_specialist")

builder.add_node(
    "sales_specialist_tools",
    create_tool_node_with_fallback(sales_specialist_tools),
)
builder.add_conditional_edges(
    "sales_specialist",
    route_sales_specialist,
    ["sales_specialist_tools", "leave_skill", END],
)
builder.add_edge("sales_specialist_tools", "sales_specialist")
```




<pre class="custom"><langgraph.graph.state.StateGraph at 0x152f7ba9ad0></pre>



```python
builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")
```




<pre class="custom"><langgraph.graph.state.StateGraph at 0x152f7ba9ad0></pre>



```python
graph = builder.compile()
```

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass
```


    
![png](./img/output_70_0.png)
    


## Let's run our service

```python
import shutil
import uuid

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
        "user_id": "user1",
    }
}

events = graph.stream(
    {"messages": ("user", "Please recommend items suitable for sporty activities.")}, config, stream_mode="values"
)
```

```python
for event in events:
    event['messages'][-1].pretty_print()
```

<pre class="custom">================================ Human Message =================================
    
    Please recommend items suitable for sporty activities.
    ================================ Human Message =================================
    
    Please recommend items suitable for sporty activities.
    ================================== Ai Message ==================================
    Tool Calls:
      check_validation (call_ezsUpd5idMyXUZATs41ORjIr)
     Call ID: call_ezsUpd5idMyXUZATs41ORjIr
      Args:
    ================================= Tool Message =================================
    Name: check_validation
    
    id='user1' name='John Doe' gender='Male' type='customer'
    {'messages': [HumanMessage(content='Please recommend items suitable for sporty activities.', additional_kwargs={}, response_metadata={}, id='29848ae2-81fe-4675-8ae8-026f4106e274'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ezsUpd5idMyXUZATs41ORjIr', 'function': {'arguments': '{}', 'name': 'check_validation'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 267, 'total_tokens': 278, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-87ffb20a-419b-4ae6-8694-69f9f80757d5-0', tool_calls=[{'name': 'check_validation', 'args': {}, 'id': 'call_ezsUpd5idMyXUZATs41ORjIr', 'type': 'tool_call'}], usage_metadata={'input_tokens': 267, 'output_tokens': 11, 'total_tokens': 278, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content="id='user1' name='John Doe' gender='Male' type='customer'", name='check_validation', id='d2543774-c76b-4d2c-a19c-bb52b57989f3', tool_call_id='call_ezsUpd5idMyXUZATs41ORjIr'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_kI8je6oRdcaJrqr9AWhlQ0FP', 'function': {'arguments': '{"request":"Please recommend items suitable for sporty activities for a male customer."}', 'name': 'ToPurchaseSpecialistAssistant'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 303, 'total_tokens': 332, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-5b1397b9-57d3-41c5-9f55-c03d54a45633-0', tool_calls=[{'name': 'ToPurchaseSpecialistAssistant', 'args': {'request': 'Please recommend items suitable for sporty activities for a male customer.'}, 'id': 'call_kI8je6oRdcaJrqr9AWhlQ0FP', 'type': 'tool_call'}], usage_metadata={'input_tokens': 303, 'output_tokens': 29, 'total_tokens': 332, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'user_info': User(id='user1', name='John Doe', gender='Male', type='customer'), 'dialog_state': []}
    ================================== Ai Message ==================================
    Tool Calls:
      ToPurchaseSpecialistAssistant (call_kI8je6oRdcaJrqr9AWhlQ0FP)
     Call ID: call_kI8je6oRdcaJrqr9AWhlQ0FP
      Args:
        request: Please recommend items suitable for sporty activities for a male customer.
    ================================= Tool Message =================================
    
    The assistant is now the Purchase Specialist. Reflect on the above conversation between the host assistant and the user. The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are Purchase Specialist, and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool. If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control. Do not mention who you are - just act as the proxy for the assistant.
    ================================== Ai Message ==================================
    Tool Calls:
      get_recommendation (call_fnL4S6LVDYluEnVNIdzPezIa)
     Call ID: call_fnL4S6LVDYluEnVNIdzPezIa
      Args:
        query: sporty activities
        color: None
        count: 5
    SELECT p.name, p.brand, p.gender, i.price, i.description
                FROM inventory i
                JOIN product p on p.id = i.product_id
                WHERE 1=1 ORDER BY i.description_ebd <=> '[0.015088276006281376, -0.012633828446269035, -0.009258963167667389, -0.02581552416086197, 0.029782086610794067, -0.004999292548745871, -0.006086810026317835, 0.04755491390824318, -0.010091722011566162, -0.02331724762916565, 0.04869448021054268, -0.002922873944044113, -0.013280312530696392, -0.010338262654840946, 0.03129420056939125, 0.024325324222445488, -0.025092339143157005, -0.0027407079469412565, -0.050929777324199677, 0.0432596318423748, 0.04183517396450043, 0.021301094442605972, -0.0202163178473711, -0.001449109986424446, -0.02447872795164585, 0.04514429718255997, -0.046590667217969894, -0.009867096319794655, 0.03230227902531624, 0.0038241164293140173, 0.005675909109413624, -0.023361077532172203, -0.0006707955035381019, -0.019635576754808426, -0.006470317021012306, -0.02168460190296173, 0.04008200019598007, 0.03675096482038498, 0.03083399310708046, -0.014759555459022522, -0.011505221016705036, 0.02331724762916565, 0.029387621209025383, -0.02491702139377594, 0.021838005632162094, 0.01910962350666523, -0.013751478865742683, -0.03274057060480118, -0.02003004215657711, -0.03223653510212898, -0.007538659032434225, -0.019547918811440468, -0.0015847071772441268, 0.020797057077288628, -0.025004681199789047, -0.01148330606520176, 0.0454949326813221, 0.0387013703584671, 0.054567620158195496, 0.00016187777509912848, 0.05658377334475517, -0.011297031305730343, 0.0181892067193985, 0.017356447875499725, -0.01622783951461315, 0.020457379519939423, -0.021706517785787582, 0.0028269970789551735, 0.014036369509994984, -0.028532948344945908, 0.017203044146299362, 0.008886413648724556, -0.023645969107747078, 0.04680981487035751, 0.0016257973620668054, 0.01370764896273613, -0.03758372366428375, -0.000849879696033895, 0.02513616904616356, 0.021180564537644386, -0.0046294815838336945, 0.022265342995524406, -0.00464865704998374, 0.010447836481034756, 0.002698248252272606, -0.03881094604730606, -0.044793661683797836, -0.003952865023165941, 0.009806831367313862, -0.0682423934340477, -0.0728006586432457, -0.008130356669425964, 0.017893357202410698, -0.023163845762610435, 0.02640722133219242, -0.021476412191987038, 0.06289520859718323, -0.013247440569102764, -0.0819171741604805, 0.009034337475895882, -0.008250887505710125, 0.008113919757306576, 0.0073852562345564365, -0.0016901717754080892, 0.000662577454932034, -0.0083878543227911, 0.0017518068198114634, -0.009078167378902435, -0.04345686361193657, 0.019482174888253212, -0.026144245639443398, 0.01639220118522644, 0.023382991552352905, 0.05338422581553459, -0.023843200877308846, 0.009631513617932796, -0.032937806099653244, 0.0019284941954538226, -0.03927115350961685, -0.03447183221578598, -0.007478393614292145, 0.01894526369869709, 0.033507585525512695, -0.08862307667732239, -0.004577434156090021, -0.0069579193368554115, -0.036816708743572235, -0.0426679328083992, 0.017575595527887344, -0.06968876719474792, 0.03694819658994675, 0.008859020657837391, 0.01611826755106449, -0.026012757793068886, -0.03970944881439209, -0.00765371136367321, 0.017334533855319023, 0.030110806226730347, 0.002477731555700302, 0.014408919960260391, -0.023163845762610435, -0.011351818218827248, 0.02041354961693287, -0.016709964722394943, -0.031557176262140274, -0.035918205976486206, 0.002713314723223448, -0.03458140790462494, 0.003391300793737173, -0.01639220118522644, 0.02065461128950119, -0.029847830533981323, -0.03537033498287201, -0.045100465416908264, -0.010393049567937851, 0.004018609412014484, 0.04413621872663498, -0.033178865909576416, -0.03721117228269577, -0.022199597209692, -0.0004201460105832666, -0.03444992005825043, 0.005043121986091137, -0.024785533547401428, -0.06850537657737732, -0.021399710327386856, -0.0637717992067337, -0.0019942382350564003, -0.021936621516942978, 0.012765316292643547, -0.045231953263282776, 0.009374015964567661, -0.031118882820010185, -0.024654045701026917, 0.008267323486506939, 0.00201204395852983, 0.002757144160568714, -0.005785482469946146, 0.000919732847250998, 0.0056074257008731365, -0.008612480014562607, 0.00502120703458786, 0.013258397579193115, -0.025289570912718773, 0.051499560475349426, 0.00640457309782505, -0.01856175623834133, 0.0038487704005092382, 0.009697257541120052, -0.04900128394365311, 0.01633741334080696, 0.012239363975822926, 0.0409366711974144, 0.026144245639443398, 0.022725550457835197, -0.048080865293741226, -0.023755542933940887, 0.044662173837423325, -0.02605658583343029, 0.07017089426517487, 0.07293214648962021, -0.0051800888031721115, 0.04702896252274513, 0.03366098925471306, -0.07955038547515869, 0.01718113012611866, 0.026209989562630653, 0.016074437648057938, -0.010135551914572716, 0.014430833980441093, 0.024873191490769386, -0.019854724407196045, -0.008185142651200294, 0.023251503705978394, 0.020786099135875702, -0.037386488169431686, -0.00973012950271368, 0.008650830946862698, 0.019986212253570557, 0.04505663737654686, -0.020314933732151985, -0.020939502865076065, 0.015493697486817837, -0.014814341440796852, 0.010195816867053509, 0.0694257915019989, 0.03668522089719772, 0.009763001464307308, 0.02572786621749401, -0.05987098067998886, 0.023141929879784584, -0.00342965149320662, 0.01611826755106449, -0.009149390272796154, 0.02445681206882, 0.05320890620350838, -0.010809428989887238, -0.014858171343803406, 0.004569216165691614, -0.037386488169431686, -0.00342965149320662, -0.0054595014080405235, -0.02081897109746933, -0.020862801000475883, 0.024829363450407982, -0.014595194719731808, -0.010316348634660244, -0.012381809763610363, -0.011187457479536533, 0.0011162803275510669, 0.0011833941098302603, 0.009209655225276947, -0.01967940665781498, -0.03694819658994675, -0.014639023691415787, -0.028817839920520782, -0.032368022948503494, 0.006437445059418678, 0.03263099864125252, -0.04904511198401451, -0.07240618765354156, 0.046020884066820145, -0.013346056453883648, -0.010113636963069439, -0.06789176166057587, 0.0409366711974144, 0.0637717992067337, 0.016589432954788208, -0.02469787374138832, -0.007604403421282768, 0.02103811874985695, 0.013378928415477276, -0.00540745398029685, -0.01468285359442234, -0.014507535845041275, 0.014200730249285698, -0.005270487163215876, 0.049571067094802856, 0.030242295935750008, 0.0546552799642086, -0.054216984659433365, -0.01896717958152294, 0.020424507558345795, 0.013795307837426662, -0.004533605184406042, 0.0313161164522171, 0.007368820253759623, 0.04632769152522087, -0.0025708689354360104, -0.006267606280744076, -0.006327871698886156, -0.01057384628802538, 0.02660445310175419, 0.001949039287865162, 0.03550182655453682, 0.008848062716424465, 0.003437869716435671, 0.017893357202410698, 0.02980400063097477, 0.018989093601703644, 0.039753276854753494, 0.031798239797353745, 0.02296661213040352, -0.04222964122891426, -0.011110756546258926, 0.059213537722826004, 0.034055452793836594, -0.032148875296115875, 0.016918154433369637, -0.005254050716757774, -0.013148823752999306, -0.02087375894188881, 0.05943268537521362, -0.006289520766586065, -0.022155769169330597, -0.012995420955121517, -0.0032598127145320177, -0.028927413746714592, -0.011253202334046364, -0.005100647918879986, 0.0017189348582178354, -0.01416785828769207, 0.030855907127261162, 0.030176552012562752, -0.0255963783711195, -0.03824116289615631, -0.04891362413764, -0.0034515662118792534, 0.0182111207395792, 0.008639873005449772, -0.025859354063868523, -0.021509284153580666, -0.019778022542595863, 0.03477863967418671, 0.03710160031914711, -0.021312052384018898, 0.007314033340662718, 0.04315005615353584, -0.028598692268133163, -0.007297597359865904, 0.018737073987722397, -0.020041000097990036, -0.05447996035218239, 0.014003497548401356, 0.029212303459644318, -0.0535157136619091, -0.0008039958192966878, 0.06017778441309929, -0.023251503705978394, -0.014923915266990662, -0.006623720284551382, -0.0007964626420289278, 0.06250074505805969, 0.020994288846850395, -0.005807397421449423, -0.039994340389966965, 0.05426081269979477, -0.06109820306301117, -0.03039569780230522, -0.00431445799767971, 0.0023667882196605206, 0.008245408535003662, 0.05259529501199722, 0.029979318380355835, -0.01943834498524666, -0.0387013703584671, 0.007335948292165995, -0.02594701200723648, -0.0381096750497818, -0.012261277996003628, 0.09291835874319077, -0.022725550457835197, -0.03883286193013191, -0.01100666169077158, -0.01631549932062626, -0.014376047998666763, -0.03552373871207237, 0.024632129818201065, 0.020227273926138878, 0.03309120610356331, 0.0019284941954538226, 0.0009916404960677028, 0.03859179839491844, -0.03900817781686783, 0.020731313154101372, -0.00921513419598341, 0.040783267468214035, 0.0671028345823288, -0.057723335921764374, -0.016600390896201134, 0.018200164660811424, -0.05759184807538986, 0.022550232708454132, -0.034099284559488297, -0.06079139560461044, -0.028357630595564842, 0.04301856830716133, -0.02331724762916565, -4.412902853800915e-05, 0.021596943959593773, -0.0014737640740349889, -0.029146559536457062, -0.018200164660811424, -0.018254950642585754, -0.03666330501437187, -0.0058950562961399555, -0.026363391429185867, 0.0038158982060849667, 0.01320361066609621, -0.012414681725203991, -0.032368022948503494, -0.0031283244024962187, -0.003703585360199213, 0.0751236155629158, 0.002243518130853772, -0.04058603569865227, 0.029321877285838127, 0.007259246427565813, 0.025311486795544624, 0.006108724512159824, 0.06666453927755356, -0.015921033918857574, 0.019832810387015343, 0.02785359136760235, -0.06223776564002037, -0.01004241406917572, 0.01472668256610632, -0.05088594928383827, -0.014463706873357296, -0.030198466032743454, -0.01195447240024805, 0.04003816843032837, -0.004221320617944002, -0.00671685766428709, -0.023514481261372566, -0.0375618077814579, -0.0051636528223752975, -0.0041254437528550625, 0.010831343941390514, -0.03267482668161392, -0.00886449869722128, 0.02065461128950119, 0.001668257056735456, -0.02003004215657711, -0.01256808452308178, 0.002720162970945239, -0.005054079461842775, 0.0010847779922187328, 0.016896238550543785, 0.0013806265778839588, -0.039314985275268555, 0.022572148591279984, -0.05075446143746376, 0.0352826789021492, 0.008009824901819229, 0.05767950788140297, -0.019986212253570557, 0.07876145839691162, -0.052726782858371735, 0.03322269394993782, -0.02991357445716858, 0.055926330387592316, 0.005202003289014101, -0.0028900019824504852, 0.011581922881305218, 0.015252635814249516, 0.006470317021012306, -0.0035282678436487913, 0.025903183966875076, -0.03734266012907028, -0.026714026927947998, -0.06723432242870331, 0.005182828288525343, 0.03352950140833855, 0.012787231244146824, 0.05268295481801033, -0.015548484399914742, 0.0062073408626019955, -0.010771078057587147, -0.06381562352180481, 0.004018609412014484, -0.0063552651554346085, 0.003366646822541952, 0.0012340719113126397, -0.037276916205883026, 0.0909898653626442, -0.0029749213717877865, -0.027590615674853325, -0.03672904893755913, -0.006218297872692347, 0.007144194561988115, 0.015559441410005093, 0.04698513075709343, -0.009017901495099068, -0.03572097048163414, -0.0128091461956501, -0.0007457848405465484, 0.004111746791750193, 0.000872479286044836, -0.014770512469112873, 0.037276916205883026, 0.012546169571578503, 0.008596044033765793, 0.04023540019989014, -0.019043879583477974, 0.021180564537644386, -0.012984463945031166, -0.06394711136817932, 0.03171057999134064, 0.020336847752332687, -0.014113071374595165, 0.025311486795544624, -0.00026622944278642535, -0.004295282531529665, -0.017093472182750702, 0.0011053229682147503, -0.017038684338331223, -0.01519784890115261, -0.02923421934247017, 0.003999433945864439, -0.00992188323289156, 0.01780569925904274, -0.0023352860007435083, 0.00025133430608548224, 0.012261277996003628, -0.033726733177900314, 0.003010532818734646, -0.03309120610356331, -0.041922833770513535, -0.03843839466571808, 0.03285014629364014, 0.02035876177251339, 0.01742219179868698, 0.0335952453315258, -0.0015422474825754762, 0.02111482061445713, 0.025399144738912582, -0.01308307982981205, -0.003352950094267726, -0.0050486004911363125, 0.007308554835617542, -0.011505221016705036, -0.02160790003836155, -0.002118878299370408, 0.019043879583477974, -0.009675342589616776, 0.0312722884118557, 0.017772827297449112, 0.007467436138540506, 0.06982025504112244, -0.04054220765829086, 0.011702453717589378, 0.027218066155910492, 0.011998302303254604, -0.0006283357506617904, -0.0036131872329860926, -0.035545654594898224, -0.03274057060480118, 0.00386794563382864, -0.02686743065714836, 0.008267323486506939, -0.023602139204740524, -0.024785533547401428, -0.003522789105772972, -0.012195534072816372, 0.028861667960882187, 0.02706466242671013, -0.02206810936331749, -0.015329337678849697, -0.022079067304730415, -0.022944698110222816, 0.0007861900958232582, -0.011050490662455559, 0.0006303902482613921, 0.03613734990358353, 0.029409537091851234, 0.02198045141994953, -0.004706183448433876, 0.03707968443632126, -0.0173454899340868, 0.006705900188535452, -0.007056535687297583, 0.020227273926138878, 0.010201295837759972, -0.02572786621749401, -0.021344924345612526, -0.0057471320033073425, -0.006952440831810236, -0.010075286030769348, -0.009916405193507671, -0.01672092080116272, -0.009998585097491741, -0.026144245639443398, -0.02925613336265087, 0.04172560200095177, -0.003289945423603058, 0.0066620707511901855, 0.02820422686636448, -0.034975871443748474, 0.011428519152104855, 0.0018285083351656795, 0.029212303459644318, -0.030746333301067352, 0.035545654594898224, -0.04404855892062187, -0.006623720284551382, 0.05167488008737564, 0.009664385579526424, -0.040805183351039886, -0.05084212124347687, -0.039446473121643066, 0.004577434156090021, -0.023602139204740524, -0.007319511845707893, -0.027042748406529427, 0.02504850924015045, -0.02236395888030529, -0.007231853436678648, 0.033047378063201904, -0.003982997965067625, 0.00862891599535942, -0.012951591983437538, 0.0454949326813221, 0.0625445768237114, 0.020424507558345795, 0.03434034436941147, -0.010113636963069439, -0.047730229794979095, -0.02144354023039341, 0.01918632537126541, 0.004213102161884308, -0.0017285224748775363, 0.02120247855782509, -0.016731878742575645, 0.003851509653031826, 0.010968310758471489, 0.027481041848659515, -0.03780286759138107, -0.008289237506687641, 0.0075496165081858635, -0.006886696442961693, 0.05623313784599304, 0.0008902849513106048, -0.006881217937916517, 0.024259580299258232, -0.05439230054616928, 0.03512927517294884, -0.011384690180420876, -0.033617161214351654, -0.020676525309681892, -0.02081897109746933, -0.040344975888729095, -0.0014518493553623557, -0.016688048839569092, 0.03607160598039627, 0.033726733177900314, -0.03572097048163414, -0.02120247855782509, 0.0019312335643917322, -0.01445274893194437, -0.04632769152522087, -0.013488502241671085, 0.0312722884118557, -0.021870877593755722, 0.008519342169165611, -0.007396213710308075, -0.030329953879117966, 0.026582539081573486, 0.04869448021054268, -0.0108149079605937, -0.012469467706978321, -0.0005793700693175197, 0.02285703830420971, 0.009702736511826515, -0.03585246205329895, 0.048650648444890976, 0.011275116354227066, -0.00020767608657479286, -0.0037912442348897457, 0.03162292018532753, -0.008617958053946495, 0.006431966554373503, -0.01565805822610855, -0.002105181571096182, -0.011395647190511227, 0.002176404232159257, -0.014836256392300129, 0.0071058436296880245, 0.007308554835617542, -0.0011868183501064777, 0.00146554596722126, 0.019558874890208244, 0.03955604508519173, -0.020643653348088264, -0.059213537722826004, -0.05645228549838066, 0.017498893663287163, -0.023010442033410072, -0.01257904153317213, 0.007231853436678648, -0.018627500161528587, -0.029168475419282913, 0.03644415736198425, -0.013872009702026844, -0.015975821763277054, 0.015055403113365173, -0.01229415088891983, 0.023865114897489548, 0.012732444331049919, -0.005955321714282036, 0.02116960659623146, 0.053997837007045746, 0.025530632585287094, 0.03992859646677971, 0.013861051760613918, 0.016819536685943604, 0.004676050506532192, -0.049658726900815964, 0.0358743742108345, -0.018254950642585754, -0.017192088067531586, 0.033858221024274826, 0.024938935413956642, 0.03427460044622421, 0.015395081602036953, -0.002650309819728136, -0.004366505425423384, 0.008245408535003662, 0.0057635679841041565, -0.008212536573410034, 0.005355406552553177, -0.017608467489480972, -0.02785359136760235, 0.01148330606520176, 0.043522607535123825, -0.016534646973013878, -0.022922784090042114, 0.01622783951461315, 0.0052129607647657394, -0.0014778730692341924, -0.0006149814580567181, -0.0307025033980608, -0.01984376646578312, 0.0026489400770515203, 0.019065795466303825, 0.01910962350666523, 0.0016148400027304888, 0.05373486131429672, 0.014463706873357296, -0.038153503090143204, -0.014573279768228531, 0.004544562194496393, -0.027239980176091194, -0.00480479933321476, 0.0007779721054248512, -0.04619620367884636, 0.01171341072767973, -0.018112504854798317, -0.01073272805660963, 0.023645969107747078, 0.02434724010527134, -0.0204683355987072, -0.014387005008757114, 0.03824116289615631, -0.0026407220866531134, 0.020862801000475883, 0.011362775228917599, -0.027239980176091194, 0.016216883435845375, 0.03138186037540436, 0.038854774087667465, -0.00026777031598612666, 0.009763001464307308, -0.025399144738912582, 0.014803384430706501, -0.02265980653464794, -0.009357579983770847, 0.028489118441939354, 0.0062073408626019955, 0.025201912969350815, -0.007944080978631973, -0.003306381404399872, -0.030001234263181686, 0.011757240630686283, 0.026538709178566933, -0.03848222643136978, 0.02179417572915554, 0.003210504539310932, -0.004109007306396961, -0.019416430965065956, -0.012370851822197437, -0.020643653348088264, -0.021531200036406517, -0.0454949326813221, -0.01268861535936594, -0.0007992019527591765, -0.006689464207738638, -0.10185955464839935, 0.01626071333885193, 0.033617161214351654, -0.03604969382286072, 0.010469751432538033, 0.01975610852241516, 0.04873830825090408, 0.03427460044622421, -0.03197355568408966, -0.0046212635934352875, -0.012589999474585056, -0.015734760090708733, -0.001925754826515913, 0.0278755072504282, -0.004065178334712982, 0.04472791776061058, 0.010299911722540855, -0.012874890118837357, -0.008771361783146858, -0.010683419182896614, 0.014978702180087566, -0.0278755072504282, -0.004739055410027504, -0.019526002928614616, -0.05903822183609009, 0.009938319213688374, -0.025968927890062332, 0.03576480224728584, 0.01256808452308178, -0.015942949801683426, -0.010880651883780956, -0.018298780545592308, -0.010683419182896614, -0.000935484073124826, 0.01740027777850628, -0.016699006780982018, -0.012524254620075226, -0.005960800219327211, 0.030198466032743454, -0.016841452568769455, -0.03848222643136978, -0.02572786621749401, -0.04058603569865227, 0.031228456646203995, -0.003865206381306052, -0.007188023999333382, -0.04052029177546501, 0.012403723783791065, 0.034318432211875916, 0.04106815904378891, -0.0003062922623939812, -0.015997735783457756, 0.04082709923386574, 0.030110806226730347, -0.01740027777850628, 0.012370851822197437, -0.02412809245288372, -0.00759344594553113, 0.023032356053590775, -0.04187900573015213, 0.0088206697255373, 0.020172487944364548, -0.023843200877308846, -0.04654683545231819, 0.027590615674853325, -0.014606151729822159, -0.04286516457796097, -0.006481274496763945, 0.0072866398841142654, 0.044114306569099426, 0.026714026927947998, 0.02116960659623146, -0.006333350203931332, 0.005552638787776232, -0.009987627156078815, -0.02638530731201172, -0.01245851069688797, -0.02581552416086197, -0.013882966712117195, 0.020512165501713753, 0.0076811048202216625, 0.0006762741249985993, -0.02019440196454525, -0.010184859856963158, 0.014584237709641457, -0.005207482259720564, 0.004032305907458067, -0.015778588131070137, 0.0202163178473711, -0.015000617131590843, 0.001729892217554152, 0.014255517162382603, -0.012250320985913277, 0.024544471874833107, 0.001427195267751813, 0.006223776843398809, 0.006223776843398809, -0.008957636542618275, -0.017761869356036186, -0.015395081602036953, -0.012140747159719467, 0.04275559261441231, 0.019931426271796227, -0.031688667833805084, -0.034055452793836594, -0.03607160598039627, 0.023799370974302292, 0.016534646973013878, 0.00021675015159416944, 0.0010293064406141639, 0.0036679741460829973, 0.026801686733961105, 0.0008519342518411577, 0.02686743065714836, -0.02122439257800579, -0.016074437648057938, -0.006946961861103773, -0.0010080764768645167, -0.005703302565962076, -0.006563454866409302, 0.016863366588950157, 0.042733676731586456, 0.018912391737103462, -0.008459077216684818, 0.02206810936331749, -0.009828746318817139, 0.0032872059382498264, 0.038284990936517715, -0.00574165303260088, -0.012885847128927708, -0.0233391635119915, 0.0190548375248909, 0.004851368255913258, 0.004413073882460594, 0.00827828049659729, -0.007867380045354366, 0.04168177396059036, 0.009779437445104122, 0.005530724301934242, -0.02366788312792778, 0.031338032335042953, 0.039314985275268555, -0.041265394538640976, 0.009472631849348545, 0.015734760090708733, 0.013696691952645779, -0.02467595972120762, 0.020446421578526497, -0.01631549932062626, 0.01736740581691265, 7.46897712815553e-05, 0.0016175792552530766, 0.011801069602370262, 0.005210221745073795, -0.041572198271751404, 0.050359997898340225, -0.0557071827352047, -0.01694006845355034, 0.05360337346792221, 0.019547918811440468, -0.026889344677329063, -0.008448119275271893, 0.0038925998378545046, 0.040213488042354584, 0.025201912969350815, -0.005207482259720564, -0.02570595033466816, -0.01962462067604065, 0.020205359905958176, 0.014474663883447647, -0.001378572080284357, 0.06425391882658005, -0.02868635207414627, -0.00487876171246171, 0.04065177962183952, 0.013400843366980553, 0.04183517396450043, 0.015910077840089798, -0.014156900346279144, -0.009593162685632706, -0.023273417726159096, 0.02070939727127552, -0.004922591149806976, -0.0026297648437321186, -0.01696198247373104, 0.015011574141681194, 0.036575645208358765, 0.006119681987911463, 0.05811780318617821, -0.02971634268760681, 0.006431966554373503, 0.015504655428230762, 0.005119823385030031, 0.018824733793735504, 0.01655656099319458, 0.01279818918555975, 0.015241678804159164, 0.030855907127261162, 0.011055969633162022, 0.007078450173139572, 0.010404006578028202, -0.005999151151627302, 0.003320078132674098, 0.004284325055778027, -0.009472631849348545, -0.03116271272301674, -0.02008482813835144, -0.029190389439463615, -0.0182111207395792, -0.0065141464583575726, 0.022440658882260323, -0.040323060005903244, 0.04297474026679993, 0.003306381404399872, -0.0019805417396128178, -0.004070656839758158, -0.008119398728013039, 0.028620606288313866, 0.023514481261372566, 0.0029502674005925655, 0.00807556975632906, -0.010371134616434574, -0.004207623656839132, 0.01410211343318224, -0.029058901593089104, 0.008201578631997108, 0.014321261085569859, -0.029409537091851234, -1.9260971839685226e-06, 0.027239980176091194, 0.030001234263181686, 0.05211317166686058, -0.03366098925471306, 0.025574462488293648, -0.019997170194983482, -0.011187457479536533, -0.01456232275813818, 0.0011073775822296739, 0.022484488785266876, -0.03661947324872017, 0.0018654894083738327, 0.004955463111400604, 0.0005173925310373306, 0.017279746010899544, 0.0159319918602705, -0.05097360908985138, -0.0016792144160717726, 0.017575595527887344, 0.013970625586807728, 0.015252635814249516, 7.169361924752593e-05, 0.007308554835617542, -0.05662760138511658, -0.016918154433369637, 0.008426204323768616, 0.01924111321568489, 0.003489917144179344, -0.0007074341410771012, 0.010683419182896614, -0.019142495468258858, 0.0038158982060849667, -0.022462574765086174, 0.012644785456359386, -0.03995050862431526, 0.00038967086584307253, 0.013006378896534443, -0.008415247313678265, 0.009330186061561108, 0.005708781071007252, -0.0022558451164513826, 0.05474293604493141, -0.02706466242671013, 0.011768197640776634, -0.053428053855895996, 0.005812875926494598, 0.023843200877308846, -0.020227273926138878, 0.012622871436178684, 0.0034049975220113993, -0.030417613685131073, -0.023711713030934334, 0.005988193675875664, -0.000982737634330988, 0.06889984011650085, 0.023295333608984947, -0.03973136469721794, -0.006629198789596558, -0.02146545611321926, 0.015132104977965355, -0.02434724010527134, 0.07214321196079254, -0.009812310338020325, 0.00859056506305933, -0.014858171343803406, 0.0066949427127838135, 0.00037460451130755246, 0.007330469321459532, 0.011614794842898846, 0.030636759474873543, -0.0013429606333374977, 0.013751478865742683, 0.007346905302256346, -0.012863933108747005, 0.017323575913906097, 0.01975610852241516, 0.03107505477964878, -0.015975821763277054, -0.037956271320581436, -0.023689797148108482, -0.0015792285557836294, 0.034975871443748474, 0.018068674951791763, 0.010376613587141037, 0.0037090640980750322, -0.0013908990658819675, 0.017224960029125214, -0.011702453717589378, 0.019252069294452667, 0.015154019929468632, -0.026911260560154915, 0.028401460498571396, -0.004508950747549534, 0.043193887919187546, -0.030987394973635674, -0.015329337678849697, -0.018123462796211243, 0.03995050862431526, -0.019931426271796227, -0.004988335072994232, -0.024544471874833107, 0.02846720442175865, 0.02399660460650921, -0.009582205675542355, -0.017378361895680428, -0.016983898356556892, -0.0352826789021492, -0.013367971405386925, 0.02412809245288372, -0.01358711812645197, -0.03166675195097923, 0.019153453409671783, -0.014934872277081013, -0.0006129269604571164, 0.00047287828056141734, 0.008223493583500385, -0.0061909048818051815, -0.029782086610794067, 0.034975871443748474, -0.01788240112364292, -0.0002992727095261216, 0.03596203401684761, -0.0139377536252141, 0.005456761922687292, 0.03650990128517151, 0.007483872584998608, -0.010316348634660244, 0.026209989562630653, -0.026582539081573486, -0.01918632537126541, -0.020566951483488083, -0.025903183966875076, -0.01867133006453514, -0.00415831571444869, -0.0042815860360860825, 0.009494546800851822, 0.0016340153524652123, -0.021191520616412163, -0.017761869356036186, 0.016359329223632812, -0.0023708974476903677, 0.010990225709974766, 0.02098333090543747, -0.015208806842565536, 0.016074437648057938, 0.02342682145535946, -0.010902566835284233, -0.015340294688940048, -0.025464888662099838, 0.02070939727127552, 0.037605635821819305, 0.06328967213630676, -0.02296661213040352, 0.045100465416908264, -0.026692112907767296, -0.0092534851282835, 0.015811460092663765, -0.03219270333647728, 0.030066978186368942, 0.016600390896201134, -0.03195164352655411, 0.014496578834950924, 0.018013888970017433, -0.012272235937416553, 0.037956271320581436, 0.05198168382048607, -0.0018339870730414987, 0.01859462819993496, 0.026209989562630653, 0.0035118318628519773, -0.03832882270216942, -0.028949327766895294, -0.013641905039548874, 0.008760403841733932, -0.01707155629992485, 0.011494264006614685, -0.040695611387491226, -0.008420726284384727, 0.027787847444415092, -0.04258027672767639, 0.003917254041880369, -0.03517310321331024, -0.02592509798705578, 0.02355830930173397, -8.230855746660382e-05, 0.031118882820010185, -0.016896238550543785, -0.012655743397772312, -0.02855486236512661, 0.015921033918857574, -0.020106744021177292, 0.021761303767561913, 0.0049253301694989204, -0.010957353748381138, -0.024434898048639297, 0.04711661860346794, 0.029782086610794067, -0.012129790149629116, 0.0022092764265835285, -0.007368820253759623, 0.004106268286705017, 0.000724554993212223, -0.024150006473064423, 0.02098333090543747, -0.031469520181417465, -0.014474663883447647, -0.012513297609984875, -0.0007841355982236564, 0.008886413648724556, 0.0033940402790904045, 0.031579092144966125, 0.01468285359442234, 0.03922732546925545, -0.01690719649195671, -0.016929110512137413, -0.010360177606344223, 0.020566951483488083, -0.008020782843232155, -0.0119435153901577, 0.018013888970017433, -0.012633828446269035, -0.0014833516906946898, 0.010760121047496796, 0.019504088908433914, -0.0016038826433941722, -0.023361077532172203, -0.005845747888088226, -0.006985312793403864, -0.03291589021682739, 0.010540974326431751, -0.02675785683095455, 0.03153526410460472, 0.019197283312678337, -0.004574695136398077, 0.02296661213040352, 0.011570964939892292, 0.002299674553796649, -0.014551365748047829, -0.03506353124976158, 0.0011381950462237, 0.030549101531505585, -0.002688660519197583, 0.013948710635304451, -0.026692112907767296, 0.03412119671702385, 0.017279746010899544, 0.04073943942785263, 0.017005812376737595, 0.0256182923913002, 0.019142495468258858, -0.018857605755329132, -0.0013552876189351082, -0.011735325679183006, 0.02013961598277092, 0.010299911722540855, 0.032368022948503494, -0.010738206095993519, -0.025399144738912582, 0.008376896381378174, 0.01986568234860897, 0.01366381999105215, 0.0043089790269732475, -0.043763671070337296, 0.003185850568115711, 0.006946961861103773, 0.025749780237674713, 0.018035802990198135, -0.021728431805968285, 0.02695508860051632, -0.03252142667770386, -0.0037748082540929317, 0.02502659521996975, -0.0059279282577335835, -0.02684551663696766, -0.01797005906701088, 0.020336847752332687, 0.03488821163773537, 0.019723236560821533, -0.0025626509450376034, -0.001176545862108469, -0.01077655702829361, -0.0358743742108345, 0.01690719649195671, -0.023054271936416626, -0.019887596368789673, 0.027020832523703575, 0.037956271320581436, 0.013861051760613918, 0.0014682853361591697, -0.018013888970017433, -0.013762435875833035, -0.003670713398605585, 0.03867945820093155, -0.034866299480199814, 0.009368536993861198, 0.02820422686636448, 0.0261661596596241, -0.030768249183893204, 0.03495395556092262, -0.020720355212688446, -0.042952824383974075, -0.007938602939248085, -0.012633828446269035, 0.0021558592561632395, -0.008152270689606667, 0.001824399339966476, 0.0261661596596241, 0.05027233809232712, -0.004508950747549534, -0.0037364575546234846, -0.02458830177783966, 0.01995334029197693, -0.0005307468236424029, 0.02421575039625168, -0.006760687101632357, -0.022900868207216263, 0.0074016922153532505, -0.011395647190511227, -0.02971634268760681, 0.004854107741266489, -0.01574571616947651, -0.04165985807776451, -0.015734760090708733, -0.004287064541131258, 0.0072866398841142654, 0.019887596368789673, 0.018408354371786118, 0.0011450434103608131, 0.02651679515838623, -0.00022924838413018733, 0.029672512784600258, -0.01491295825690031, 0.021816089749336243, -0.019043879583477974, 0.03451566398143768, -0.0027694711461663246, -0.0035693577956408262, -0.00979587435722351, 0.030110806226730347, -0.00030201204936020076, 0.015493697486817837, 0.012787231244146824, -8.132410584948957e-05, -0.0153512516990304, -0.020698441192507744, -0.03666330501437187, -0.004634960554540157, 0.021596943959593773, 0.00792764499783516, 0.024829363450407982, -0.021892791613936424, -0.009971191175282001, 0.010639590211212635, 0.011198415420949459, -0.02030397579073906, 0.016019649803638458, 0.018200164660811424, 0.01780569925904274, -0.006273084785789251, 0.008464555256068707, -0.011395647190511227, -0.021859919652342796, -0.022615976631641388, -0.015811460092663765, 0.01370764896273613, -0.018386438488960266, -0.012261277996003628, 0.017575595527887344, -0.00368167064152658, -0.01234893687069416, -0.004322675988078117, 0.018200164660811424, 0.024829363450407982, -0.005037643015384674, -0.018934305757284164, -0.04122156277298927, 0.014978702180087566, 0.00808652676641941, 0.017575595527887344, 0.029760172590613365, 0.006645634770393372, -0.014945830218493938, -0.005999151151627302, 0.0015449868515133858, -0.018386438488960266, 0.022484488785266876, -0.002862608525902033, -0.03331035375595093, 0.005495112854987383, 0.032718658447265625, -0.026100415736436844, 0.014649981632828712, 0.012820103205740452, 0.0018695984035730362, -0.004251453094184399, -0.010130072943866253, 0.009965713135898113, -0.017389319837093353, 0.014891043305397034, 0.01962462067604065, -0.006607284303754568, -0.0255963783711195, 0.008826147764921188, 0.01742219179868698, -0.00597175769507885, 0.021290138363838196, 0.017772827297449112, -0.03596203401684761, 0.0199204683303833, 0.024895107373595238, 0.002898219972848892, 0.026253819465637207, 0.020293017849326134, 0.010836822912096977, 0.019997170194983482, 0.03609352186322212, -0.021728431805968285, -0.013225525617599487, 0.013981582596898079, -0.0053444490768015385, -0.0375618077814579, -0.014529450796544552, -0.02660445310175419, 0.035896290093660355, 0.0004937657504342496, -0.007505787070840597, -0.022550232708454132, 0.004021348897367716, 0.011214851401746273, 0.007505787070840597, -0.02592509798705578, 0.007538659032434225, 0.001597034279257059, -0.025333400815725327, -0.04628385975956917, 0.021081948652863503, 0.028379544615745544, 0.008952157571911812, 0.04860682040452957, 0.00030423776479437947, 0.01077655702829361, 0.018923349678516388, -0.006399094592779875, 0.011055969633162022, 0.045451100915670395, 0.005295141134411097, -0.03173249587416649, -0.006557975895702839, 0.03219270333647728, 0.01078751403838396, 0.022703636437654495, 0.020314933732151985, 0.0014066502917557955, 0.015614228323101997, -0.012820103205740452, -0.01530742272734642, -0.017783785238862038, -0.02412809245288372, -0.013685734011232853, 0.03324460983276367, 0.016677092760801315, -0.0013648753520101309, 0.012721487320959568, 0.02831380069255829, 0.02087375894188881, -0.018200164660811424, 0.03946838527917862, 0.04150645434856415, -0.009034337475895882, 0.007412649691104889, -0.01563614420592785, 0.01617305353283882, -0.0014134985394775867, -0.03938072919845581, 0.0014134985394775867, 0.021158648654818535, 0.00690861139446497, 0.004881500732153654, 0.010338262654840946, 0.021859919652342796, -0.018879519775509834, -0.022615976631641388, 0.007467436138540506, -0.007483872584998608, -0.01843026839196682, -0.009724651463329792, 0.0181892067193985, 0.0013751478400081396, 0.00960959866642952, 0.007368820253759623, 0.007746848743408918, 0.019120581448078156, 0.0018449444323778152, 0.03508544713258743, -0.011516178026795387, -0.013981582596898079, -0.03096548095345497, -0.04913277179002762, 0.019482174888253212, -0.003024229547008872, 0.0009293205221183598, 0.020260145887732506, -0.039073921740055084, -0.030899737030267715, 0.005464979913085699, 0.0500970184803009, -0.017652295529842377, -0.00944523885846138, -0.0057635679841041565, -0.002884523244574666]' LIMIT 5;
    ================================= Tool Message =================================
    Name: get_recommendation
    
    [["PARFAIT Plus Size Blue Solid Underwired Lightly Padded Plunge Bra 2801", "PARFAIT", "Women", 1300, "A nurturing and supportive product designed to provide comfort and style, perfect for enhancing your wardrobe."], ["HIGHLANDER Men Mustard & Black Slim Fit Checked Casual Shirt", "HIGHLANDER", "Men", 749, "A stylish casual shirt that reflects a thoughtful and introspective approach to fashion, perfect for deep conversations and meaningful experiences."], ["Difference of Opinion Men Grey Melange Printed Round Neck T-shirt", "Difference of Opinion", "Men", 599, "This practical and stylish Grey printed T-shirt features a round neck and long sleeves, perfect for resourceful individuals who appreciate comfort and quality."], ["Gini and Jony Girls Blue Solid Jacket", "Gini and Jony", "Girls", 1300, "A delightful blue solid jacket that brings joy to any occasion, featuring a spread collar and four pockets for your essentials. Perfect for girls who love style and comfort."], ["AIGNER Men Platinum Eau De Toilette Perfume 100 ml", "AIGNER", "Men", 4000, "This product is designed with a pragmatic approach, ensuring a sophisticated fragrance experience suitable for any occasion."]]
    ================================== Ai Message ==================================
    
    Here are some item recommendations suitable for sporty activities:
    
    1. **PARFAIT Plus Size Blue Solid Underwired Lightly Padded Plunge Bra 2801**
       - **Brand:** PARFAIT
       - **Price:** ₹1300
       - **Description:** A nurturing and supportive product designed to provide comfort and style, perfect for enhancing your wardrobe.
    
    2. **HIGHLANDER Men Mustard & Black Slim Fit Checked Casual Shirt**
       - **Brand:** HIGHLANDER
       - **Price:** ₹749
       - **Description:** A stylish casual shirt that reflects a thoughtful and introspective approach to fashion, perfect for deep conversations and meaningful experiences.
    
    3. **Difference of Opinion Men Grey Melange Printed Round Neck T-shirt**
       - **Brand:** Difference of Opinion
       - **Price:** ₹599
       - **Description:** This practical and stylish grey printed T-shirt features a round neck and long sleeves, perfect for resourceful individuals who appreciate comfort and quality.
    
    4. **Gini and Jony Girls Blue Solid Jacket**
       - **Brand:** Gini and Jony
       - **Price:** ₹1300
       - **Description:** A delightful blue solid jacket that brings joy to any occasion, featuring a spread collar and four pockets for your essentials. Perfect for girls who love style and comfort.
    
    5. **AIGNER Men Platinum Eau De Toilette Perfume 100 ml**
       - **Brand:** AIGNER
       - **Price:** ₹4000
       - **Description:** This product is designed with a pragmatic approach, ensuring a sophisticated fragrance experience suitable for any occasion.
    
    Let me know if you'd like to purchase any of these items or need further assistance!
</pre>

```python
cursor.close()
conn.close()
```

With this, our shopping mall system is now complete.  
Although it is, in fact, almost incomplete.  
And I realized while making this tutorial that the example I created was not very suitable for applying the concept of multi-agent.

Still, I believe you were able to grasp the concept of how multi-agents are structured   
and how they operate by following this tutorial.

By using the concepts you have learned, I believe you will be able to create more suitable and impressive services!

Thank you for your hard work this time as well! Thank you.
