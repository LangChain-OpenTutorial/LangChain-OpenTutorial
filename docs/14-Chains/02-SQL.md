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

# SQL

- Author: [Jinu Cho](https://github.com/jinucho)
- Peer Review: 
- Proofread:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/14-Chains/02-SQL.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/14-Chains/02-SQL.ipynb)

## Overview

This tutorial covers how to use ```create_sql_query_chain``` to generate SQL queries, execute them, and derive answers. 

Additionally, let's explore the differences in operation between this method and the SQL Agent.

![sql-chain-work-flow](./img/02-sql-sql-chain-work-flow.png)

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Load SQL Database](#load-sql-database)
- [SQL generate chain](#sql-generate-chain)
- [Using SQL generating chain with an Agent](#using-sql-generating-chain-with-an-agent)
- [Appendix : Chain with gpt-4o and a Post-Processing Function](#appendix--chain-with-gpt-4o-and-a-post-processing-function)                                                                

### References
- [SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html#sqldatabase)
- [SQL_query_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sql_database.query.create_sql_query_chain.html)
- [SQL_agent](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.base.create_sql_agent.html)
---

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
        "langchain_openai",
        "langchain_community",
    ],
    verbose=False,
    upgrade=False,
)
```

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "02-SQL",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
# Load environment variables
# Reload any variables that need to be overwritten from the previous cell

from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Load SQL Database

### Usage methods for various databases and required library list.

| **Database**        | **Required Library**      | **Code Example**                                                                                                                                    |
|---------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **PostgreSQL**      | ```psycopg2-binary```         | db = SQLDatabase.from_uri("postgresql://<username>:<password>@<host>:<port>/<database>")                                                          |
| **MySQL**           | ```pymysql```                | db = SQLDatabase.from_uri("mysql+pymysql://<username>:<password>@<host>:<port>/<database>")                                                       |
| **SQLite**          | Included in standard lib | db = SQLDatabase.from_uri("sqlite:///path/to/your_database.db")                                                                                   |
| **Oracle**          | ```cx_Oracle```              | db = SQLDatabase.from_uri("oracle+cx_oracle://<username>:<password>@<host>:<port>/<sid>")                                                         |

example for postgresql : 
- db = SQLDatabase.from_uri("postgresql://postgre_user_name:password@ip_address:port/db_name")

Load and verify the sample database data.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# Connect to the SQLite database.
db = SQLDatabase.from_uri("sqlite:///data/finance.db")

# Output the database dialect.
print(db.dialect)

# Output the available table names.
print(db.get_usable_table_names())
```

<pre class="custom">sqlite
    ['accounts', 'customers', 'transactions']
</pre>

## SQL generate chain

```create_sql_query_chain``` generates a chain for creating SQL queries based on natural language input. 

It leverages LLMs to translate natural language into SQL statements.

[RECOMMED] Create an LLM object and generate a chain by providing the LLM and DB as parameters.

Since changing the model may cause unexpected behavior, this tutorial will proceed with **gpt-3.5-turbo** .

```python
# Create an OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Generate a chain by providing the LLM and DB as parameters.
chain = create_sql_query_chain(
    llm=llm, db=db, k=10
)  # k(for query Limit)'s default value is 5
```

```python
chain.invoke({"question": "List the all customer names."})
```




<pre class="custom">'SELECT "name" FROM customers;'</pre>



### If the latest version is used?

Using the latest version of OpenAI's LLM may cause issues with the output.

```python
# Create an OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Generate a chain by providing the LLM and DB as parameters.
bad_case_chain = create_sql_query_chain(
    llm=llm, db=db, k=10
)  # k(for query Limit)'s default values is 5
```

Unnecessary information, such as **'SQLQuery: '** , is included in the output along with the query.

```python
bad_case_chain.invoke({"question": "List the all customer names."})
```




<pre class="custom">'SQLQuery: SELECT "name" FROM customers LIMIT 10;'</pre>



(Optional) You can specify the prompt directly using the method below.

When writing it yourself, you can include **table_info** along with descriptive **column descriptions** for better explanation.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
    Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. 
    You can order the results by a relevant column to return the most interesting examples in the database.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Here is the description of the columns in the tables:
`cust`: customer name
`prod`: product name
`trans`: transaction date

Question: {input}
"""
).partial(dialect=db.dialect)

# Create an OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Generate a chain by providing the LLM and DB as parameters.
chain = create_sql_query_chain(llm, db, prompt)
```

Executing the chain generates queries based on the database.

```python
# Execute the chain and display the results.
generated_sql_query = chain.invoke({"question": "List the all customer names."})

# Print the generated query.
print(generated_sql_query.__repr__())
```

<pre class="custom">'SELECT name\nFROM customers'
</pre>

### How to use the ```get_prompts``` method

The chain.get_prompt() method allows you to retrieve the current prompt template used in a LangChain chain. 

This prompt contains the instructions given to the LLM, including the input structure, expected variables, and contextual guidelines.

**Key Features**
1. Prompt Retrieval:
- Fetches the active prompt template to inspect or debug the chain's behavior.
2. Dynamic Variable Substitution:
- Displays how variables are dynamically substituted within the template.
3. Customizability:
- Enables users to modify parts of the prompt dynamically.


check the .get_prompts()'s contents

There are various elements:  
- ```input_variables```
- ```input_types```
- ```partial_variables```
- ```template```

```python
# check the prompt template configuration
print(f"input_variables : {chain.get_prompts()[0].input_variables}", "\n")
print(f"input_types : {chain.get_prompts()[0].input_types}", "\n")
print(f"partial_variables : {chain.get_prompts()[0].partial_variables}", "\n")
print(f"template : {chain.get_prompts()[0].template}", "\n")
```

<pre class="custom">input_variables : ['input', 'table_info'] 
    
    input_types : {} 
    
    partial_variables : {'dialect': 'sqlite', 'top_k': '5'} 
    
    template : Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
        Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. 
        You can order the results by a relevant column to return the most interesting examples in the database.
    
    Use the following format:
    
    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    Only use the following tables:
    {table_info}
    
    Here is the description of the columns in the tables:
    `cust`: customer name
    `prod`: product name
    `trans`: transaction date
    
    Question: {input}
     
    
</pre>

Modify the variable values and check the results.

```python
# Modify the dialect to MySQL
chain.get_prompts()[0].partial_variables["dialect"] = "my_sql"

# check the modified prompt
print(f"input_variables : {chain.get_prompts()[0].input_variables}", "\n")
print(f"input_types : {chain.get_prompts()[0].input_types}", "\n")
print(f"partial_variables : {chain.get_prompts()[0].partial_variables}", "\n")
print(f"template : {chain.get_prompts()[0].template}", "\n")
```

<pre class="custom">input_variables : ['input', 'table_info'] 
    
    input_types : {} 
    
    partial_variables : {'dialect': 'my_sql', 'top_k': '5'} 
    
    template : Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
        Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. 
        You can order the results by a relevant column to return the most interesting examples in the database.
    
    Use the following format:
    
    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    Only use the following tables:
    {table_info}
    
    Here is the description of the columns in the tables:
    `cust`: customer name
    `prod`: product name
    `trans`: transaction date
    
    Question: {input}
     
    
</pre>

You can specify variables, including `dialect`, when invoking `create_sql_query_chain`.

```python
chain.invoke({"question": "List all customer names.", "dialect": "mysql"})
```




<pre class="custom">'SELECT name\nFROM customers;'</pre>



### QuerySQLDatabaseTool

1. Executing SQL Queries:
- Executes the provided SQL query on the connected database and retrieves the results.
- Encapsulates the functionality for interacting with the database, promoting code reusability.
2. Integration with LangChain Agents:
- LangChain agents are used to convert natural language into SQL queries.
- This tool performs the execution step by running the agent-generated SQL query and returning the results.
3. Database Abstraction:
- Supports various types of databases, such as MySQL, PostgreSQL, SQLite, and more.
- Handles direct database operations internally, reducing the dependency of user code on database-specific details.

Let's verify if the generated query executes correctly.

```python
from langchain_community.tools import QuerySQLDatabaseTool

# Create a tool to execute the generated query.
execute_query = QuerySQLDatabaseTool(db=db)
```

```python
execute_query.invoke({"query": generated_sql_query})
```




<pre class="custom">"[('Altman',), ('Huang',), ('Zuckerberg',), ('Musk',), ('Hassabis',), ('Chase',)]"</pre>



```python
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

# Tool
execute_query = QuerySQLDatabaseTool(db=db)

# SQL query generation chain
write_query = create_sql_query_chain(llm, db, prompt)

# Create a chain to execute the generated query.
chain = write_query | execute_query
```

```python
# Check the execution result
chain.invoke({"question": "Retrieve Altman's email address."})
```




<pre class="custom">"[('Sam@example.com',)]"</pre>



### Enhance and generate answers using the LLM

Using the chain created in the previous step results in short, concise answers. This can be adjusted using an LCEL-style chain to provide more natural and detailed responses.

```python
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Define the prompt for generating answers
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Create a pipeline for generating natural answers
answer = answer_prompt | llm | StrOutputParser()

# Create a chain to execute the generated query and produce an answer
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)
```

```python
# Check the execution result
chain.invoke({"question": "Calculate the total of Altman's transactions."})
```




<pre class="custom">"The total of Altman's transactions is -965.7."</pre>



## Using SQL generating chain with an Agent

### What is ```agent_toolkits``` ?

```agent_toolkits``` in LangChain is a collection of tools designed to simplify the creation and use of Agents optimized for specific domains or use cases. 

Each toolkit encapsulates the functionality and workflows needed for specific tasks (e.g., SQL query processing, file system operations, API calls), enabling developers to perform complex tasks with ease.

### ```create_sql_agent```

```create_sql_agent``` is a specialized function within the agent_toolkits library that simplifies the process of interacting with SQL databases. 

It is designed to streamline SQL query generation and execution by leveraging LangChain’s agent capabilities. Developers can integrate this tool to enable agents to:
- Connect to SQL databases seamlessly.
- Automatically generate SQL queries based on natural language input.
- Retrieve and format results for easy consumption.

This functionality is particularly useful for scenarios requiring dynamic database interactions, such as reporting, analytics, or user-facing applications that need query-based responses.

Using an Agent, you can generate SQL queries and output the results as answers.

Agents work well with models like **gpt-4o** and **gpt-4o-mini**, in contrast to the issues encountered with chains when changing the model.

```python
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Connect to the SQLite database
db = SQLDatabase.from_uri("sqlite:///data/finance.db")

# Create the Agent
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
```

```python
# Check the execution result
agent_executor.invoke(
    {"input": "Calculate and compare the total transactions of Altman and Zuckerberg."}
)
```

<pre class="custom">
    
    [1m> Entering new SQL Agent Executor chain...[0m
    [32;1m[1;3m
    Invoking: `sql_db_list_tables` with `{}`
    
    
    [0m[38;5;200m[1;3maccounts, customers, transactions[0m[32;1m[1;3m
    Invoking: `sql_db_schema` with `{'table_names': 'accounts'}`
    
    
    [0m[33;1m[1;3m
    CREATE TABLE accounts (
    	account_id INTEGER, 
    	customer_id INTEGER, 
    	balance REAL, 
    	PRIMARY KEY (account_id), 
    	FOREIGN KEY(customer_id) REFERENCES customers (customer_id)
    )
    
    /*
    3 rows from accounts table:
    account_id	customer_id	balance
    1	1	1000.5
    2	2	2500.75
    3	3	1500.0
    */[0m[32;1m[1;3m
    Invoking: `sql_db_schema` with `{'table_names': 'customers'}`
    
    
    [0m[33;1m[1;3m
    CREATE TABLE customers (
    	customer_id INTEGER, 
    	name TEXT, 
    	age INTEGER, 
    	email TEXT
    )
    
    /*
    3 rows from customers table:
    customer_id	name	age	email
    1	Altman	40	Sam@example.com
    2	Huang	62	Jensen@example.com
    3	Zuckerberg	41	Mark@example.com
    */[0m[32;1m[1;3m
    Invoking: `sql_db_schema` with `{'table_names': 'transactions'}`
    
    
    [0m[33;1m[1;3m
    CREATE TABLE transactions (
    	transaction_id INTEGER, 
    	account_id INTEGER, 
    	amount REAL, 
    	transaction_date TEXT, 
    	PRIMARY KEY (transaction_id), 
    	FOREIGN KEY(account_id) REFERENCES accounts (account_id)
    )
    
    /*
    3 rows from transactions table:
    transaction_id	account_id	amount	transaction_date
    1	1	74.79	2024-07-13
    2	1	-224.1	2024-05-13
    3	1	-128.9	2024-01-25
    */[0m[32;1m[1;3m
    Invoking: `sql_db_query_checker` with `{'query': "SELECT customer_id, name FROM customers WHERE name IN ('Altman', 'Zuckerberg')"}`
    responded: The relevant tables and their structures are as follows:
    
    1. **accounts**: 
       - `account_id`: INTEGER
       - `customer_id`: INTEGER
       - `balance`: REAL
    
    2. **customers**: 
       - `customer_id`: INTEGER
       - `name`: TEXT
       - `age`: INTEGER
       - `email`: TEXT
    
    3. **transactions**: 
       - `transaction_id`: INTEGER
       - `account_id`: INTEGER
       - `amount`: REAL
       - `transaction_date`: TEXT
    
    To calculate and compare the total transactions of Altman and Zuckerberg, I will need to:
    1. Find the `customer_id` for both Altman and Zuckerberg from the `customers` table.
    2. Use their `customer_id` to find their corresponding `account_id` from the `accounts` table.
    3. Sum the `amount` from the `transactions` table for each `account_id`.
    
    Let's construct the SQL queries for these steps. 
    
    First, I will get the `customer_id` for Altman and Zuckerberg. Then, I will get their `account_id` and finally sum their transactions. 
    
    I will start by querying the `customer_id` for both customers.
    
    [0m[36;1m[1;3m```sql
    SELECT customer_id, name FROM customers WHERE name IN ('Altman', 'Zuckerberg')
    ```[0m[32;1m[1;3m
    Invoking: `sql_db_query` with `{'query': "SELECT customer_id, name FROM customers WHERE name IN ('Altman', 'Zuckerberg')"}`
    
    
    [0m[36;1m[1;3m[(1, 'Altman'), (3, 'Zuckerberg')][0m[32;1m[1;3m
    Invoking: `sql_db_query_checker` with `{'query': 'SELECT account_id FROM accounts WHERE customer_id IN (1, 3)'}`
    
    
    [0m[36;1m[1;3m```sql
    SELECT account_id FROM accounts WHERE customer_id IN (1, 3)
    ```[0m[32;1m[1;3m
    Invoking: `sql_db_query` with `{'query': 'SELECT account_id FROM accounts WHERE customer_id IN (1, 3)'}`
    
    
    [0m[36;1m[1;3m[(1,), (3,)][0m[32;1m[1;3m
    Invoking: `sql_db_query_checker` with `{'query': 'SELECT SUM(amount) FROM transactions WHERE account_id IN (1, 3) GROUP BY account_id'}`
    
    
    [0m[36;1m[1;3m```sql
    SELECT SUM(amount) FROM transactions WHERE account_id IN (1, 3) GROUP BY account_id
    ```[0m[32;1m[1;3m
    Invoking: `sql_db_query` with `{'query': 'SELECT SUM(amount) FROM transactions WHERE account_id IN (1, 3) GROUP BY account_id'}`
    
    
    [0m[36;1m[1;3m[(-965.7,), (656.6400000000002,)][0m[32;1m[1;3mThe total transactions for each customer are as follows:
    
    - **Altman** (account_id 1): Total transactions amount to **-965.7**.
    - **Zuckerberg** (account_id 3): Total transactions amount to **656.64**.
    
    In summary, Zuckerberg has a positive total transaction amount, while Altman has a negative total transaction amount.[0m
    
    [1m> Finished chain.[0m
</pre>




    {'input': 'Calculate and compare the total transactions of Altman and Zuckerberg.',
     'output': 'The total transactions for each customer are as follows:\n\n- **Altman** (account_id 1): Total transactions amount to **-965.7**.\n- **Zuckerberg** (account_id 3): Total transactions amount to **656.64**.\n\nIn summary, Zuckerberg has a positive total transaction amount, while Altman has a negative total transaction amount.'}



### Differences Between create_sql_query_chain and SQL Agent
1. create_sql_query_chain:
    - Translates user input into a single SQL query and executes it directly.
    - Best for simple, direct query execution.
2. SQL Agent:
    - Handles more complex workflows, involving multiple queries and reasoning steps.
    - Ideal for dynamic or multi-step tasks.
3. Conclusion: It is recommended to use ```create_sql_query_chain``` for simple queries, while ```SQL Agent``` is suggested for complex or iterative processes.

---

## Appendix : Chain with gpt-4o and a Post-Processing Function

As observed earlier, `gpt-4o` output can be inconsistent.

To improve this, a chain can be constructed with a post-processing function applied.

This part consists of the following procedure:
-   Natural language question input → SQL query generation by LLM → Post-processing of the generated query → Query execution on the database → Natural language answer output by LLM

Load and verify the sample database data.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase

# Connect to the SQLite database.
db = SQLDatabase.from_uri("sqlite:///data/finance.db")

# Output the database dialect.
print(db.dialect)

# Output the available table names.
print(db.get_usable_table_names())
```

<pre class="custom">sqlite
    ['accounts', 'customers', 'transactions']
</pre>

Create an LLM object and use a custom template to generate a chain by providing the LLM and DB as parameters.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
    Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. 
    You can order the results by a relevant column to return the most interesting examples in the database.

[Important] You must respond strictly in the format 'select column_name from table_name [Options:LIMIT,ORDER BY,others]'.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Here is the description of the columns in the tables:
`cust`: customer name
`prod`: product name
`trans`: transaction date

Question: {input}
"""
).partial(dialect=db.dialect)

# Create an OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Generate a chain by providing the LLM and DB as parameters.
write_query = create_sql_query_chain(llm, db, prompt, k=10)
```

Check the execution result

```python
# The result contains unnecessary text, such as 'SQLQuery: '
write_query.invoke({"question": "List the all customer names."})
```




<pre class="custom">'SQLQuery: select name from customers ORDER BY name LIMIT 10'</pre>



Define a function for post-processing the SQL query.

```python
# It is necessary to remove unnecessary text.
import re


# Define the regex parsing function
def parse_sqlquery(query):
    match = re.search(r"SQLQuery:\s*(.*)", query)
    if match:
        return match.group(1).strip()
    else:
        return query
```

Create a tool to execute the generated query.

```python
from langchain_community.tools import QuerySQLDatabaseTool

execute_query = QuerySQLDatabaseTool(db=db)
```

Combine the chains: SQL query generation → Post-processing of the generated query → Query execution on the database

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        query=lambda x: parse_sqlquery(x["query"])
    )
    | execute_query
)
```

```python
# Check the execution result
chain.invoke({"question": "List the all customer names."})
```




<pre class="custom">"[('Altman',), ('Chase',), ('Hassabis',), ('Huang',), ('Musk',), ('Zuckerberg',)]"</pre>



```python
# Check the execution result
chain.invoke({"question": "Calculate the total of Altman's transactions."})
```




<pre class="custom">'[(-965.7,)]'</pre>



Combine the chains: SQL query generation → Post-processing of the generated query → Query execution on the database → Natural language answer output by LLM

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Define the prompt for generating answers
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}ㅑㅑ
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Create a pipeline for generating natural answers
answer = answer_prompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        query=lambda x: parse_sqlquery(x["query"])
    )
    | RunnablePassthrough.assign(result=execute_query)
    | answer
)
```

```python
# Check the execution result
chain.invoke({"question": "List the all customer names."})
```




<pre class="custom">'The customer names are: Altman, Chase, Hassabis, Huang, Musk, and Zuckerberg.'</pre>



```python
# Check the execution result
chain.invoke({"question": "Calculate the total of Altman's transactions."})
```




<pre class="custom">"The total of Altman's transactions is -965.7."</pre>


