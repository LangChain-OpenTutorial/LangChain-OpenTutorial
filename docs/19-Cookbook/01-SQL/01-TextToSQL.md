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

# TextToSQL

- Author: [Jaehun Choi](https://github.com/ash-hun)
- Design: 
- Peer Review: [Dooil Kwak](https://github.com/back2zion), [Ilgyun Jeong](https://github.com/johnny9210)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/99-TEMPLATE/00-BASE-TEMPLATE-EXAMPLE.ipynb)

## Overview

This tutorial provides a feature for generating SQL query statements based on arbitrary database information. While it does not cover extracting database information directly, it may include details about column information and descriptions for specific tables. Using OpenAI’s GPT models (e.g., gpt-4o) and prompt templates, the tutorial demonstrates how to generate SQL queries.

**Features**

- Database Information : Introduces the format of database information required for generating SQL queries.
- TextToSQL : Generates customized SQL queries based on the provided database information.
- Evaluation : Conducts a lightweight evaluation of the generated SQL queries.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Database Information](#database-information)
- [TextToSQL](#text-to-sql)
- [Evaluation](#evaluation)


### References

- [JsonOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.json.JsonOutputParser.html)
- [arXiv : Enhancing Text-to-SQL Translation for Financial System Design](https://arxiv.org/abs/2312.14725)
- [Github : SQAM](https://github.com/ezzini/SQAM)
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
        "LANGCHAIN_PROJECT": "01-TextToSQL",
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




<pre class="custom">True</pre>



## Database Information

Descriptions of database information can be utilized in two main forms:

- Providing raw table or column information without modification.
- Providing table or column information with descriptions included.


This tutorial will proceed based on the following example table:

- Database Name: CompanyDB
- Column Information for the 'employees' table (Employee Information):

    ```
    id (INT, PRIMARY KEY, AUTO_INCREMENT)
    name (VARCHAR, Employee Name)
    position (VARCHAR, Position)
    department (VARCHAR, Department Name)
    salary (DECIMAL, Salary)
    hire_date (DATE, Hire Date)
    departments (VARCHAR, Department Information)
    ```


```python
# Providing raw table or column information without modification.

db_schema = """
employees table
- id (INT, PRIMARY KEY, AUTO_INCREMENT)
- name (VARCHAR, Employee Name)
- position (VARCHAR, Position)
- department (VARCHAR, Department Name)
- salary (DECIMAL, Salary)
- hire_date (DATE, Hire Date)
- departments (VARCHAR, Department Information)
"""
```

```python
# Providing table or column information with descriptions included.

db_schema_description = """
The employees table stores information about the employees in the organization. It includes the following fields:

- id: An integer that serves as the primary key and is auto-incremented for each employee.
- name: A string (VARCHAR) representing the name of the employee.
- position: A string (VARCHAR) indicating the job title or position of the employee.
- department: A string (VARCHAR) specifying the department to which the employee belongs.
- salary: A decimal value representing the employee's salary.
- hire_date: A date field indicating when the employee was hired.

"""
```

## Text to SQL

Customized SQL queries are generated based on the two types of database schema information mentioned above.

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Define Datamodel with Pydantic
class SqlSchema(BaseModel):
    statement: str = Field(description="SQL Query Statement")
```

```python
# Define Common function for Inference
def generate(datamodel:BaseModel, database_schema:str, user_question:str) -> str:
    # Create an OpenAI object
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    # Set up the parser and inject the instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=datamodel)
    
    # Set up the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a expert of SQL. Answer questions concisely."),
            ("user", "Please generate a direct and accurate SQL Query statement from the Schema_Info.\n\n#Format: {format_instructions}\n\n#Schema_Info: {schema_info}\n\n#Question: {question}"),
        ]
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    # Combine the prompt, model, and JsonOutputParser into a chain
    chain = prompt | model | parser

    # Run the chain with your question : raw style prompt
    answer = chain.invoke({"schema_info": database_schema, "question": user_question})
    return answer['statement']
```

```python
# Generate SQL Query with both raw style and description style prompts
question = "Please show me the names and job titles of all employees in the Engineering department."

print("Raw style prompt result:")
print(generate(datamodel=SqlSchema, database_schema=db_schema, user_question=question))

print("\nDescription style prompt result:")
print(generate(datamodel=SqlSchema, database_schema=db_schema_description, user_question=question))
```

<pre class="custom">Raw style prompt result:
    SELECT name, position FROM employees WHERE department = 'Engineering';
    
    Description style prompt result:
    SELECT name, position FROM employees WHERE department = 'Engineering';
</pre>

Additionally, cases involving the generation of SQL queries referencing two or more tables are also introduced.

```python
# Generating SQL Queries Using Multiple Database Schemas

db_multi_schema = """
employees table
- id (INT, PRIMARY KEY, AUTO_INCREMENT)
- name (VARCHAR, Employee Name)
- position (VARCHAR, Position)
- hire_date (DATE, Hire Date)

departments table
- id (INT, PRIMARY KEY, AUTO_INCREMENT)
- name (VARCHAR, Department Name)
- salary (DECIMAL, Salary of Year)
- manager_id (INT, Foreign KEY: employees table's id column)
"""

# Run the chain with your question : Multi Schema Information prompt
multi_table_question = "Please generate a query to calculate the average salary for each department."
generate(datamodel=SqlSchema, database_schema=db_multi_schema, user_question=multi_table_question)
```




<pre class="custom">'SELECT departments.name AS department_name, AVG(departments.salary) AS average_salary FROM departments GROUP BY departments.name;'</pre>



## Evaluation

As detailed in the paper "[Enhancing Text-to-SQL Translation for Financial System Design](https://arxiv.org/abs/2312.14725)" referenced in the References section, SQL evaluation cannot be assessed using a single metric alone. In this tutorial, we utilize code excerpted from the [SQAM GitHub repository](https://github.com/ezzini/SQAM) for evaluation purposes, selected from among various evaluation metrics. For more information, please refer to the original paper linked in the References section.

**[Note]**  
The Structural Query Alignment Metric (SQAM) is a Python package that provides functions to compare SQL queries based on their syntax and structure. Given a query and a ground truth query, the package computes an accuracy score that reflects the degree of similarity between the two queries. The accuracy score is based on the percentage of matching query subitems (e.g., select columns, where conditions, order by clauses) weighted by their importance in the overall query structure.

The evaluation will proceed in the following order.

1. Comparison of Query Components : Divide the SQL queries into major components such as SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, and LIMIT. Extract detailed elements within each component (e.g., selected columns, conditions, sorting criteria, etc.).

2. Weight Assignment to Components : Assign weights to each component based on its importance in the query structure. For example, SELECT and WHERE may have relatively higher weights, while FROM and ORDER BY may have lower weights.

3. Accuracy Calculation : Compare the components of the query being evaluated with the reference query, calculating how many detailed elements (sub-components) match between them. Then compute the ratio of matching elements to the total compared elements and return the accuracy as a percentage (%).


```python
import re

def split_sql_query(query):
    query = query.replace(';','').replace('select ','SELECT ').strip()
    for keyword in ['from','where','group by','having','order by','limit']:
      query = query.replace(' '+keyword+' ',' '+keyword.upper()+' ')

    # extract SELECT statement
    select_end = query.find(' FROM ')
    select_clause = query[:select_end] if select_end != -1 else query
    select_items = [item.strip().split()[-1].split(".")[-1].lower() for item in select_clause.split('SELECT ')[-1].split(',') if item.strip()]

    # extract FROM statement
    from_start = select_end + 6 if select_end != -1 else 0
    from_end = query.find(' WHERE ') if ' WHERE ' in query else len(query)
    from_clause = query[from_start:from_end].strip()
    if from_start>=from_end:
        from_items=['']
    else:
        from_items = [item.strip().split()[0].lower() for item in from_clause.split('JOIN') if item.strip()]

    # extract WHERE conditions
    where_start = from_end + 7 if ' WHERE ' in query else len(query)
    where_end = query.find(' GROUP BY ') if ' GROUP BY ' in query else len(query)
    where_clause = query[where_start:where_end].strip()
    if where_start>=where_end:
        where_items=['']
    else:
        where_items = [re.sub('[' +  ''.join(['\'',' ','"']) +  ']', '', item).lower().split('.')[-1] for item in re.split(r'\s+(?:AND|OR)\s+', where_clause, flags=re.IGNORECASE) if item.strip()] if where_clause != '' else None

    # extract GROUP BY statement
    group_start = where_end + 10 if ' GROUP BY ' in query else len(query)
    group_end = query.find(' HAVING ') if ' HAVING ' in query else len(query)
    group_clause = query[group_start:group_end].strip()
    if group_start>=group_end:
        group_items=['']
    else:
        group_items = [item.strip().lower() for item in group_clause.split(',') if item.strip()] if group_clause != '' else None

    # extract HAVING conditions
    having_start = group_end + 8 if ' HAVING ' in query else len(query)
    having_end = query.find(' ORDER BY ') if ' ORDER BY ' in query else len(query)
    having_clause = query[having_start:having_end].strip()
    if having_start>=having_end:
        having_items=['']
    else:
        having_items = [item.strip().lower() for item in re.split(r'\s+(?:AND|OR)\s+', having_clause, flags=re.IGNORECASE) if item.strip()] if having_clause != '' else None

    # extract ORDER BY statement
    order_start = having_end + 10 if ' ORDER BY ' in query else len(query)
    order_end = len(query)
    order_clause = query[order_start:order_end].strip()
    if order_start>=order_end:
        order_items=['']
    else:
        order_items = [item.strip().lower() for item in order_clause.split(',') if item.strip()] if order_clause != '' else None

    # extract LIMIT number
    limit_start = query.find(' LIMIT ') + 7 if ' LIMIT ' in query else len(query)
    limit_clause = query[limit_start:].strip()
    limit_number = int(limit_clause) if limit_clause.isdigit() else None

    # return dictionary of subitems
    return {'SELECT': select_items, 'FROM': from_items, 'WHERE': where_items, 
            'GROUP BY': group_items, 'HAVING': having_items, 'ORDER BY': order_items, 'LIMIT': [limit_number]}

def sql_query_accuracy(query, true_query):
    # split the queries into parts using the updated split_sql_query function
    query_parts = split_sql_query(query)
    true_query_parts = split_sql_query(true_query)

    # define the weights for each main query part
    weights = {'SELECT': 2, 'FROM': 1, 'WHERE': 3, 'GROUP BY': 2, 'HAVING': 2, 'ORDER BY': 1, 'LIMIT': 2}

    # initialize the total and matching subitems counts
    total_count = 0
    matching_count = 0

    # iterate over the query parts and compare them with the true query parts
    for part_name, part_list in query_parts.items():
        true_part_list = true_query_parts.get(part_name, [])

        # calculate the weight for the current part
        weight = weights.get(part_name, 1)

        # skip the loop iteration if the part_list is None
        if part_list is None:
          if true_part_list is None:
            continue
          else:
            total_count += weight
            continue
        elif true_part_list is None:
          total_count += weight
          continue

        # iterate over the subitems in the query part and compare them with the true query part
        for subitem in set(part_list).union(set(true_part_list)):
            total_count += weight
            if subitem in true_part_list and subitem in part_list:
                matching_count += weight

    # calculate the accuracy score as the percentage of matching subitems
    if total_count == 0:
        accuracy_score = 0
    else:
        accuracy_score = matching_count / total_count * 100

    return accuracy_score
```

**Evaluation Case #1** : This refers to cases where the same columns and conditions are used, producing identical execution results, but the query expressions differ. A key characteristic is the inclusion of aliases, which do not affect the evaluation.

```python
# Evaluation Case #1
sql1 = "SELECT name, age as AGE FROM users WHERE AGE > 20"
sql2 = "SELECT age, name as NAME FROM users WHERE age > 20"

accuracy = sql_query_accuracy(sql1, sql2)
print(f"Accuracy score: {accuracy:.2f}%")
```

<pre class="custom">Accuracy score: 100.00%
</pre>

**Evaluation Case #2** : This refers to cases where the same columns are used but with different conditions, resulting in variations in execution outcomes. Due to the nature of the evaluation algorithm, differences in the WHERE clause, which pertains to conditions within SELECT, FROM, and WHERE, lead to an inconsistency rate of approximately 33.3%.

```python
# Evaluation Case #2
sql3 = "SELECT name, age FROM users WHERE age = 20"
sql4 = "SELECT name, age FROM users WHERE age > 20"

accuracy = sql_query_accuracy(sql3, sql4)
print(f"Accuracy score: {accuracy:.2f}%")
```

<pre class="custom">Accuracy score: 66.67%
</pre>
