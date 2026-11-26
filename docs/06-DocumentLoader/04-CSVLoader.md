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

# CSV Loader

- Author: [JoonHo Kim](https://github.com/jhboyo)
- Peer Review : [syshin0116](https://github.com/syshin0116), [forwardyoung](https://github.com/forwardyoung)
- Proofread : [Q0211](https://github.com/Q0211)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/04-CSVLoader.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/04-CSVLoader.ipynb)


## Overview

This tutorial provides a comprehensive guide on how to use the ```CSVLoader``` utility in LangChain to seamlessly integrate data from CSV files into your applications. The ```CSVLoader``` is a powerful tool for processing structured data, enabling developers to extract, parse, and utilize information from CSV files within the LangChain framework.

[Comma-Separated Values (CSV)](https://en.wikipedia.org/wiki/Comma-separated_values) is one of the most common formats for storing and exchanging data.

```CSVLoader``` simplifies the process of loading, parsing, and extracting data from CSV files, allowing developers to seamlessly incorporate this information into LangChain workflows.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [How to load CSVs](#how-to-load-csvs)
- [Customizing the CSV parsing and loading](#customizing-the-csv-parsing-and-loading)
- [Specify a column to identify the document source](#specify-a-column-to-identify-the-document-source)
- [Generating XML document format](#generating-xml-document-format)
- [UnstructuredCSVLoader](#unstructuredcsvloader)
- [DataFrameLoader](#dataframeloader)


### References

- [Langchain CSVLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.csv_loader.CSVLoader.html)
- [Langchain How to load CSVs](https://python.langchain.com/docs/how_to/document_loader_csv)
- [Langchain DataFrameLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dataframe.DataFrameLoader.html#dataframeloader)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can check out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.
- ```unstructured``` package is a Python library for extracting text and metadata from various document formats like PDF and CSV


```python
%%capture --no-stderr
%pip install langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain_community",
        "unstructured"
    ],
    verbose=False,
    upgrade=False,
)
```

```python
# Set environment variables
from langchain_opentutorial import set_env
from dotenv import load_dotenv

if not load_dotenv():
    set_env(
        {
            "OPENAI_API_KEY": "",
            "LANGCHAIN_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            "LANGCHAIN_PROJECT": "04-CSV-Loader",
        }
    )
```

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



## How to load CSVs

A comma-separated values (CSV) file is a delimited text file that uses a comma to separate values. LangChain can help you load CSV files easily—just import ```CSVLoader``` to get started. 

Each line of the file is a data record, and each record consists of one or more fields, separated by commas. 

We use a sample CSV file for the example.

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# Create CSVLoader instance
loader = CSVLoader(file_path="./data/titanic.csv")

# Load documents
docs = loader.load()

for record in docs[:2]:
    print(record)
```

<pre class="custom">page_content='PassengerId: 1
    Survived: 0
    Pclass: 3
    Name: Braund, Mr. Owen Harris
    Sex: male
    Age: 22
    SibSp: 1
    Parch: 0
    Ticket: A/5 21171
    Fare: 7.25
    Cabin: 
    Embarked: S' metadata={'source': './data/titanic.csv', 'row': 0}
    page_content='PassengerId: 2
    Survived: 1
    Pclass: 1
    Name: Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    Sex: female
    Age: 38
    SibSp: 1
    Parch: 0
    Ticket: PC 17599
    Fare: 71.2833
    Cabin: C85
    Embarked: C' metadata={'source': './data/titanic.csv', 'row': 1}
</pre>

```python
print(docs[1].page_content)
```

<pre class="custom">PassengerId: 2
    Survived: 1
    Pclass: 1
    Name: Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    Sex: female
    Age: 38
    SibSp: 1
    Parch: 0
    Ticket: PC 17599
    Fare: 71.2833
    Cabin: C85
    Embarked: C
</pre>

## Customizing the CSV parsing and loading

```CSVLoader``` accepts a ```csv_args``` keyword argument that supports customization of the parameters passed to Python's ```csv.DictReader```. This allows you to handle various CSV formats, such as custom delimiters, quote characters, or specific newline handling. 

See Python's [csv module](https://docs.python.org/3/library/csv.html) documentation for more information on supported ```csv_args``` and how to tailor the parsing to your specific needs.

```python
loader = CSVLoader(
    file_path="./data/titanic.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": [
            "Passenger ID",
            "Survival (1: Survived, 0: Died)",
            "Passenger Class",
            "Name",
            "Sex",
            "Age",
            "Number of Siblings/Spouses Aboard",
            "Number of Parents/Children Aboard",
            "Ticket Number",
            "Fare",
            "Cabin",
            "Port of Embarkation",
        ],
    },
)

docs = loader.load()

print(docs[1].page_content)
```

<pre class="custom">Passenger ID: 1
    Survival (1: Survived, 0: Died): 0
    Passenger Class: 3
    Name: Braund, Mr. Owen Harris
    Sex: male
    Age: 22
    Number of Siblings/Spouses Aboard: 1
    Number of Parents/Children Aboard: 0
    Ticket Number: A/5 21171
    Fare: 7.25
    Cabin: 
    Port of Embarkation: S
</pre>

## Specify a column to identify the document source

You should use the ```source_column``` argument to specify the source of the documents generated from each row. Otherwise ```file_path``` will be used as the source for all documents created from the CSV file.

This is particularly useful when using the documents loaded from a CSV file in a chain designed to answer questions based on their source.

```python
loader = CSVLoader(
    file_path="./data/titanic.csv",
    source_column="PassengerId",  # Specify the source column
)

docs = loader.load()  

print(docs[1])
print(docs[1].metadata)
```

<pre class="custom">page_content='PassengerId: 2
    Survived: 1
    Pclass: 1
    Name: Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    Sex: female
    Age: 38
    SibSp: 1
    Parch: 0
    Ticket: PC 17599
    Fare: 71.2833
    Cabin: C85
    Embarked: C' metadata={'source': '2', 'row': 1}
    {'source': '2', 'row': 1}
</pre>

## Generating XML document format

This example shows how to generate XML Document format from ```CSVLoader```. By processing data from a CSV file, you can convert its rows and columns into a structured XML representation.

Convert a row in the document.

```python
row = docs[1].page_content.split("\n")  # split by new line
row_str = "<row>"
for element in row:
    splitted_element = element.split(":")  # split by ":"
    value = splitted_element[-1]  # get value
    col = ":".join(splitted_element[:-1])  # get column name

    row_str += f"<{col}>{value.strip()}</{col}>"
row_str += "</row>"
print(row_str)
```

<pre class="custom"><row><PassengerId>2</PassengerId><Survived>1</Survived><Pclass>1</Pclass><Name>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</Name><Sex>female</Sex><Age>38</Age><SibSp>1</SibSp><Parch>0</Parch><Ticket>PC 17599</Ticket><Fare>71.2833</Fare><Cabin>C85</Cabin><Embarked>C</Embarked></row>
</pre>

Convert entire rows in the document.

```python
for doc in docs[1:6]:  # skip header
    row = doc.page_content.split("\n")
    row_str = "<row>"
    for element in row:
        splitted_element = element.split(":")  # split by ":"
        value = splitted_element[-1]  # get value
        col = ":".join(splitted_element[:-1])  # get column name
        row_str += f"<{col}>{value.strip()}</{col}>"
    row_str += "</row>"
    print(row_str)
```

<pre class="custom"><row><PassengerId>2</PassengerId><Survived>1</Survived><Pclass>1</Pclass><Name>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</Name><Sex>female</Sex><Age>38</Age><SibSp>1</SibSp><Parch>0</Parch><Ticket>PC 17599</Ticket><Fare>71.2833</Fare><Cabin>C85</Cabin><Embarked>C</Embarked></row>
    <row><PassengerId>3</PassengerId><Survived>1</Survived><Pclass>3</Pclass><Name>Heikkinen, Miss. Laina</Name><Sex>female</Sex><Age>26</Age><SibSp>0</SibSp><Parch>0</Parch><Ticket>STON/O2. 3101282</Ticket><Fare>7.925</Fare><Cabin></Cabin><Embarked>S</Embarked></row>
    <row><PassengerId>4</PassengerId><Survived>1</Survived><Pclass>1</Pclass><Name>Futrelle, Mrs. Jacques Heath (Lily May Peel)</Name><Sex>female</Sex><Age>35</Age><SibSp>1</SibSp><Parch>0</Parch><Ticket>113803</Ticket><Fare>53.1</Fare><Cabin>C123</Cabin><Embarked>S</Embarked></row>
    <row><PassengerId>5</PassengerId><Survived>0</Survived><Pclass>3</Pclass><Name>Allen, Mr. William Henry</Name><Sex>male</Sex><Age>35</Age><SibSp>0</SibSp><Parch>0</Parch><Ticket>373450</Ticket><Fare>8.05</Fare><Cabin></Cabin><Embarked>S</Embarked></row>
    <row><PassengerId>6</PassengerId><Survived>0</Survived><Pclass>3</Pclass><Name>Moran, Mr. James</Name><Sex>male</Sex><Age></Age><SibSp>0</SibSp><Parch>0</Parch><Ticket>330877</Ticket><Fare>8.4583</Fare><Cabin></Cabin><Embarked>Q</Embarked></row>
</pre>

## UnstructuredCSVLoader 

```UnstructuredCSVLoader``` can be used in both ```single``` and ```elements``` mode. If you use the loader in “elements” mode, the CSV file will be a single Unstructured Table element. If you use the loader in elements” mode, an HTML representation of the table will be available in the ```text_as_html``` key in the document metadata.

```python
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader

# Generate UnstructuredCSVLoader instance with elements mode
loader = UnstructuredCSVLoader(file_path="./data/titanic.csv", mode="elements")

docs = loader.load()

html_content = docs[0].metadata["text_as_html"]

# Partial output due to space constraints
print(html_content[:810]) 
```

<pre class="custom"><table><tr><td>PassengerId</td><td>Survived</td><td>Pclass</td><td>Name</td><td>Sex</td><td>Age</td><td>SibSp</td><td>Parch</td><td>Ticket</td><td>Fare</td><td>Cabin</td><td>Embarked</td></tr><tr><td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris</td><td>male</td><td>22</td><td>1</td><td>0</td><td>A/5 21171</td><td>7.25</td><td/><td>S</td></tr><tr><td>2</td><td>1</td><td>1</td><td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td>female</td><td>38</td><td>1</td><td>0</td><td>PC 17599</td><td>71.2833</td><td>C85</td><td>C</td></tr><tr><td>3</td><td>1</td><td>3</td><td>Heikkinen, Miss. Laina</td><td>female</td><td>26</td><td>0</td><td>0</td><td>STON/O2. 3101282</td><td>7.925</td><td/><td>S</td></tr><tr><td>4</td><td>1</td><td>1</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
</pre>

## DataFrameLoader

```Pandas``` is an open-source data analysis and manipulation tool for the Python programming language. This library is widely used in data science, machine learning, and various fields for working with data.

LangChain's ```DataFrameLoader``` is a powerful utility designed to seamlessly integrate ```Pandas```  ```DataFrames``` into LangChain workflows.

```python
import pandas as pd

df = pd.read_csv("./data/titanic.csv")
```

Search the first 5 rows.

```python
df.head(n=5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Parameters ```page_content_column``` (str) – Name of the column containing the page content. Defaults to “text”.



```python
from langchain_community.document_loaders import DataFrameLoader

# The Name column of the DataFrame is specified to be used as the content of each document.
loader = DataFrameLoader(df, page_content_column="Name")

docs = loader.load()

print(docs[0].page_content)

```

<pre class="custom">Braund, Mr. Owen Harris
</pre>

```Lazy Loading``` for large tables. Avoid loading the entire table into memory

```python
# Lazy load records from dataframe.
for row in loader.lazy_load():
    print(row)
    break  # print only the first row

```

<pre class="custom">page_content='Braund, Mr. Owen Harris' metadata={'PassengerId': 1, 'Survived': 0, 'Pclass': 3, 'Sex': 'male', 'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': nan, 'Embarked': 'S'}
</pre>
