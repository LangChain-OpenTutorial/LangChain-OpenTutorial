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

# JSON

Let's look at how to load files with the ```.json``` extension using a loader.

- Author: [leebeanbin](https://github.com/leebeanbin)
- Peer Review : [syshin0116](https://github.com/syshin0116), [Teddy Lee](https://github.com/teddylee777)
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/tree/main/06-DocumentLoader)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/09-JSONLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/09-JSONLoader.ipynb)

## Overview
This tutorial demonstrates how to use LangChain's ```JSONLoader``` to load and process JSON files. We'll explore how to extract specific data from structured JSON files using jq-style queries.

### Table of Contents
- [Environment Set up](#environment-setup)
- [JSON](#json)
- [Overview](#overview)
- [Generate JSON Data](#generate-json-data)
- [JSONLoader](#jsonloader)
  
When you want to extract values under the content field within the message key of JSON data, you can easily do this using ```JSONLoader``` as shown below.


### References
- https://python.langchain.com/docs/how_to/document_loader_json/

---


## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

[Note]
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can check out the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "langchain_community",
        "langchain_openai"
    ],
    verbose=False,
    upgrade=False,
)
```

```python
%pip install rq
```

<pre class="custom">Requirement already satisfied: rq in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (2.1.0)
    Requirement already satisfied: click>=5 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from rq) (8.1.8)
    Requirement already satisfied: redis>=3.5 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from rq) (5.2.1)
    Note: you may need to restart the kernel to use updated packages.
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
        "LANGCHAIN_PROJECT": "09-JSONLoader",
    }
)
```

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
# Load environment variables
# Reload any variables that need to be overwritten from the previous cell

from dotenv import load_dotenv

load_dotenv(override=True)
```

## Generate JSON Data

---

If you want to generate JSON data, you can use the following code.


```python
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import json
import os

# Load .env file
load_dotenv()

# Initialize ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Create prompt template
prompt = PromptTemplate(
    input_variables=[],
    template="""Generate a JSON array containing detailed personal information for 5 people. 
        Include various fields like name, age, contact details, address, personal preferences, and any other interesting information you think would be relevant."""
)

# Create and invoke runnable sequence using the new pipe syntax
response = (prompt | llm).invoke({})
generated_data = json.loads(response.content)

# Save to JSON file
current_dir = Path().absolute()
data_dir = current_dir / "data"
data_dir.mkdir(exist_ok=True)

file_path = data_dir / "people.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(generated_data, f, ensure_ascii=False, indent=2)

print("Generated and saved JSON data:")
pprint(generated_data)
```

<pre class="custom">Generated and saved JSON data:
    {'people': [{'address': {'city': 'Springfield',
                             'country': 'USA',
                             'state': 'IL',
                             'street': '123 Maple St',
                             'zip': '62701'},
                 'age': 28,
                 'contact_details': {'email': 'alice.johnson@example.com',
                                     'phone': '+1-555-123-4567'},
                 'interesting_information': {'pet': {'breed': 'Golden Retriever',
                                                     'name': 'Buddy',
                                                     'type': 'dog'},
                                             'travel_history': [{'country': 'Japan',
                                                                 'year': 2019},
                                                                {'country': 'Italy',
                                                                 'year': 2021}]},
                 'name': 'Alice Johnson',
                 'personal_preferences': {'favorite_food': 'sushi',
                                          'hobbies': ['reading',
                                                      'hiking',
                                                      'photography'],
                                          'music_genres': ['jazz',
                                                           'classical',
                                                           'indie']}},
                {'address': {'city': 'Denver',
                             'country': 'USA',
                             'state': 'CO',
                             'street': '456 Oak Ave',
                             'zip': '80202'},
                 'age': 34,
                 'contact_details': {'email': 'michael.smith@example.com',
                                     'phone': '+1-555-234-5678'},
                 'interesting_information': {'pet': {'breed': 'Siamese',
                                                     'name': 'Whiskers',
                                                     'type': 'cat'},
                                             'volunteering': {'organization': 'Local '
                                                                              'Food '
                                                                              'Bank',
                                                              'years_active': 5}},
                 'name': 'Michael Smith',
                 'personal_preferences': {'favorite_food': 'pizza',
                                          'hobbies': ['cycling',
                                                      'cooking',
                                                      'gaming'],
                                          'music_genres': ['rock',
                                                           'pop',
                                                           'hip-hop']}},
                {'address': {'city': 'Austin',
                             'country': 'USA',
                             'state': 'TX',
                             'street': '789 Pine Rd',
                             'zip': '73301'},
                 'age': 22,
                 'contact_details': {'email': 'emily.davis@example.com',
                                     'phone': '+1-555-345-6789'},
                 'interesting_information': {'pet': None,
                                             'study': {'graduation_year': 2024,
                                                       'major': 'Fine Arts',
                                                       'university': 'University '
                                                                     'of Texas'}},
                 'name': 'Emily Davis',
                 'personal_preferences': {'favorite_food': 'tacos',
                                          'hobbies': ['painting',
                                                      'traveling',
                                                      'yoga'],
                                          'music_genres': ['country',
                                                           'folk',
                                                           'dance']}},
                {'address': {'city': 'Seattle',
                             'country': 'USA',
                             'state': 'WA',
                             'street': '101 Birch Blvd',
                             'zip': '98101'},
                 'age': 45,
                 'contact_details': {'email': 'david.brown@example.com',
                                     'phone': '+1-555-456-7890'},
                 'interesting_information': {'career': {'job_title': 'Software '
                                                                     'Engineer',
                                                        'years_experience': 20},
                                             'pet': {'breed': 'Canary',
                                                     'name': 'Tweety',
                                                     'type': 'bird'}},
                 'name': 'David Brown',
                 'personal_preferences': {'favorite_food': 'steak',
                                          'hobbies': ['golf', 'reading', 'fishing'],
                                          'music_genres': ['blues',
                                                           'classic rock',
                                                           'jazz']}},
                {'address': {'city': 'Miami',
                             'country': 'USA',
                             'state': 'FL',
                             'street': '202 Cedar Ct',
                             'zip': '33101'},
                 'age': 39,
                 'contact_details': {'email': 'sophia.wilson@example.com',
                                     'phone': '+1-555-567-8901'},
                 'interesting_information': {'pet': {'breed': 'Bulldog',
                                                     'name': 'Max',
                                                     'type': 'dog'},
                                             'travel_history': [{'country': 'Spain',
                                                                 'year': 2018},
                                                                {'country': 'Brazil',
                                                                 'year': 2020}]},
                 'name': 'Sophia Wilson',
                 'personal_preferences': {'favorite_food': 'paella',
                                          'hobbies': ['dancing',
                                                      'gardening',
                                                      'cooking'],
                                          'music_genres': ['latin',
                                                           'pop',
                                                           'salsa']}}]}
</pre>

The case of loading JSON data is as follows when you want to load your own JSON data.

```python
import json
from pathlib import Path
from pprint import pprint


file_path = "data/people.json"
data = json.loads(Path(file_path).read_text())

pprint(data)
```

<pre class="custom">{'people': [{'address': {'city': 'Springfield',
                             'country': 'USA',
                             'state': 'IL',
                             'street': '123 Maple St',
                             'zip': '62704'},
                 'age': 28,
                 'contact': {'email': 'alice.johnson@example.com',
                             'phone': '+1-555-0123',
                             'social_media': {'linkedin': 'linkedin.com/in/alicejohnson',
                                              'twitter': '@alice_j'}},
                 'interesting_fact': 'Alice has traveled to over 15 countries and '
                                     'speaks 3 languages.',
                 'name': {'first': 'Alice', 'last': 'Johnson'},
                 'personal_preferences': {'favorite_food': 'Italian',
                                          'hobbies': ['Reading',
                                                      'Hiking',
                                                      'Cooking'],
                                          'music_genre': 'Jazz',
                                          'travel_destinations': ['Japan',
                                                                  'Italy',
                                                                  'Canada']}},
                {'address': {'city': 'Metropolis',
                             'country': 'USA',
                             'state': 'NY',
                             'street': '456 Oak Ave',
                             'zip': '10001'},
                 'age': 34,
                 'contact': {'email': 'bob.smith@example.com',
                             'phone': '+1-555-0456',
                             'social_media': {'linkedin': 'linkedin.com/in/bobsmith',
                                              'twitter': '@bobsmith34'}},
                 'interesting_fact': 'Bob is an avid gamer and has competed in '
                                     'several national tournaments.',
                 'name': {'first': 'Bob', 'last': 'Smith'},
                 'personal_preferences': {'favorite_food': 'Mexican',
                                          'hobbies': ['Photography',
                                                      'Cycling',
                                                      'Video Games'],
                                          'music_genre': 'Rock',
                                          'travel_destinations': ['Brazil',
                                                                  'Australia',
                                                                  'Germany']}},
                {'address': {'city': 'Gotham',
                             'country': 'USA',
                             'state': 'NJ',
                             'street': '789 Pine Rd',
                             'zip': '07001'},
                 'age': 45,
                 'contact': {'email': 'charlie.davis@example.com',
                             'phone': '+1-555-0789',
                             'social_media': {'linkedin': 'linkedin.com/in/charliedavis',
                                              'twitter': '@charliedavis45'}},
                 'interesting_fact': 'Charlie has a small farm where he raises '
                                     'chickens and grows organic vegetables.',
                 'name': {'first': 'Charlie', 'last': 'Davis'},
                 'personal_preferences': {'favorite_food': 'Barbecue',
                                          'hobbies': ['Gardening',
                                                      'Fishing',
                                                      'Woodworking'],
                                          'music_genre': 'Country',
                                          'travel_destinations': ['Canada',
                                                                  'New Zealand',
                                                                  'Norway']}},
                {'address': {'city': 'Star City',
                             'country': 'USA',
                             'state': 'CA',
                             'street': '234 Birch Blvd',
                             'zip': '90001'},
                 'age': 22,
                 'contact': {'email': 'dana.lee@example.com',
                             'phone': '+1-555-0111',
                             'social_media': {'linkedin': 'linkedin.com/in/danalee',
                                              'twitter': '@danalee22'}},
                 'interesting_fact': 'Dana is a dance instructor and has won '
                                     'several local competitions.',
                 'name': {'first': 'Dana', 'last': 'Lee'},
                 'personal_preferences': {'favorite_food': 'Thai',
                                          'hobbies': ['Dancing',
                                                      'Sketching',
                                                      'Traveling'],
                                          'music_genre': 'Pop',
                                          'travel_destinations': ['Thailand',
                                                                  'France',
                                                                  'Spain']}},
                {'address': {'city': 'Central City',
                             'country': 'USA',
                             'state': 'TX',
                             'street': '345 Cedar St',
                             'zip': '75001'},
                 'age': 31,
                 'contact': {'email': 'ethan.garcia@example.com',
                             'phone': '+1-555-0999',
                             'social_media': {'linkedin': 'linkedin.com/in/ethangarcia',
                                              'twitter': '@ethangarcia31'}},
                 'interesting_fact': 'Ethan runs a popular travel blog where he '
                                     'shares his adventures and culinary '
                                     'experiences.',
                 'name': {'first': 'Ethan', 'last': 'Garcia'},
                 'personal_preferences': {'favorite_food': 'Indian',
                                          'hobbies': ['Running',
                                                      'Travel Blogging',
                                                      'Cooking'],
                                          'music_genre': 'Hip-Hop',
                                          'travel_destinations': ['India',
                                                                  'Italy',
                                                                  'Mexico']}}]}
</pre>

```python
print(type(data))
```

<pre class="custom"><class 'dict'>
</pre>

## ```JSONLoader```

---

When you want to extract values under the content field within the message key of JSON data, you can easily do this using ```JSONLoader``` as shown below.

### Basic Usage

This usage shows off how to execute load JSON and print what I get from

```python
from langchain_community.document_loaders import JSONLoader

# Create JSONLoader
loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[]",  # Access each item in the people array
    text_content=False,
)

# Load documents
docs = loader.load()
pprint(docs)
```

<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='{"name": "Alice Johnson", "age": 28, "contact_details": {"email": "alice.johnson@example.com", "phone": "+1-555-123-4567"}, "address": {"street": "123 Maple St", "city": "Springfield", "state": "IL", "zip": "62701", "country": "USA"}, "personal_preferences": {"hobbies": ["reading", "hiking", "photography"], "favorite_food": "sushi", "music_genres": ["jazz", "classical", "indie"]}, "interesting_information": {"pet": {"type": "dog", "name": "Buddy", "breed": "Golden Retriever"}, "travel_history": [{"country": "Japan", "year": 2019}, {"country": "Italy", "year": 2021}]}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='{"name": "Michael Smith", "age": 34, "contact_details": {"email": "michael.smith@example.com", "phone": "+1-555-234-5678"}, "address": {"street": "456 Oak Ave", "city": "Denver", "state": "CO", "zip": "80202", "country": "USA"}, "personal_preferences": {"hobbies": ["cycling", "cooking", "gaming"], "favorite_food": "pizza", "music_genres": ["rock", "pop", "hip-hop"]}, "interesting_information": {"pet": {"type": "cat", "name": "Whiskers", "breed": "Siamese"}, "volunteering": {"organization": "Local Food Bank", "years_active": 5}}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='{"name": "Emily Davis", "age": 22, "contact_details": {"email": "emily.davis@example.com", "phone": "+1-555-345-6789"}, "address": {"street": "789 Pine Rd", "city": "Austin", "state": "TX", "zip": "73301", "country": "USA"}, "personal_preferences": {"hobbies": ["painting", "traveling", "yoga"], "favorite_food": "tacos", "music_genres": ["country", "folk", "dance"]}, "interesting_information": {"pet": null, "study": {"major": "Fine Arts", "university": "University of Texas", "graduation_year": 2024}}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='{"name": "David Brown", "age": 45, "contact_details": {"email": "david.brown@example.com", "phone": "+1-555-456-7890"}, "address": {"street": "101 Birch Blvd", "city": "Seattle", "state": "WA", "zip": "98101", "country": "USA"}, "personal_preferences": {"hobbies": ["golf", "reading", "fishing"], "favorite_food": "steak", "music_genres": ["blues", "classic rock", "jazz"]}, "interesting_information": {"pet": {"type": "bird", "name": "Tweety", "breed": "Canary"}, "career": {"job_title": "Software Engineer", "years_experience": 20}}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='{"name": "Sophia Wilson", "age": 39, "contact_details": {"email": "sophia.wilson@example.com", "phone": "+1-555-567-8901"}, "address": {"street": "202 Cedar Ct", "city": "Miami", "state": "FL", "zip": "33101", "country": "USA"}, "personal_preferences": {"hobbies": ["dancing", "gardening", "cooking"], "favorite_food": "paella", "music_genres": ["latin", "pop", "salsa"]}, "interesting_information": {"pet": {"type": "dog", "name": "Max", "breed": "Bulldog"}, "travel_history": [{"country": "Spain", "year": 2018}, {"country": "Brazil", "year": 2020}]}}')]
</pre>

### Loading Each Person as a Separate Document

We can load each person object from ```people.json``` as an individual document using the ```jq_schema=".people[]"```

```python
loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[]",
    text_content=False,
)

data = loader.load()
data
```




<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='{"name": "Alice Johnson", "age": 28, "contact_details": {"email": "alice.johnson@example.com", "phone": "+1-555-123-4567"}, "address": {"street": "123 Maple St", "city": "Springfield", "state": "IL", "zip": "62701", "country": "USA"}, "personal_preferences": {"hobbies": ["reading", "hiking", "photography"], "favorite_food": "sushi", "music_genres": ["jazz", "classical", "indie"]}, "interesting_information": {"pet": {"type": "dog", "name": "Buddy", "breed": "Golden Retriever"}, "travel_history": [{"country": "Japan", "year": 2019}, {"country": "Italy", "year": 2021}]}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='{"name": "Michael Smith", "age": 34, "contact_details": {"email": "michael.smith@example.com", "phone": "+1-555-234-5678"}, "address": {"street": "456 Oak Ave", "city": "Denver", "state": "CO", "zip": "80202", "country": "USA"}, "personal_preferences": {"hobbies": ["cycling", "cooking", "gaming"], "favorite_food": "pizza", "music_genres": ["rock", "pop", "hip-hop"]}, "interesting_information": {"pet": {"type": "cat", "name": "Whiskers", "breed": "Siamese"}, "volunteering": {"organization": "Local Food Bank", "years_active": 5}}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='{"name": "Emily Davis", "age": 22, "contact_details": {"email": "emily.davis@example.com", "phone": "+1-555-345-6789"}, "address": {"street": "789 Pine Rd", "city": "Austin", "state": "TX", "zip": "73301", "country": "USA"}, "personal_preferences": {"hobbies": ["painting", "traveling", "yoga"], "favorite_food": "tacos", "music_genres": ["country", "folk", "dance"]}, "interesting_information": {"pet": null, "study": {"major": "Fine Arts", "university": "University of Texas", "graduation_year": 2024}}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='{"name": "David Brown", "age": 45, "contact_details": {"email": "david.brown@example.com", "phone": "+1-555-456-7890"}, "address": {"street": "101 Birch Blvd", "city": "Seattle", "state": "WA", "zip": "98101", "country": "USA"}, "personal_preferences": {"hobbies": ["golf", "reading", "fishing"], "favorite_food": "steak", "music_genres": ["blues", "classic rock", "jazz"]}, "interesting_information": {"pet": {"type": "bird", "name": "Tweety", "breed": "Canary"}, "career": {"job_title": "Software Engineer", "years_experience": 20}}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='{"name": "Sophia Wilson", "age": 39, "contact_details": {"email": "sophia.wilson@example.com", "phone": "+1-555-567-8901"}, "address": {"street": "202 Cedar Ct", "city": "Miami", "state": "FL", "zip": "33101", "country": "USA"}, "personal_preferences": {"hobbies": ["dancing", "gardening", "cooking"], "favorite_food": "paella", "music_genres": ["latin", "pop", "salsa"]}, "interesting_information": {"pet": {"type": "dog", "name": "Max", "breed": "Bulldog"}, "travel_history": [{"country": "Spain", "year": 2018}, {"country": "Brazil", "year": 2020}]}}')]</pre>



### Using ```content_key``` within ```jq_schema```

To load documents from a JSON file using ```content_key``` within the ```jq_schema```, set ```is_content_key_jq_parsable=True```. Ensure that ```content_key``` is compatible and can be parsed using the ```jq_schema```.

```python
loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[]",
    content_key="name",
    text_content=False
)

data = loader.load()
data
```




<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='Alice Johnson'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='Michael Smith'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='Emily Davis'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='David Brown'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='Sophia Wilson')]</pre>



### Extracting Metadata from ```people.json```

Let's define a ```metadata_func``` to extract relevant information like name, age, and city from each person object.


```python
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["name"] = record.get("name")
    metadata["age"] = record.get("age")
    metadata["city"] = record.get("address", {}).get("city")
    return metadata

loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[]",
    content_key="name",
    metadata_func=metadata_func,
    text_content=False
)

data = loader.load()
data
```




<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1, 'name': 'Alice Johnson', 'age': 28, 'city': 'Springfield'}, page_content='Alice Johnson'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2, 'name': 'Michael Smith', 'age': 34, 'city': 'Denver'}, page_content='Michael Smith'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3, 'name': 'Emily Davis', 'age': 22, 'city': 'Austin'}, page_content='Emily Davis'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4, 'name': 'David Brown', 'age': 45, 'city': 'Seattle'}, page_content='David Brown'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5, 'name': 'Sophia Wilson', 'age': 39, 'city': 'Miami'}, page_content='Sophia Wilson')]</pre>



### Understanding JSON Query Syntax

Let's explore the basic syntax of jq-style queries used in ```JSONLoader```:

Basic Selectors
   - **```.```** : Current object
   - **```.key```** : Access specific key in object
   - **```.[]```** : Iterate over array elements

Pipe Operator
   - **```|```** : Pass result of left expression as input to right expression
   
Object Construction
   - **```{key: value}```** : Create new object

Example JSON:
```json
{
  "people": [
    {"name": "Alice", "age": 30, "contactDetails": {"email": "alice@example.com", "phone": "123-456-7890"}},
    {"name": "Bob", "age": 25, "contactDetails": {"email": "bob@example.com", "phone": "098-765-4321"}}
  ]
}
```

**Common Query Patterns**:
- ```.people[]``` : Access each array element
- ```.people[].name``` : Get all names
- ```.people[] | {name: .name}``` : Create new object with name
- ```.people[] | {name, email: .contact.email}``` : Extract nested data

[Note] 
- Always use ```text_content=False``` when working with complex JSON data
- This ensures proper handling of non-string values (objects, arrays, numbers)

### Advanced Queries

Here are examples of extracting specific information using different jq schemas:

```python
# Extract only contact details
contact_loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[] | {name: .name, contact: .contactDetails}",
    text_content=False
)

docs = contact_loader.load()
docs
```




<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='{"name": "Alice Johnson", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='{"name": "Michael Smith", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='{"name": "Emily Davis", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='{"name": "David Brown", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='{"name": "Sophia Wilson", "contact": null}')]</pre>



```python
# Extract nested data
hobbies_loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[] | {name: .name, hobbies: .personalPreferences.hobbies}",
    text_content=False
)

docs = hobbies_loader.load()
docs
```




<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='{"name": "Alice Johnson", "hobbies": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='{"name": "Michael Smith", "hobbies": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='{"name": "Emily Davis", "hobbies": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='{"name": "David Brown", "hobbies": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='{"name": "Sophia Wilson", "hobbies": null}')]</pre>



```python
# Get all interesting facts
facts_loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[] | {name: .name, facts: .interestingFacts}",
    text_content=False
)

docs = facts_loader.load()
docs
```




<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='{"name": "Alice Johnson", "facts": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='{"name": "Michael Smith", "facts": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='{"name": "Emily Davis", "facts": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='{"name": "David Brown", "facts": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='{"name": "Sophia Wilson", "facts": null}')]</pre>



```python
# Extract email and phone together
contact_info = JSONLoader(
    file_path="data/people.json",
    jq_schema='.people[] | {name: .name, email: .contactDetails.email, phone: .contactDetails.phone}',
    text_content=False
)

docs = contact_loader.load()
docs
```




<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='{"name": "Alice Johnson", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='{"name": "Michael Smith", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='{"name": "Emily Davis", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='{"name": "David Brown", "contact": null}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='{"name": "Sophia Wilson", "contact": null}')]</pre>



These examples demonstrate the flexibility of jq queries in fetching data in various ways.
