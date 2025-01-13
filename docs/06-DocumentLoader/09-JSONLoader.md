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

Let's look at how to load files with the `.json` extension using a loader.

- Author: [leebeanbin](https://github.com/leebeanbin)
- Design:
- Peer Review : [syshin0116](https://github.com/syshin0116), [Teddy Lee](https://github.com/teddylee777)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/tree/main/06-DocumentLoader)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/10-JSON-Loader.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/10-JSON-Loader.ipynb)

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The `langchain-opentutorial` is a bundle of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

## Overview
This tutorial demonstrates how to use LangChain's JSONLoader to load and process JSON files. We'll explore how to extract specific data from structured JSON files using jq-style queries.

### Table of Contents
- [Environment Set up](#environment-setup)
- [JSON](#json)
- [Overview](#overview)
- [Generate JSON Data](#generate-json-data)
- [JSONLoader](#jsonloader)
  
When you want to extract values under the content field within the message key of JSON data, you can easily do this using JSONLoader as shown below.


### reference
- https://python.langchain.com/docs/how_to/document_loader_json/

## Environment Setup

You can set and load `OPENAI_API_KEY` from a `.env` file when you'd like to make new json file.


```python
%pip install langchain langchain_openai langchain_community rq
```

<pre class="custom">Requirement already satisfied: langchain in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (0.3.13)
    Requirement already satisfied: langchain_openai in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (0.2.14)
    Requirement already satisfied: langchain_community in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (0.3.13)
    Collecting rq
      Downloading rq-2.1.0-py3-none-any.whl.metadata (5.8 kB)
    Requirement already satisfied: PyYAML>=5.3 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (6.0.2)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (2.0.36)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (3.11.11)
    Requirement already satisfied: langchain-core<0.4.0,>=0.3.26 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (0.3.28)
    Requirement already satisfied: langchain-text-splitters<0.4.0,>=0.3.3 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (0.3.4)
    Requirement already satisfied: langsmith<0.3,>=0.1.17 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (0.2.7)
    Requirement already satisfied: numpy<2,>=1.22.4 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (1.26.4)
    Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (2.10.4)
    Requirement already satisfied: requests<3,>=2 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (2.32.3)
    Requirement already satisfied: tenacity!=8.4.0,<10,>=8.1.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain) (9.0.0)
    Requirement already satisfied: openai<2.0.0,>=1.58.1 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain_openai) (1.58.1)
    Requirement already satisfied: tiktoken<1,>=0.7 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain_openai) (0.8.0)
    Requirement already satisfied: dataclasses-json<0.7,>=0.5.7 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain_community) (0.6.7)
    Requirement already satisfied: httpx-sse<0.5.0,>=0.4.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain_community) (0.4.0)
    Requirement already satisfied: pydantic-settings<3.0.0,>=2.4.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain_community) (2.7.1)
    Collecting click>=5 (from rq)
      Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
    Collecting redis>=3.5 (from rq)
      Downloading redis-5.2.1-py3-none-any.whl.metadata (9.1 kB)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.4.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (24.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.1.0)
    Requirement already satisfied: propcache>=0.2.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (0.2.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.18.3)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain_community) (3.23.2)
    Requirement already satisfied: typing-inspect<1,>=0.4.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain_community) (0.9.0)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain-core<0.4.0,>=0.3.26->langchain) (1.33)
    Requirement already satisfied: packaging<25,>=23.2 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain-core<0.4.0,>=0.3.26->langchain) (24.2)
    Requirement already satisfied: typing-extensions>=4.7 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langchain-core<0.4.0,>=0.3.26->langchain) (4.12.2)
    Requirement already satisfied: httpx<1,>=0.23.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langsmith<0.3,>=0.1.17->langchain) (0.27.2)
    Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langsmith<0.3,>=0.1.17->langchain) (3.10.13)
    Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from langsmith<0.3,>=0.1.17->langchain) (1.0.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain_openai) (4.7.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain_openai) (1.9.0)
    Requirement already satisfied: jiter<1,>=0.4.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain_openai) (0.8.2)
    Requirement already satisfied: sniffio in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain_openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from openai<2.0.0,>=1.58.1->langchain_openai) (4.67.1)
    Requirement already satisfied: annotated-types>=0.6.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.7.0)
    Requirement already satisfied: pydantic-core==2.27.2 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.27.2)
    Requirement already satisfied: python-dotenv>=0.21.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from pydantic-settings<3.0.0,>=2.4.0->langchain_community) (1.0.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from requests<3,>=2->langchain) (2024.12.14)
    Requirement already satisfied: regex>=2022.1.18 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from tiktoken<1,>=0.7->langchain_openai) (2024.11.6)
    Requirement already satisfied: httpcore==1.* in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from httpx<1,>=0.23.0->langsmith<0.3,>=0.1.17->langchain) (1.0.7)
    Requirement already satisfied: h11<0.15,>=0.13 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith<0.3,>=0.1.17->langchain) (0.14.0)
    Requirement already satisfied: jsonpointer>=1.9 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.4.0,>=0.3.26->langchain) (3.0.0)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/leejungbin/Library/Caches/pypoetry/virtualenvs/langchain-opentutorial-LGorndcz-py3.11/lib/python3.11/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain_community) (1.0.0)
    Downloading rq-2.1.0-py3-none-any.whl (96 kB)
    Downloading click-8.1.8-py3-none-any.whl (98 kB)
    Downloading redis-5.2.1-py3-none-any.whl (261 kB)
    Installing collected packages: redis, click, rq
    Successfully installed click-8.1.8 redis-5.2.1 rq-2.1.0
    Note: you may need to restart the kernel to use updated packages.
</pre>

## Generate JSON Data

---

if you want to generate JSON data, you can use the following code.


```python
from langchain.prompts import PromptTemplate
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

# JSONLoader

---

When you want to extract values under the content field within the message key of JSON data, you can easily do this using JSONLoader as shown below.

```python
from langchain_community.document_loaders import JSONLoader

# Create JSONLoader
loader = JSONLoader(
    file_path="data/people.json",
    jq_schema=".people[]",  # Access each item in the people array
    text_content=False,
)

# Example: extract only contact_details
# loader = JSONLoader(
#     file_path="data/people.json",
#     jq_schema=".people[].contact_details",
#     text_content=False,
# )

# Or extract only hobbies from personal_preferences
# loader = JSONLoader(
#     file_path="data/people.json",
#     jq_schema=".people[].personal_preferences.hobbies",
#     text_content=False,
# )

# Load documents
docs = loader.load()
pprint(docs)
```

<pre class="custom">[Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 1}, page_content='{"name": "Alice Smith", "age": 32, "contact": {"email": "alice.smith@example.com", "phone": "555-123-4567"}, "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}, "personal_preferences": {"favorite_color": "blue", "hobbies": ["reading", "yoga"], "favorite_food": "sushi"}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 2}, page_content='{"name": "John Doe", "age": 45, "contact": {"email": "john.doe@example.com", "phone": "555-987-6543"}, "address": {"street": "456 Elm St", "city": "Los Angeles", "state": "CA", "zip": "90001"}, "personal_preferences": {"favorite_color": "green", "hobbies": ["hiking", "gardening"], "favorite_food": "pizza"}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 3}, page_content='{"name": "Emily Johnson", "age": 28, "contact": {"email": "emily.johnson@example.com", "phone": "555-456-7890"}, "address": {"street": "789 Oak St", "city": "Chicago", "state": "IL", "zip": "60601"}, "personal_preferences": {"favorite_color": "pink", "hobbies": ["painting", "traveling"], "favorite_food": "tacos"}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 4}, page_content='{"name": "Michael Brown", "age": 38, "contact": {"email": "michael.brown@example.com", "phone": "555-234-5678"}, "address": {"street": "321 Maple St", "city": "Houston", "state": "TX", "zip": "77001"}, "personal_preferences": {"favorite_color": "red", "hobbies": ["playing guitar", "cooking"], "favorite_food": "barbecue"}}'),
     Document(metadata={'source': '/Users/leejungbin/Downloads/LangChain-OpenTutorial/06-DocumentLoader/data/people.json', 'seq_num': 5}, page_content='{"name": "Sarah Wilson", "age": 35, "contact": {"email": "sarah.wilson@example.com", "phone": "555-345-6789"}, "address": {"street": "654 Pine St", "city": "Miami", "state": "FL", "zip": "33101"}, "personal_preferences": {"favorite_color": "purple", "hobbies": ["photography", "dancing"], "favorite_food": "sushi"}}')]
</pre>
