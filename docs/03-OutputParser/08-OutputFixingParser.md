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

# Output Fixing Parser

- Author: [Jeongeun Lim](https://www.linkedin.com/in/jeongeun-lim-808978188/)
- Peer Review : [Junseong Kim](https://www.linkedin.com/in/%EC%A4%80%EC%84%B1-%EA%B9%80-591b351b2/)
- Proofread : [Two-Jay](https://github.com/Two-Jay)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/08-OutputFixingParser.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/03-OutputParser/08-OutputFixingParser.ipynb)
## Overview

The ```OutputFixingParser``` in LangChain provides an automated mechanism for correcting errors that may occur during the output parsing process. This parser is designed to wrap around another parser, such as the ```PydanticOutputParser```, and intervenes when the underlying parser encounters outputs that are malformed or do not conform to the expected format. It achieves this by leveraging additional LLM calls to fix the errors and ensure proper formatting.

At its core, the ```OutputFixingParser``` addresses situations where the initial output does not comply with a predefined schema. If such an issue arises, the parser automatically detects the formatting errors and submits a new request to the model, including specific instructions for correcting the issue. These instructions highlight the problem areas and provide clear guidelines for restructuring the data in the correct format.

This functionality is particularly useful in scenarios where strict adherence to a schema is critical. For example, when using the ```PydanticOutputParser``` to generate outputs conforming to a particular data schema, issues such as missing fields or incorrect data types might occur. 

- The ```OutputFixingParser``` steps in as follows:

1. **Error Detection** : It recognizes that the output does not meet the schema requirements.
2. **Error Correction** : It generates a follow-up request to the LLM with explicit instructions to address the issues.
3. **Reformatted Output with Specific Instructions** : The ```OutputFixingParser``` ensures that the correction instructions precisely identify the errors, such as missing fields or incorrect data types. The instructions guide the LLM to reformat the output to meet the schema requirements accurately.


---- 
Practical Example:

Suppose you are using the ```PydanticOutputParser``` to enforce a schema requiring specific fields like ```name``` (string), ```age``` (integer), and ```email``` (string). If the LLM produces an output where the ```age``` field is missing or the ```email``` field is not a valid string, the ```OutputFixingParser``` automatically intervenes. It would issue a new request to the LLM with detailed instructions such as:

- "The output is missing the ```age``` field. Add an appropriate integer value for ```age```."
- "The ```email``` field contains an invalid format. Correct it to match a valid email string."

This iterative process ensures the final output conforms to the specified schema without requiring manual intervention.


---- 
Key Benefits: 

- **Error Recovery**: Automatically handles malformed outputs without requiring user input.
- **Enhanced Accuracy**: Ensures outputs conform to predefined schemas, reducing the risk of inconsistencies.
- **Streamlined Workflow**: Minimizes the need for manual corrections, saving time and improving efficiency.


---- 
Implementation Steps: 

To use the ```OutputFixingParser``` effectively, follow these steps:

1. **Wrap a Parser**: Instantiate the ```OutputFixingParser``` with another parser, such as the ```PydanticOutputParser```, as its base.
2. **Define the Schema**: Specify the schema or format that the output must adhere to.
3. **Enable Error Correction**: Allow the ```OutputFixingParser``` to detect and correct errors automatically through additional LLM calls, ensuring that correction instructions precisely identify and address issues for accurate reformatting.

By integrating the ```OutputFixingParser``` into your workflow, you can ensure robust error handling and maintain consistent output quality in your LangChain applications.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Define Data Model and Set Up PydanticOutputParser](#define-data-model-and-set-up-pydanticoutputparser)
- [Using OutputFixingParser to Correct Incorrect Formatting](#using-outputfixingparser-to-correct-incorrect-formatting)

### References

- [LangChain API Reference](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.fix.OutputFixingParser.html)
- [Pydantic Docs](https://docs.pydantic.dev/latest/api/base_model/)

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "08-OutputFixingParser",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```OPENAI_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```OPENAI_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Define Data Model and Set Up PydanticOutputParser

- The Actor class is defined using the Pydantic model, where name and film_names are fields representing the actor's name and a list of films they starred in.
- The ```PydanticOutputParser``` is used to parse outputs into an Actor object.

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Define the Actor class using Pydantic
class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


# A query to generate the filmography for a random actor
actor_query = "Generate the filmography for a random actor."

# Use PydanticOutputParser to parse the output into an Actor object
parser = PydanticOutputParser(pydantic_object=Actor)
```

### Attempt to Parse Misformatted Input Data

- The misformatted variable contains an incorrectly formatted string, which does not match the expected structure (using ' instead of ").
- Calling parser.parse() will result in an error because of the format mismatch.

```python
# Intentionally input misformatted data
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"
parser.parse(misformatted)

# An error will be printed because the data is incorrectly formatted
```


    ---------------------------------------------------------------------------

    JSONDecodeError                           Traceback (most recent call last)

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\output_parsers\json.py:83, in JsonOutputParser.parse_result(self, result, partial)
         82 try:
    ---> 83     return parse_json_markdown(text)
         84 except JSONDecodeError as e:
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\utils\json.py:144, in parse_json_markdown(json_string, parser)
        143     json_str = json_string if match is None else match.group(2)
    --> 144 return _parse_json(json_str, parser=parser)
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\utils\json.py:160, in _parse_json(json_str, parser)
        159 # Parse the JSON string into a Python dictionary
    --> 160 return parser(json_str)
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\utils\json.py:118, in parse_partial_json(s, strict)
        115 # If we got here, we ran out of characters to remove
        116 # and still couldn't parse the string as JSON, so return the parse error
        117 # for the original string.
    --> 118 return json.loads(s, strict=strict)
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\json\__init__.py:359, in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        358     kw['parse_constant'] = parse_constant
    --> 359 return cls(**kw).decode(s)
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\json\decoder.py:337, in JSONDecoder.decode(self, s, _w)
        333 """Return the Python representation of ``s`` (a ``str`` instance
        334 containing a JSON document).
        335 
        336 """
    --> 337 obj, end = self.raw_decode(s, idx=_w(s, 0).end())
        338 end = _w(s, end).end()
    

    File ~\AppData\Local\Programs\Python\Python311\Lib\json\decoder.py:353, in JSONDecoder.raw_decode(self, s, idx)
        352 try:
    --> 353     obj, end = self.scan_once(s, idx)
        354 except StopIteration as err:
    

    JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)

    
    The above exception was the direct cause of the following exception:
    

    OutputParserException                     Traceback (most recent call last)

    Cell In[6], line 3
          1 # Intentionally input misformatted data
          2 misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"
    ----> 3 parser.parse(misformatted)
          5 # An error will be printed because the data is incorrectly formatted
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\output_parsers\pydantic.py:83, in PydanticOutputParser.parse(self, text)
         74 def parse(self, text: str) -> TBaseModel:
         75     """Parse the output of an LLM call to a pydantic object.
         76 
         77     Args:
       (...)
         81         The parsed pydantic object.
         82     """
    ---> 83     return super().parse(text)
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\output_parsers\json.py:97, in JsonOutputParser.parse(self, text)
         88 def parse(self, text: str) -> Any:
         89     """Parse the output of an LLM call to a JSON object.
         90 
         91     Args:
       (...)
         95         The parsed JSON object.
         96     """
    ---> 97     return self.parse_result([Generation(text=text)])
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\output_parsers\pydantic.py:72, in PydanticOutputParser.parse_result(self, result, partial)
         70 if partial:
         71     return None
    ---> 72 raise e
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\output_parsers\pydantic.py:67, in PydanticOutputParser.parse_result(self, result, partial)
         54 """Parse the result of an LLM call to a pydantic object.
         55 
         56 Args:
       (...)
         64     The parsed pydantic object.
         65 """
         66 try:
    ---> 67     json_object = super().parse_result(result)
         68     return self._parse_obj(json_object)
         69 except OutputParserException as e:
    

    File c:\Users\Machine_K\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-fOxWcZdD-py3.11\Lib\site-packages\langchain_core\output_parsers\json.py:86, in JsonOutputParser.parse_result(self, result, partial)
         84 except JSONDecodeError as e:
         85     msg = f"Invalid json output: {text}"
    ---> 86     raise OutputParserException(msg, llm_output=text) from e
    

    OutputParserException: Invalid json output: {'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}
    For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE 


## Using OutputFixingParser to Correct Incorrect Formatting
### Set Up OutputFixingParser to Automatically Correct the Error
- ```OutputFixingParser``` wraps around the existing ```PydanticOutputParser``` and automatically fixes errors by making additional calls to the LLM.
- The from_llm() method connects ```OutputFixingParser``` with ```ChatOpenAI``` to correct the formatting issues in the output.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser

# Define a custom prompt to provide the fixing instructions
fixing_prompt = PromptTemplate(
    template=(
        "The following JSON is incorrectly formatted or incomplete: {completion}\n"
    ),
    input_variables=[
        "completion",
    ],
)

# Use OutputFixingParser to automatically fix the error
new_parser = OutputFixingParser.from_llm(
    parser=parser, llm=ChatOpenAI(model="gpt-4o"), prompt=fixing_prompt
)
```

```python
# Misformatted Output Data
misformatted
```




<pre class="custom">"{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"</pre>



### Parse the Misformatted Output Using OutputFixingParser
- The new_parser.parse() method is used to parse the misformatted data. OutputFixingParser will correct the errors in the data and generate a valid Actor object.

```python
actor = new_parser.parse(misformatted)

# Attempt to parse the misformatted JSON with Exception Handling
# try:
#     actor = new_parser.parse(misformatted)
#     print("Parsed actor:", actor)
# except Exception as e:
#     print("Error while parsing:", e)
```

### Check the Parsed Result
- After parsing, the result is a valid Actor object with the corrected format. The errors in the initial misformatted string have been automatically fixed by OutputFixingParser.

```python
actor
```




<pre class="custom">Actor(name='Tom Hanks', film_names=['Forrest Gump'])</pre>


