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

# Split code with Langchain

- Author: [Jongcheol Kim](https://github.com/greencode-99)
- Peer Review: [kofsitho87](https://github.com/kofsitho87), [teddylee777](https://github.com/teddylee777)
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/05-CodeSplitter.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/05-CodeSplitter.ipynb)
## Overview

```RecursiveCharacterTextSplitter``` includes pre-built separator lists optimized for splitting text in different programming languages.

The ```CodeTextSplitter``` provides even more specialized functionality for splitting code.

To use it, import the ```Language``` enum(enumeration) and specify the desired programming language.


### Table of Contents

- [Overview](#Overview)
- [Environment Setup](#environment-setup)
- [Code Spliter Examples](#code-splitter-examples)
   - [Python](#python)
   - [JavaScript](#javascript)
   - [TypeScript](#typescript)
   - [Markdown](#markdown)
   - [LaTeX](#latex)
   - [HTML](#html)
   - [Solidity](#solidity)
   - [C#](#c)
   - [PHP](#php)
   - [Kotlin](#kotlin)


### References
- [How to split code](https://python.langchain.com/docs/how_to/code_splitter/)
----

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
        "langchain_text_splitters",
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
        "LANGCHAIN_PROJECT": "Code-Splitter",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv()
```




<pre class="custom">True</pre>



## Code Splitter Examples

Here is an example of splitting text using the ```RecursiveCharacterTextSplitter```.

- Import the ```Language``` and ```RecursiveCharacterTextSplitter``` classes from the ```langchain_text_splitters``` module.
- ```RecursiveCharacterTextSplitter``` is a text splitter that recursively splits text at the character level.

```python
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
```

Supported languages are stored in the langchain_text_splitters.Language enum. 

API Reference: [Language](https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.Language.html#language) | [RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#recursivecharactertextsplitter)

See below for the full list of supported languages.

```python
# Get the full list of supported languages.
[e.value for e in Language]
```




<pre class="custom">['cpp',
     'go',
     'java',
     'kotlin',
     'js',
     'ts',
     'php',
     'proto',
     'python',
     'rst',
     'ruby',
     'rust',
     'scala',
     'swift',
     'markdown',
     'latex',
     'html',
     'sol',
     'csharp',
     'cobol',
     'c',
     'lua',
     'perl',
     'haskell',
     'elixir',
     'powershell']</pre>



You can use the ```get_separators_for_language``` method of the ```RecursiveCharacterTextSplitter``` class to see the separators used for a given language.

- For example, passing ```Language.PYTHON``` retrieves the separators used for Python:

```python
# You can check the separators used for the given language.
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
```




<pre class="custom">['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']</pre>



### Python

Here's how to split Python code into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.PYTHON``` for the ```language``` parameter. It tells the splitter you're working with Python code.
- Then, set ```chunk_size``` to 50. This limits the size of each resulting chunk to a maximum of 50 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.

```python
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

hello_world()
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)

# Create `Document`. The created `Document` is returned as a list.
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs
```




<pre class="custom">[Document(metadata={}, page_content='def hello_world():\n    print("Hello, World!")'),
     Document(metadata={}, page_content='hello_world()')]</pre>



```python
# This section iterates through the list of documents created by the RecursiveCharacterTextSplitter
# and prints each document's content followed by a separator line for readability.
for doc in python_docs:
    print(doc.page_content, end="\n==================\n")
```

<pre class="custom">def hello_world():
        print("Hello, World!")
    ==================
    hello_world()
    ==================
</pre>

### JavaScript

Here's how to split JavaScript code into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.JS``` for the ```language``` parameter. It tells the splitter you're working with JavaScript code.
- Then, set ```chunk_size``` to 60. This limits the size of each resulting chunk to a maximum of 60 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.


```python
JS_CODE = """
function helloWorld() {
  console.log("Hello, World!");
}

helloWorld();
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)

# Create `Document`. The created `Document` is returned as a list.
js_docs = js_splitter.create_documents([JS_CODE])
js_docs
```




<pre class="custom">[Document(metadata={}, page_content='function helloWorld() {\n  console.log("Hello, World!");\n}'),
     Document(metadata={}, page_content='helloWorld();')]</pre>



### TypeScript

Here's how to split TypeScript code into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.TS``` for the ```language``` parameter. It tells the splitter you're working with TypeScript code.
- Then, set ```chunk_size``` to 60. This limits the size of each resulting chunk to a maximum of 60 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.


```python
TS_CODE = """
function helloWorld(): void {
  console.log("Hello, World!");
}

helloWorld();
"""

ts_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.TS, chunk_size=60, chunk_overlap=0
)


ts_docs = ts_splitter.create_documents([TS_CODE])
ts_docs
```




<pre class="custom">[Document(metadata={}, page_content='function helloWorld(): void {'),
     Document(metadata={}, page_content='console.log("Hello, World!");\n}'),
     Document(metadata={}, page_content='helloWorld();')]</pre>



### Markdown

Here's how to split Markdown text into smaller chunks using the ```RecursiveCharacterTextSplitter```.

- First, Specify ```Language.MARKDOWN``` for the ```language``` parameter. It tells the splitter you're working with Markdown text.
- Then, set ```chunk_size``` to 60. This limits the size of each resulting chunk to a maximum of 60 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.

```python
markdown_text = """
# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

## What is LangChain?

# Hopefully this code block isn't split
LangChain is a framework for...

As an open-source project in a rapidly developing field, we are extremely open to contributions.
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=60,
    chunk_overlap=0,
)

md_docs = md_splitter.create_documents([markdown_text])
md_docs
```




<pre class="custom">[Document(metadata={}, page_content='# 🦜️🔗 LangChain'),
     Document(metadata={}, page_content='⚡ Building applications with LLMs through composability ⚡'),
     Document(metadata={}, page_content='## What is LangChain?'),
     Document(metadata={}, page_content="# Hopefully this code block isn't split"),
     Document(metadata={}, page_content='LangChain is a framework for...'),
     Document(metadata={}, page_content='As an open-source project in a rapidly developing field, we'),
     Document(metadata={}, page_content='are extremely open to contributions.')]</pre>



### LaTeX

LaTeX is a markup language for document creation, widely used for representing mathematical symbols and formulas.

Here's how to split LaTeX text into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.LATEX``` for the ```language``` parameter. It tells the splitter you're working with LaTeX text.
- Then, set ```chunk_size``` to 60. This limits the size of each resulting chunk to a maximum of 60 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.

```python
latex_text = """
\documentclass{article}

\begin{document}

\maketitle

\section{Introduction}
Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.

\subsection{History of LLMs}
The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

\subsection{Applications of LLMs}
LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

\end{document}
"""

latex_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.LATEX,
    chunk_size=60,
    chunk_overlap=0,
)

latex_docs = latex_splitter.create_documents([latex_text])
latex_docs
```




<pre class="custom">[Document(metadata={}, page_content='\\documentclass{article}\n\n\x08egin{document}\n\n\\maketitle'),
     Document(metadata={}, page_content='\\section{Introduction}\nLarge language models (LLMs) are a'),
     Document(metadata={}, page_content='type of machine learning model that can be trained on vast'),
     Document(metadata={}, page_content='amounts of text data to generate human-like language. In'),
     Document(metadata={}, page_content='recent years, LLMs have made significant advances in a'),
     Document(metadata={}, page_content='variety of natural language processing tasks, including'),
     Document(metadata={}, page_content='language translation, text generation, and sentiment'),
     Document(metadata={}, page_content='analysis.'),
     Document(metadata={}, page_content='\\subsection{History of LLMs}\nThe earliest LLMs were'),
     Document(metadata={}, page_content='developed in the 1980s and 1990s, but they were limited by'),
     Document(metadata={}, page_content='the amount of data that could be processed and the'),
     Document(metadata={}, page_content='computational power available at the time. In the past'),
     Document(metadata={}, page_content='decade, however, advances in hardware and software have'),
     Document(metadata={}, page_content='made it possible to train LLMs on massive datasets, leading'),
     Document(metadata={}, page_content='to significant improvements in performance.'),
     Document(metadata={}, page_content='\\subsection{Applications of LLMs}\nLLMs have many'),
     Document(metadata={}, page_content='applications in industry, including chatbots, content'),
     Document(metadata={}, page_content='creation, and virtual assistants. They can also be used in'),
     Document(metadata={}, page_content='academia for research in linguistics, psychology, and'),
     Document(metadata={}, page_content='computational linguistics.\n\n\\end{document}')]</pre>



### HTML

Here's how to split HTML text into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.HTML``` for the ```language``` parameter. It tells the splitter you're working with HTML.
- Then, set ```chunk_size``` to 60. This limits the size of each resulting chunk to a maximum of 60 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.


```python
html_text = """
<!DOCTYPE html>
<html>
    <head>
        <title>🦜️🔗 LangChain</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            h1 {
                color: darkblue;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🦜️🔗 LangChain</h1>
            <p>⚡ Building applications with LLMs through composability ⚡</p>
        </div>
        <div>
            As an open-source project in a rapidly developing field, we are extremely open to contributions.
        </div>
    </body>
</html>
"""

html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=60, chunk_overlap=0
)

html_docs = html_splitter.create_documents([html_text])
html_docs
```




<pre class="custom">[Document(metadata={}, page_content='<!DOCTYPE html>\n<html>'),
     Document(metadata={}, page_content='<head>\n        <title>🦜️🔗 LangChain</title>'),
     Document(metadata={}, page_content='<style>\n            body {\n                font-family: Aria'),
     Document(metadata={}, page_content='l, sans-serif;\n            }\n            h1 {'),
     Document(metadata={}, page_content='color: darkblue;\n            }\n        </style>\n    </head'),
     Document(metadata={}, page_content='>'),
     Document(metadata={}, page_content='<body>'),
     Document(metadata={}, page_content='<div>\n            <h1>🦜️🔗 LangChain</h1>'),
     Document(metadata={}, page_content='<p>⚡ Building applications with LLMs through composability ⚡'),
     Document(metadata={}, page_content='</p>\n        </div>'),
     Document(metadata={}, page_content='<div>\n            As an open-source project in a rapidly dev'),
     Document(metadata={}, page_content='eloping field, we are extremely open to contributions.'),
     Document(metadata={}, page_content='</div>\n    </body>\n</html>')]</pre>



### Solidity

Here's how to split Solidity code (sotred as a string in the ```SOL_CODE``` variable) into smaller chunks by creating a ```RecursiveCharacterTextSplitter``` instance called ```sol_splitter``` to handle the splitting.
- First, specify ```Language.SOL``` for the ```language``` parameter. It tells the splitter you're working with Solidity code.
- Then, set ```chunk_size``` to 128. This limits the size of each resulting chunk to a maximum of 128 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.
- The ```sol_splitter.create_documents()``` method splits the Solidity code(```SOL_CODE```) into chunks and stores them in the ```sol_docs``` variable.
- Print or display the output(```sol_docs```) to verify the split.


```python
SOL_CODE = """
pragma solidity ^0.8.20; 
contract HelloWorld {  
   function add(uint a, uint b) pure public returns(uint) {
       return a + b;
   }
}
"""

sol_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.SOL, chunk_size=128, chunk_overlap=0
)

sol_docs = sol_splitter.create_documents([SOL_CODE])
sol_docs
```




<pre class="custom">[Document(metadata={}, page_content='pragma solidity ^0.8.20;'),
     Document(metadata={}, page_content='contract HelloWorld {  \n   function add(uint a, uint b) pure public returns(uint) {\n       return a + b;\n   }\n}')]</pre>



### C#

Here's how to split C# code into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.CSHARP``` for the ```language``` parameter. It tells the splitter you're working with C# code.
- Then, set ```chunk_size``` to 128. This limits the size of each resulting chunk to a maximum of 128 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.

```python
C_CODE = """
using System;
class Program
{
    static void Main()
    {
        Console.WriteLine("Enter a number (1-5):");
        int input = Convert.ToInt32(Console.ReadLine());
        for (int i = 1; i <= input; i++)
        {
            if (i % 2 == 0)
            {
                Console.WriteLine($"{i} is even.");
            }
            else
            {
                Console.WriteLine($"{i} is odd.");
            }
        }
        Console.WriteLine("Goodbye!");
    }
}
"""

c_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.CSHARP, chunk_size=128, chunk_overlap=0
)

c_docs = c_splitter.create_documents([C_CODE])
c_docs
```




<pre class="custom">[Document(metadata={}, page_content='using System;'),
     Document(metadata={}, page_content='class Program\n{\n    static void Main()\n    {\n        Console.WriteLine("Enter a number (1-5):");'),
     Document(metadata={}, page_content='int input = Convert.ToInt32(Console.ReadLine());\n        for (int i = 1; i <= input; i++)\n        {'),
     Document(metadata={}, page_content='if (i % 2 == 0)\n            {\n                Console.WriteLine($"{i} is even.");\n            }\n            else'),
     Document(metadata={}, page_content='{\n                Console.WriteLine($"{i} is odd.");\n            }\n        }\n        Console.WriteLine("Goodbye!");'),
     Document(metadata={}, page_content='}\n}')]</pre>



### PHP

Here's how to split PHP code into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.PHP``` for the ```language``` parameter. It tells the splitter you're working with PHP code.
- Then, set ```chunk_size``` to 50. This limits the size of each resulting chunk to a maximum of 50 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.

```python
PHP_CODE = """<?php
namespace foo;
class Hello {
    public function __construct() { }
}
function hello() {
    echo "Hello World!";
}
interface Human {
    public function breath();
}
trait Foo { }
enum Color
{
    case Red;
    case Blue;
}"""

php_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PHP, chunk_size=50, chunk_overlap=0
)

php_docs = php_splitter.create_documents([PHP_CODE])
php_docs
```




<pre class="custom">[Document(metadata={}, page_content='<?php\nnamespace foo;'),
     Document(metadata={}, page_content='class Hello {'),
     Document(metadata={}, page_content='public function __construct() { }\n}'),
     Document(metadata={}, page_content='function hello() {\n    echo "Hello World!";\n}'),
     Document(metadata={}, page_content='interface Human {\n    public function breath();\n}'),
     Document(metadata={}, page_content='trait Foo { }\nenum Color\n{\n    case Red;'),
     Document(metadata={}, page_content='case Blue;\n}')]</pre>



### Kotlin

Here's how to split Kotline code into smaller chunks using the ```RecursiveCharacterTextSplitter```.
- First, specify ```Language.KOTLIN``` for the ```language``` parameter. It tells the splitter you're working with Kotline code.
- Then, set ```chunk_size``` to 100. This limits the size of each resulting chunk to a maximum of 100 characters.
- Finally, set ```chunk_overlap``` to 0. It prevents any of the chunks from overlapping.

```python
KOTLIN_CODE = """
fun main() {
    val directoryPath = System.getProperty("user.dir")
    val files = File(directoryPath).listFiles()?.filter { !it.isDirectory }?.sortedBy { it.lastModified() } ?: emptyArray()

    files.forEach { file ->
        println("Name: ${file.name} | Last Write Time: ${file.lastModified()}")
    }
}
"""

kotlin_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.KOTLIN, chunk_size=100, chunk_overlap=0
)

kotlin_docs = kotlin_splitter.create_documents([KOTLIN_CODE])
kotlin_docs
```




<pre class="custom">[Document(metadata={}, page_content='fun main() {\n    val directoryPath = System.getProperty("user.dir")'),
     Document(metadata={}, page_content='val files = File(directoryPath).listFiles()?.filter { !it.isDirectory }?.sortedBy {'),
     Document(metadata={}, page_content='it.lastModified() } ?: emptyArray()'),
     Document(metadata={}, page_content='files.forEach { file ->'),
     Document(metadata={}, page_content='println("Name: ${file.name} | Last Write Time: ${file.lastModified()}")\n    }\n}')]</pre>


