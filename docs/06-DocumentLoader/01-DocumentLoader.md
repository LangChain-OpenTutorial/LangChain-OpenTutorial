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

# Document & Document Loader

- Author: [Jaemin Hong](https://github.com/geminii01)
- Peer Review : [Taylor(Jihyun Kim)](https://github.com/Taylor0819), [Yejin Park](https://github.com/ppakyeah)
- Proofread : [Q0211](https://github.com/Q0211)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/01-DocumentLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/01-DocumentLoader.ipynb)
## Overview

This tutorial covers the fundamental methods for loading Documents.

By completing this tutorial, you will learn how to load Documents and check their content and associated metadata.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Document](#document)
- [Document Loader](#document-loader)

### References

- [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html)
- [Load Documents](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.base.BaseLoader.html#langchain_core.document_loaders.base.BaseLoader)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
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
        "langchain_core",
        "langchain_community",
        "langchain_text_splitters",
        "pypdf",
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
        "LANGCHAIN_PROJECT": "01-DocumentLoader",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Document

Class for storing a piece of text and its associated metadata.

- ```page_content``` (Required): Stores a piece of text as a string.
- ```metadata``` (Optional): Stores metadata related to ```page_content``` as a dictionary.

```python
from langchain_core.documents import Document

document = Document(page_content="Hello, welcome to LangChain Open Tutorial!")

# Check the attributes using __dict__
document.__dict__
```




<pre class="custom">{'id': None,
     'metadata': {},
     'page_content': 'Hello, welcome to LangChain Open Tutorial!',
     'type': 'Document'}</pre>



The metadata is empty. Let's add some values.

```python
# Add metadata
document.metadata["source"] = "./example-file.pdf"
document.metadata["page"] = 0

# Check metadata
document.metadata
```




<pre class="custom">{'source': './example-file.pdf', 'page': 0}</pre>



## Document Loader

Document Loader is a class that loads Documents from various sources.

Listed below are some examples of Document Loaders.

- ```PyPDFLoader```: Loads PDF files
- ```CSVLoader```: Loads CSV files
- ```UnstructuredHTMLLoader```: Loads HTML files
- ```JSONLoader```: Loads JSON files
- ```TextLoader```: Loads text files
- ```DirectoryLoader```: Loads documents from a directory

Now, let's learn how to load Documents .

```python
# Example file path
FILE_PATH = "./data/01-document-loader-sample.pdf"
```

```python
from langchain_community.document_loaders import PyPDFLoader

# Set up the loader
loader = PyPDFLoader(FILE_PATH)
```

### ```load()```

- Loads Documents and returns them as a ```list[Document]```.

```python
# Load Documents
docs = loader.load()
```

```python
# Check the number of loaded Documents
len(docs)
```




<pre class="custom">48</pre>



```python
# Check Documents
docs[0:10]
```




<pre class="custom">[Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 0}, page_content=' \n \n \nOctober  2016 \n \n \n \n \n \n \n \n \n \nTHE NATIONAL  \nARTIFICIAL INTELLIGENCE \nRESEARCH AND DEVELOPMENT \nSTRATEGIC PLAN  \nNational Science and Technology Council  \n \nNetworking and Information Technology \nResearch and Development Subcommittee  \n '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 1}, page_content=' ii  \n '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content=' \n  \n iii About the National Science and Technology Council  \nThe National Science and Technology Council (NSTC) is the principal means by which the Executive \nBranch coordinates science and technology policy across the diverse entities that make up the Federal \nresearch and development (R&D) enterprise . One of the NSTC’s primary objectives is establishing clear \nnational goal s for Federal science and technology investments . The NSTC prepares R&D packages aimed \nat accomplishing multiple national goals . The NSTC’s work is organized under five committees: \nEnvironment, Natural Resources, and Sustainability; Homeland and National S ecurity; Science, \nTechnology, Engineering, and Mathematics (STEM) Education; Science; and Technology . Each of these \ncommittees oversees subcommittees and working groups that are focused on different aspects of \nscience and technology . More information is available at  www.whitehouse.gov/ostp/nstc . \nAbout the Office of Science and Technology Policy  \nThe Office of Science and Technology Policy (OSTP) was established by the National Science and \nTechnology Policy, Organization, and Priorities Act of 1976 . The mission of OSTP is threefold; first, to \nprovide the President and his senior staff with accurate, relevant, and timely scientific and technical advice \non all matters of consequence; second, to ensure th at the policies of the Executive Branch are informed by \nsound science; and third, to ensure that the scientific and technical work of the Executive Branch is \nproperly coordinated so as to provide the greatest benefit to society.  The Director of OSTP also s erves as \nAssistant to the President for Science and Technology and manages the NSTC . More information is \navailable at www.whitehouse.gov/ostp . \nAbout the Subcommittee on Networking and Information Technology  Research and \nDevelopment  \nThe Subcommittee on Networking and Information Technology Research and Development (NITRD) is a \nbody under the Committee on Technology (CoT ) of the National Science and Technology Council (NSTC). \nThe NITRD Subcommittee coordinates multiagency research and development programs to help assure \ncontinued U.S. leadership in networking and information technology, satisfy the needs of the Federal \nGovernment for advanced networking and information technology, and accelerate development and \ndeployment of advanced networking and information technology. It also implements relevant provisions \nof the High -Performance Computing Act of 1991 (P.L. 102 -194), a s amended by the Next Generation \nInternet Research Act of 1998 (P. L. 105 -305), and the America Creating Opportunities to Meaningfully \nPromote Excellence in Technology, Education and Science (COMPETES) Act of 2007 (P.L. 110 -69). For \nmore information, see www.nitrd.gov . \nAcknowledgments  \nThis document was developed through the contributions of the members and staff of the NITRD Task \nForce on Artificial Intelligence. A special thanks and appreciation to additional contribut ors who helped \nwrite, edit, and review the document: Chaitan Baru (NSF), Eric Daimler (Presidential Innovation Fellow), \nRonald Ferguson (DoD), Nancy Forbes (NITRD), Eric Harder (DHS), Erin Kenneally (DHS), Dai Kim (DoD), \nTatiana Korelsky (NSF), David Kuehn  (DOT), Terence Langendoen (NSF), Peter Lyster (NITRD), KC Morris \n(NIST), Hector Munoz -Avila (NSF), Thomas Rindflesch (NIH), Craig Schlenoff (NIST), Donald Sofge (NRL) , \nand Sylvia Spengler (NSF).  \n  '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 3}, page_content=' \n  \n iv Copyright Information  \nThis is a work of the U.S. Government and is in the public domain. It may be freely distributed, copied, \nand translated; acknowledgment of publication by the Office of Science and Technology Policy  is \nappreciated. Any translation should include a disclaime r that the accuracy of the translation is the \nresponsibility of the translator and not OSTP . It is requested that a copy of any translation be sent to \nOSTP . This work is available for worldwide use and reuse and under the Creative Commons CC0 1.0 \nUniversal  license.  \n  '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 4}, page_content=''),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 5}, page_content=' \n  \n vi  \n  '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 6}, page_content=' \n  \n vii National Science and Technology Council  \nChair  \nJohn P. Holdren  \nAssistant to the President for Science and \nTechnology and Director, Office of Science and \nTechnology PolicyStaff  \nAfua Bruce  \nExecutive  Director  \nOffice of Science and Technology  Policy  \nSubcommittee on  \nMachine Learning and Artificial Intelligence  \nCo-Chair  \nEd Felten  \nDeputy U.S. Chief Technology Officer  \nOffice of Science and Technology Policy  \n Co-Chair  \nMichael Garris  \nSenior Scientist  \nNational Institute of Standards and Technology  \nU.S. Department of Commerce  \nSubcommittee on  \nNetworking and Information Technology Research and Development  \nCo-Chair  \nBryan Biegel  \nDirector, National C oordination Office for  \nNetworking and Information Technology  \nResearch and Development  Co-Chair  \nJames Kurose  \nAssistant Director, Computer and Information \nScience and Engineering  \nNational Science Foundation  \nNetworking and Information Technology Research and Development  \nTask Force on Artificial Intelligence  \n \nCo-Chair  \nLynne Parker  \nDivision Director  \nInformation and Intelligent System s \nNational Science Foundation  Co-Chair  \nJason Matheny  \nDirector  \nIntelligence Advanced Research Projects Activity   \n \nMembers   \nMilton Corn  \nNational Institutes of Health   \nNikunj Oza  \nNational Aeronautics and Space Administration  \nWilliam Ford  \nNational Institute of Justice  Robinson Pino  \nDepartment of Energy  \nMichael Garris  \nNational Institute of Standards  and Technology  Gregory Shannon  \nOffice of Science and Technology Policy  \nSteven Knox  \nNational Security Agency  Scott Tousley  \nDepartment of Homeland Security  '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 7}, page_content=' \nviii \n John Launchbury  \nDefense Advanced Research Projects Agency  Faisal D’Souza  \nTechnical Coordinator  \nNational Coordination Office for Networking and \nInformation Technology Research and Development  Richard Linderman  \nOffice of the Secretary of Defense  \n '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 8}, page_content='NATIONAL ARTIFICIAL INTELLIGENCE RESEARCH AND DEVELOPMENT STRATEGIC PLAN  \n \n 1 Contents  \nAbout the National Science and Technology Council  ................................ ................................ ..........................  iii \nAbout the Office of Science and Technology Policy  ................................ ................................ ............................  iii \nAbout the Subcommittee on Networking and Information Technology Research and Development  ................  iii \nAcknowledgments  ................................ ................................ ................................ ................................ ...............  iii \nCopyright Information  ................................ ................................ ................................ ................................ .......... iv \nNational Science and Technology Council  ................................ ................................ ................................ ...........  vii \nSubcommittee on Machine Learning and Artificial Intelligence  ................................ ................................ ..........  vii \nSubcommittee on Networking and Information Technology Research and Development  ................................ . vii \nTask Force on  Artificial Intelligence  ................................ ................................ ................................ .....................  vii \nExecutive Summary  ................................ ................................ ................................ ................................ ...................  3 \nIntroduction  ................................ ................................ ................................ ................................ ...............................  5 \nPurpose of the National AI R&D Strategic Plan  ................................ ................................ ................................  5 \nDesired Outcome  ................................ ................................ ................................ ................................ .............  7 \nA Vision for Advancing our National Priorities with AI  ................................ ................................ ....................  8 \nCurrent State of AI ................................ ................................ ................................ ................................ ..........  12 \nR&D Strategy  ................................ ................................ ................................ ................................ ...........................  15 \nStrategy 1: Make  Long-Term Investments in AI Research  ................................ ................................ .............  16 \nStrategy 2: Develop Effective Methods for Human -AI Collaboration   ................................ ...........................  22 \nStrategy 3 : Understand and Address the Ethical, Le gal, and Societal Implications of AI  ...............................  26 \nStrategy 4 : Ensure the Safety and Security of AI Systems  ................................ ................................ ..............  27 \nStrategy 5: Develop Shared Public Datasets and Environments for AI Training and Testing  .........................  30 \nStrategy 6: Measure and Evaluate AI Technologies through Standards and Benchmarks .............................  32 \nStrategy 7 : Better Understand the National AI R&D Workforce Needs  ................................ .........................  35 \nRecommendations  ................................ ................................ ................................ ................................ ...................  37 \nAcronyms  ................................ ................................ ................................ ................................ ................................ . 39 '),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 9}, page_content='NATIONAL ARTIFICIAL INTELLIGENCE RESEARCH AND DEVELOPMENT STRATEGIC PLAN  \n \n 2   ')]</pre>



### ```aload()```

- Asynchronously loads Documents and returns them as a ```list[Document]```.

```python
# Load Documents asynchronously
docs = await loader.aload()
```

### ```load_and_split()```

- Loads Documents and automatically splits them into chunks using TextSplitter , and returns them as a ```list[Document]```.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up the TextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)

# Split Documents into chunks
docs = loader.load_and_split(text_splitter=text_splitter)
```

```python
# Check the number of loaded Documents
len(docs)
```




<pre class="custom">1441</pre>



```python
# Check Documents
docs[0:10]
```




<pre class="custom">[Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 0}, page_content='October  2016 \n \n \n \n \n \n \n \n \n \nTHE NATIONAL  \nARTIFICIAL INTELLIGENCE \nRESEARCH AND DEVELOPMENT \nSTRATEGIC PLAN'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 0}, page_content='National Science and Technology Council  \n \nNetworking and Information Technology \nResearch and Development Subcommittee'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 1}, page_content='ii'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content='iii About the National Science and Technology Council'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content='The National Science and Technology Council (NSTC) is the principal means by which the Executive'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content='Branch coordinates science and technology policy across the diverse entities that make up the Federal'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content='research and development (R&D) enterprise . One of the NSTC’s primary objectives is establishing clear'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content='national goal s for Federal science and technology investments . The NSTC prepares R&D packages aimed'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content='at accomplishing multiple national goals . The NSTC’s work is organized under five committees:'),
     Document(metadata={'source': './data/01-document-loader-sample.pdf', 'page': 2}, page_content='Environment, Natural Resources, and Sustainability; Homeland and National S ecurity; Science,')]</pre>



### ```lazy_load()```

- Loads Documents sequentially and returns them as an ```Iterator[Document]```.

```python
loader.lazy_load()
```




<pre class="custom"><generator object PyPDFLoader.lazy_load at 0x000001902A0117B0></pre>



It can be observed that this method operates as a ```generator```. This is a special type of iterator that produces values on-the-fly, without storing them all in memory at once.

```python
# Load Documents sequentially
docs = loader.lazy_load()
for doc in docs:
    print(doc.metadata)
    break  # Used to limit the output length
```

<pre class="custom">{'source': './data/01-document-loader-sample.pdf', 'page': 0}
</pre>

### ```alazy_load()```

- Asynchronously loads Documents sequentially and returns them as an ```AsyncIterator[Document]```.

```python
loader.alazy_load()
```




<pre class="custom"><async_generator object BaseLoader.alazy_load at 0x000001902A00B140></pre>



It can be observed that this method operates as an ```async_generator```. This is a special type of asynchronous iterator that produces values on-the-fly, without storing them all in memory at once.

```python
# Load Documents asynchronously and sequentially
docs = loader.alazy_load()
async for doc in docs:
    print(doc.metadata)
    break  # Used to limit the output length
```

<pre class="custom">{'source': './data/01-document-loader-sample.pdf', 'page': 0}
</pre>
