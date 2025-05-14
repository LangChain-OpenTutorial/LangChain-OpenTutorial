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

# WebBaseLoader

- Author: [Kane](https://github.com/HarryKane11)
- Design: [Kane](https://github.com/HarryKane11)
- Peer Review : [JoonHo Kim](https://github.com/jhboyo), [Sunyoung Park (architectyou)](https://github.com/Architectyou)
- Author: [Yejin Park](https://github.com/ppakyeah)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/03-WebBaseLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/03-WebBaseLoader.ipynb)
## Overview

```WebBaseLoader``` is a specialized document loader in LangChain designed for processing web-based content. 

It leverages the ```BeautifulSoup4``` library to parse web pages effectively, offering customizable parsing options through ```SoupStrainer``` and additional ```bs4``` parameters.

This tutorial demonstrates how to use ```WebBaseLoader``` to:
1. Load and parse web documents effectively
2. Customize parsing behavior using ```BeautifulSoup``` options
3. Handle different web content structures flexibly.

### Table of Contents 

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Load Web-based documents](#load-web-based-documents)
- [Load Multiple URLs Concurrently with alazy_load](#load-multiple-urls-concurrently-with-alazy_load)
- [Load XML Documents](#load-xml-documents)
- [Load Web based document Using Proxies](#load-web-based-document-using-proxies)
- [Simple Web Content Loading with MarkItDown](#simple-web-content-loading-with-markitdown)

### References

- [WebBaseLoader API Documentation](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)
- [BeautifulSoup4 Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain-opentutorial markitdown
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langchain_community",
    ],
    verbose=False,
    upgrade=False,
)
```

## Load Web-based documents

```WebBaseLoader``` is a loader designed for loading web-based documents.

It uses the ```bs4``` library to parse web pages.

Key Features:
- Uses ```bs4.SoupStrainer``` to specify elements to parse.
- Accepts additional arguments for ```bs4.SoupStrainer``` through the ```bs_kwargs``` parameter.

For more details, refer to the API documentation.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Load news article content using WebBaseLoader
loader = WebBaseLoader(
   web_paths=("https://techcrunch.com/2024/12/28/google-ceo-says-ai-model-gemini-will-the-companys-biggest-focus-in-2025/",),
   # Configure BeautifulSoup to parse only specific div elements
   bs_kwargs=dict(
       parse_only=bs4.SoupStrainer(
           "div",
           attrs={"class": ["entry-content wp-block-post-content is-layout-constrained wp-block-post-content-is-layout-constrained"]},
       )
   ),
   # Set user agent in request header to mimic browser
   header_template={
       "User_Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
   },
)

# Load and process the documents
docs = loader.load()
print(f"Number of documents: {len(docs)}")
docs[0]
```

<pre class="custom">Number of documents: 1
</pre>




    Document(metadata={'source': 'https://techcrunch.com/2024/12/28/google-ceo-says-ai-model-gemini-will-the-companys-biggest-focus-in-2025/'}, page_content='\nCEO Sundar Pichai reportedly told Google employees that 2025 will be a “critical” year for the company.\nCNBC reports that it obtained audio from a December 18 strategy meeting where Pichai and other executives put on ugly holiday sweaters and laid out their priorities for the coming year.\n\n\n\n\n\n\n\n\n“I think 2025 will be critical,” Pichai said. “I think it’s really important we internalize the urgency of this moment, and need to move faster as a company. The stakes are high.”\nThe moment, of course, is one where tech companies like Google are making heavy investments in AI, and often with mixed results. Pichai acknowledged that the company has some catching up to do on the AI side — he described the Gemini app (based on the company’s AI model of the same name) as having “strong momentum,” while also acknowledging “we have some work to do in 2025 to close the gap and establish a leadership position there as well.”\n“Scaling Gemini on the consumer side will be our biggest focus next year,” he said.\n')



To bypass SSL authentication errors, you can set the ```“verify”``` option.

```python
# Bypass SSL certificate verification
loader.requests_kwargs = {"verify": False}

# Load documents from the web
docs = loader.load()
docs[0]
```

<pre class="custom">c:\Users\gram\AppData\Local\pypoetry\Cache\virtualenvs\langchain-opentutorial-RXtDr8w5-py3.11\Lib\site-packages\urllib3\connectionpool.py:1097: InsecureRequestWarning: Unverified HTTPS request is being made to host 'techcrunch.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
      warnings.warn(
</pre>




    Document(metadata={'source': 'https://techcrunch.com/2024/12/28/google-ceo-says-ai-model-gemini-will-the-companys-biggest-focus-in-2025/'}, page_content='\nCEO Sundar Pichai reportedly told Google employees that 2025 will be a “critical” year for the company.\nCNBC reports that it obtained audio from a December 18 strategy meeting where Pichai and other executives put on ugly holiday sweaters and laid out their priorities for the coming year.\n\n\n\n\n\n\n\n\n“I think 2025 will be critical,” Pichai said. “I think it’s really important we internalize the urgency of this moment, and need to move faster as a company. The stakes are high.”\nThe moment, of course, is one where tech companies like Google are making heavy investments in AI, and often with mixed results. Pichai acknowledged that the company has some catching up to do on the AI side — he described the Gemini app (based on the company’s AI model of the same name) as having “strong momentum,” while also acknowledging “we have some work to do in 2025 to close the gap and establish a leadership position there as well.”\n“Scaling Gemini on the consumer side will be our biggest focus next year,” he said.\n')



You can also load multiple webpages at once. To do this, you can pass a list of **urls** to the loader, which will return a list of documents in the order of the **urls** passed.

```python
# Initialize the WebBaseLoader with web page paths and parsing configurations
loader = WebBaseLoader(
    web_paths=[
        # List of web pages to load
        "https://techcrunch.com/2024/12/28/revisiting-the-biggest-moments-in-the-space-industry-in-2024/",
        "https://techcrunch.com/2024/12/29/ai-data-centers-could-be-distorting-the-us-power-grid/",
    ],
    bs_kwargs=dict(
        # BeautifulSoup settings to parse only the specific content section
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["entry-content wp-block-post-content is-layout-constrained wp-block-post-content-is-layout-constrained"]},
        )
    ),
    header_template={
        # Custom HTTP headers for the request (e.g., User-Agent for simulating a browser)
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)

# Load the data from the specified web pages
docs = loader.load()

# Check and print the number of documents loaded
print(len(docs))

```

<pre class="custom">2
</pre>

Output the results fetched from the web.

```python
print(docs[0].page_content[:500])
print("===" * 10)
print(docs[1].page_content[:500])
```

<pre class="custom">
    We are at the dawn of a new space age.  If you doubt, simply look back at the last year: From SpaceX’s historic catch of the Super Heavy booster to the record-breaking number of lunar landing attempts, this year was full of historic and ambitious missions and demonstrations. 
    We’re taking a look back at the five most significant moments or trends in the space industry this year. Naysayers might think SpaceX is overrepresented on this list, but that just shows how far ahead the space behemoth is
    ==============================
    
    The proliferation of data centers aiming to meet the computational needs of AI could be bad news for the US power grid, according to a new report in Bloomberg.
    Using the 1 million residential sensors tracked by Whisker Labs, along with market intelligence data from DC Byte, Bloomberg found that more than half of the households showing the worst power distortions live within 20 miles of significant data center activity.
    
    
    
    
    
    
    
    
    In other words, there appears to be a link between data center proxi
</pre>

## Load Multiple URLs Concurrently with ```alazy_load()```

You can speed up the process of scraping and parsing multiple URLs by using asynchronous loading. This allows you to fetch documents concurrently, improving efficiency while adhering to rate limits.

### Key Points:
- **Rate Limit** : The ```requests_per_second``` parameter controls how many requests are made per second. In this example, it's set to 1 to avoid overloading the server.
- **Asynchronous Loading** : The ```alazy_load()``` function is used to load documents asynchronously, enabling faster processing of multiple URLs.
- **Jupyter Notebook Compatibility** : If running in Jupyter Notebook, ```nest_asyncio``` is required to handle asynchronous tasks properly.

The code below demonstrates how to configure and load documents asynchronously:


```python
# only for jupyter notebook (asyncio)
import nest_asyncio

nest_asyncio.apply()
```

```python
# Set the requests per second rate limit
loader.requests_per_second = 1

# Load documents asynchronously
# The aload() is deprecated and alazy_load() is used since the langchain 3.14 update)
docs=[]
async for doc in loader.alazy_load():
    docs.append(doc)
```

```python
# Display loaded documents
docs
```




<pre class="custom">[Document(metadata={'source': 'https://techcrunch.com/2024/12/28/google-ceo-says-ai-model-gemini-will-the-companys-biggest-focus-in-2025/'}, page_content='\nCEO Sundar Pichai reportedly told Google employees that 2025 will be a “critical” year for the company.\nCNBC reports that it obtained audio from a December 18 strategy meeting where Pichai and other executives put on ugly holiday sweaters and laid out their priorities for the coming year.\n\n\n\n\n\n\n\n\n“I think 2025 will be critical,” Pichai said. “I think it’s really important we internalize the urgency of this moment, and need to move faster as a company. The stakes are high.”\nThe moment, of course, is one where tech companies like Google are making heavy investments in AI, and often with mixed results. Pichai acknowledged that the company has some catching up to do on the AI side — he described the Gemini app (based on the company’s AI model of the same name) as having “strong momentum,” while also acknowledging “we have some work to do in 2025 to close the gap and establish a leadership position there as well.”\n“Scaling Gemini on the consumer side will be our biggest focus next year,” he said.\n')]</pre>



## Load XML Documents

```WebBaseLoader``` can process XML files by specifying a different ```BeautifulSoup``` parser. This is particularly useful when working with structured XML content like sitemaps or government data.

### Basic XML Loading

The following example demonstrates loading an XML document from a government website:

```python
from langchain_community.document_loaders import WebBaseLoader

# Initialize loader with XML document URL
loader = WebBaseLoader(
    "https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml"
)

# Set parser to XML mode
loader.default_parser = "xml"

# Load and process the document
docs = loader.load()
```

### Memory-Efficient Loading

For handling large documents, ```WebBaseLoader``` provides two memory-efficient loading methods:

1. lazy_load() - loads one page at a time
2. alazy_load() - asynchronous page loading for better performance

```python
# Lazy Loading Example
pages = []
for doc in loader.lazy_load():
    pages.append(doc)

# Print first 100 characters and metadata of the first page
print(pages[0].page_content[:100])
print(pages[0].metadata)
```

<pre class="custom">
    
    10
    Energy
    3
    2018-01-01
    2018-01-01
    false
    Uniform test method for the measurement of energy efficien
    {'source': 'https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml'}
</pre>

```python
# Async Loading Example
pages = []
async for doc in loader.alazy_load():
    pages.append(doc)

# Print first 100 characters and metadata of the first page
print(pages[0].page_content[:100])
print(pages[0].metadata)
```

<pre class="custom">
    
    10
    Energy
    3
    2018-01-01
    2018-01-01
    false
    Uniform test method for the measurement of energy efficien
    {'source': 'https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml'}
</pre>

## Load Web-based Document Using Proxies

Sometimes you may need to use proxies to bypass IP blocking.

To use a proxy, you can pass a proxy dictionary to the loader (and its underlying ```requests``` library).

### ⚠️ Warning:
- Replace ```{username}```, ```{password}```, and ```proxy.service.com``` with your actual proxy credentials and server information.
- Without a valid proxy configuration, errors such as **ProxyError** or **AuthenticationError** may occur.

```python
loader = WebBaseLoader(
   "https://www.google.com/search?q=parrots",
   proxies={
       "http": "http://{username}:{password}:@proxy.service.com:6666/",
       "https": "https://{username}:{password}:@proxy.service.com:6666/",
   },
   # Initialize the web loader with proxy settings
   # Configure proxy for both HTTP and HTTPS requests
)

# Load documents using the proxy
docs = loader.load()
```

## Simple Web Content Loading with ```MarkItDown```

Unlike ```WebBaseLoader``` which uses ```BeautifulSoup4``` for sophisticated HTML parsing, ```MarkItDown``` provides a naive but simpler approach to web content loading. It directly fetches web content using HTTP requests and transfrom it into markdown format without detailed parsing capabilities.

Below is a basic example of loading web content using ```MarkItDown```:

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("https://techcrunch.com/2024/12/28/revisiting-the-biggest-moments-in-the-space-industry-in-2024/")
result_text = result.text_content
```

```python
print(result_text[:1000])
```

<pre class="custom">
    
    [![](https://techcrunch.com/wp-content/uploads/2024/09/tc-lockup.svg) TechCrunch Desktop Logo](https://techcrunch.com)
    
    [![](https://techcrunch.com/wp-content/uploads/2024/09/tc-logo-mobile.svg) TechCrunch Mobile Logo](https://techcrunch.com)
    
    * [Latest](/latest/)
    * [Startups](/category/startups/)
    * [Venture](/category/venture/)
    * [Apple](/tag/apple/)
    * [Security](/category/security/)
    * [AI](/category/artificial-intelligence/)
    * [Apps](/category/apps/)
    
    * [Events](/events/)
    * [Podcasts](/podcasts/)
    * [Newsletters](/newsletters/)
    
    [Sign In](https://oidc.techcrunch.com/login/?dest=https%3A%2F%2Ftechcrunch.com%2F2024%2F12%2F28%2Frevisiting-the-biggest-moments-in-the-space-industry-in-2024%2F)
    [![]()](https://techcrunch.com/my-account/)
    
    SearchSubmit
    
    Site Search Toggle
    
    Mega Menu Toggle
    
    ### Topics
    
    [Latest](/latest/)
    
    [AI](/category/artificial-intelligence/)
    
    [Amazon](/tag/amazon/)
    
    [Apps](/category/apps/)
    
    [Biotech & Health](/category/biotech-health/)
    
    [Climate](/category/climate/)
    
    [
</pre>
