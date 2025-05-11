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

# HTMLHeaderTextSplitter  

- Author: [ChangJun Lee](https://www.linkedin.com/in/cjleeno1/)
- Peer Review: [YooKyung Jeon](https://github.com/sirena1), [Wooseok Jeong](https://github.com/jeong-wooseok)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/07-HTMLHeaderTextSplitter.ipynb)
[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/07-TextSplitter/07-HTMLHeaderTextSplitter.ipynb)

## Overview

This is a "structure-aware" chunk generator that splits text at the element level and adds metadata for each header, conceptually similar to the <a href="https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/markdown_header_metadata">`MarkdownHeaderTextSplitter`</a>.

It adds metadata "related" to each chunk.

`HTMLHeaderTextSplitter` can return chunks by element or combine elements with the same metadata,

- (a) semantically (approximately) group related text and
- (b) preserve context-rich information encoded in the document structure.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Using HTML Strings](#using-html-strings)
- [Connecting with Other Splitters and Loading HTML from a Web URL](#connecting-with-other-splitters-and-loading-html-from-a-web-url)
- [Limitations](#limitations)

### References

- [HTML Header Text Splitter](https://python.langchain.com/api_reference/text_splitters/html/langchain_text_splitters.html.HTMLHeaderTextSplitter.html#)
---

## Environment Setup

Setting up your environment is the first step. See the [Environment Setup](https://wikidocs.net/257836) guide for more details.

**[Note]**
- The `langchain-opentutorial` is a package of easy-to-use environment setup guidance, useful functions and utilities for tutorials.
- Check out the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

```python
%%capture --no-stderr
%pip install langchain langchain-opentutorial
```

```python
# Install required packages
from langchain_opentutorial import package

package.install(
    [
        "langsmith",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
    ]
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
        "LANGCHAIN_PROJECT": "RecursiveJsonSplitter",
    }
)
```

## Using HTML Strings

- Specify the header tags and their names to split on in the `headers_to_split_on` list as tuples.
- Create an `HTMLHeaderTextSplitter` object and pass the list of headers to split on to the `headers_to_split_on` parameter.

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Header 1</h1>
        <p>Text included in Header 1</p>
        <div>
            <h2>Header 2-1 Title</h2>
            <p>Text included in Header 2-1</p>
            <h3>Header 3-1 Title</h3>
            <p>Text included in Header 3-1</p>
            <h3>Header 3-2 Title</h3>
            <p>Text included in Header 3-2</p>
        </div>
        <div>
            <h2>Header 2-2 Title</h2>
            <p>Text included in Header 2-2</p>
        </div>
        <br>
        <p>Last content</p>
    </div>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),  # Specify the header tags and their names to split on.
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# Create an HTMLHeaderTextSplitter object to split the HTML text based on the specified headers.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# Split the HTML string and store the result in the html_header_splits variable.
html_header_splits = html_splitter.split_text(html_string)
# Print the split results.
for header in html_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

<pre class="custom">Header 1
    {'Header 1': 'Header 1'}
    =====================
    Text included in Header 1
    {'Header 1': 'Header 1'}
    =====================
    Header 2-1 Title
    {'Header 1': 'Header 1', 'Header 2': 'Header 2-1 Title'}
    =====================
    Text included in Header 2-1
    {'Header 1': 'Header 1', 'Header 2': 'Header 2-1 Title'}
    =====================
    Header 3-1 Title
    {'Header 1': 'Header 1', 'Header 2': 'Header 2-1 Title', 'Header 3': 'Header 3-1 Title'}
    =====================
    Text included in Header 3-1
    {'Header 1': 'Header 1', 'Header 2': 'Header 2-1 Title', 'Header 3': 'Header 3-1 Title'}
    =====================
    Header 3-2 Title
    {'Header 1': 'Header 1', 'Header 2': 'Header 2-1 Title', 'Header 3': 'Header 3-2 Title'}
    =====================
    Text included in Header 3-2
    {'Header 1': 'Header 1', 'Header 2': 'Header 2-1 Title', 'Header 3': 'Header 3-2 Title'}
    =====================
    Header 2-2 Title
    {'Header 1': 'Header 1', 'Header 2': 'Header 2-2 Title'}
    =====================
    Text included in Header 2-2  
    Last content
    {'Header 1': 'Header 1'}
    =====================
</pre>

## Connecting with Other Splitters and Loading HTML from a Web URL

In this example, we load HTML content from a web URL and then process it by connecting it with other splitters in a pipeline.


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

url = "https://plato.stanford.edu/entries/goedel/"  # Specify the URL of the text to split.

headers_to_split_on = [  # Specify the HTML header tags and their names to split on.
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

# Create an HTMLHeaderTextSplitter object to split the text based on the specified HTML headers.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Fetch the text from the URL and split it based on the HTML headers.
html_header_splits = html_splitter.split_text_from_url(url)

chunk_size = 500  # Specify the size of the chunks to split the text into.
chunk_overlap = 30  # Specify the number of overlapping characters between chunks.
text_splitter = RecursiveCharacterTextSplitter(  # Create a RecursiveCharacterTextSplitter object to recursively split the text.
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split the text that was split by HTML headers into chunks of the specified size.
splits = text_splitter.split_documents(html_header_splits)

# Print chunks 80 to 85 of the split text.
for header in splits[80:85]:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

<pre class="custom">a formula
     Prf( , ) 
     of number theory, representable in  , so that  
    Proof:  
    x  
    y  
    [ ]  
    11  
    P  
    codes a proof of φ ⇒   ⊢
    Prf( ,
     ).  
    n  
    P  
    n  
    ⌈  
    φ  
    ⌉  
    and  
    does not code a proof of φ ⇒  
    ⊢ ¬Prf( ,
     ).  
    n  
    P  
    n  
    ⌈  
    φ  
    ⌉  
    Let Prov( ) denote the formula ∃ 
     Prf( , ) .
     By Theorem 2 there is a sentence φ with the property  
    y  
    x  
    x  
    y  
    [ ]  
    12  
    ⊢ (φ ↔
    ¬Prov( )).  
    P  
    ⌈  
    φ  
    ⌉  
    Thus φ says ‘I am not provable.’ We now observe, if
      ⊢ φ, then by (1) there is   such that
    {'Header 1': 'Kurt Gödel'}
    =====================
    ⊢ Prf( ,
     ), hence  
    ⊢ Prov( ), hence,
    by (3)   ⊢ ¬φ, so   is inconsistent.
    Thus  
    P  
    n  
    P  
    n  
    ⌈  
    φ  
    ⌉  
    P  
    ⌈  
    φ  
    ⌉  
    P  
    P  
    ⊬ φ  
    P  
    Furthermore, by (4) and (2), we have   ⊢
    ¬Prf( ,
     ) for all natural
    numbers  . By ω-consistency   ⊬
    ∃  Prf( ,
     ). Thus (3) gives
      ⊬ ¬φ. We have shown that if   is
    ω-consistent, then φ is independent of  .  
    P  
    n  
    ⌈  
    φ  
    ⌉  
    n  
    P  
    x  
    x  
    ⌈  
    φ  
    ⌉  
    P  
    P  
    P  
    On concluding the proof of the first theorem, Gödel remarks,
    {'Header 1': 'Kurt Gödel'}
    =====================
    “we can readily see that the proof just given is constructive;
    that is … proved in an intuitionistically unobjectionable
    manner…” (Gödel 1986, p. 177). This is because, as
    he points out, all the existential statements are based on his theorem
    V (giving the numeralwise expressibility of primitive recursive
    relations), which is intuitionistically unobjectionable.  
    2.2.3 The Second Incompleteness Theorem  
    The Second Incompleteness Theorem establishes the unprovability, in
    {'Header 1': 'Kurt Gödel'}
    =====================
    number theory, of the consistency of number theory. First we have to
    write down a number-theoretic formula that expresses the consistency
    of the axioms. This is surprisingly simple. We just let
    Con( ) be the sentence ¬Prov( ).  
    P  
    ⌈  
    0 =
    1  
    ⌉  
    (Gödel’s Second Incompleteness
    Theorem) If   is consistent, then Con( ) is not
    provable from  .  
    Theorem 4  
    P  
    P  
    P  
    Let φ be as in (3). The reasoning used to infer
    ‘if   ⊢ φ, then   ⊢ 0 ≠
    1‘ does not go beyond elementary number theory, and can
    {'Header 1': 'Kurt Gödel'}
    =====================
    therefore, albeit with a lot of effort (see below), be formalized in
     . This yields:   ⊢
    (Prov( ) →
    ¬Con( )), and thus by (3),   ⊢
    (Con( ) → φ). Since   ⊬ φ, we
    must have   ⊬ Con( ).  
    Proof:  
    P  
    P  
    P  
    P  
    ⌈  
    φ  
    ⌉  
    P  
    P  
    P  
    P  
    P  
    P  
    The above proof (sketch) of the Second Incompleteness Theorem is
    deceptively simple as it avoids the formalization. A rigorous proof
    would have to establish the proof of ‘if   ⊢
    φ, then   ⊢ 0 ≠ 1’ in  .  
    P  
    P  
    P
    {'Header 1': 'Kurt Gödel'}
    =====================
</pre>

## Limitations

HTMLHeaderTextSplitter attempts to handle structural differences between HTML documents, but it may sometimes miss specific headers.

For example, this algorithm assumes that headers are always nodes "above" the related text, i.e., in previous sibling nodes, ancestor nodes, and combinations thereof.

In the following news article (as of the time of writing), the text of the top headline is tagged as "h1", but it is in a **separate subtree** from the text element we expect.

Therefore, the text related to the "h1" element does not appear in the chunk metadata, but the text related to "h2" does, if applicable.


```python
# Specify the URL of the HTML page to split.
url = "https://www.cnn.com/2023/09/25/weather/el-nino-winter-us-climate/index.html"

headers_to_split_on = [
    ("h1", "Header 1"),  # Specify the header tags and their names to split on.
    ("h2", "Header 2"),  # Specify the header tags and their names to split on.
]

# Create an HTMLHeaderTextSplitter object to split the HTML text based on the specified headers.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Split the HTML page from the specified URL and store the result in the html_header_splits variable.
html_header_splits = html_splitter.split_text_from_url(url)

# Print the split results.
for header in html_header_splits:
    print(f"{header.page_content[:100]}")
    print(f"{header.metadata}", end="\n=====================\n")
```

<pre class="custom">CNN values your feedback  
    1. How relevant is this ad to you?  
    2. Did you encounter any technical i
    {}
    =====================
    An El Niño winter is coming. Here’s what that could mean for the US
    {'Header 1': 'An El Niño winter is coming. Here’s what that could mean for the US'}
    =====================
    By  , CNN Meteorologist  
    Mary Gilbert  
    3 minute read  
    Published
            4:44 AM EDT, Mon Septembe
    {'Header 1': 'An El Niño winter is coming. Here’s what that could mean for the US'}
    =====================
    What could this winter look like?
    {'Header 1': 'An El Niño winter is coming. Here’s what that could mean for the US', 'Header 2': 'What could this winter look like?'}
    =====================
    No two El Niño winters are the same, but many have temperature and precipitation trends in common.  
    {}
    =====================
</pre>
