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

# UpstageDocumentParseLoader 
- Author: [Taylor(Jihyun Kim)](https://github.com/Taylor0819)
- Peer Review : [JoonHo Kim](https://github.com/jhboyo), [Jaemin Hong](https://github.com/geminii01), [leebeanbin](https://github.com/leebeanbin), [Dooil Kwak](https://github.com/back2zion)
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/11-UpstageDocumentParseLoader.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/11-UpstageDocumentParseLoader.ipynb)

## Overview 

The ```UpstageDocumentParseLoader``` is a robust document analysis tool designed by Upstage that seamlessly integrates with the LangChain framework as a document loader. It specializes in transforming documents into structured HTML by analyzing their layout and content.

**Key Features** :

-	Comprehensive Layout Analysis : 
	Analyzes and identifies structural elements like headings, paragraphs, tables, and images across various document formats (e.g., PDFs, images).

-	Automated Structural Recognition : 
	Automatically detects and serializes document elements based on reading order for accurate conversion to HTML.

-	Optional OCR Support : 
	Includes optical character recognition for handling scanned or image-based documents. The OCR mode supports:
	
	```force``` : Extracts text from images using OCR.
	
	```auto``` : Extracts text from PDFs (throws an error if the input is not in PDF format).

By recognizing and preserving the relationships between document elements, the ```UpstageDocumentParseLoader``` enables precise and context-aware document analysis.

**Migration from Layout Analysis** :
Upstage has launched Document Parse to replace Layout Analysis! Document Parse now supports a wider range of document types, markdown output, chart detection, equation recognition, and additional features planned for upcoming releases. The last version of Layout Analysis, layout-analysis-0.4.0, will be officially discontinued by November 10, 2024.

### Table of Contents 

- [Overview](#overview)
- [Key Changes from Layout Analysis](#key-changes-from-layout-analysis)
- [Environment Setup](#environment-setup)
- [UpstageDocumentParseLoader Key Parameters](#upstagedocumentparseloader-key-parameters)
- [Usage Example](#usage-example)

### Key Changes from Layout Analysis

**Changes to Existing Options** :
1. ```use_ocr``` → ```ocr``` 
   
   ```use_ocr``` option has been replaced with ```ocr``` . Instead of ```True/False``` , it now accepts ```force``` or ```auto``` for more precise control.

2. ```output_type``` → ```output_format``` 
   
   ```output_type``` option has been renamed to ```output_format``` for specifying the format of the output.

3. ```exclude``` → ```base64_encoding```

    The ```exclude``` option has been replaced with ```base64_encoding``` . While ```exclude``` was used to exclude specific elements from the output, ```base64_encoding``` specifies whether to encode elements of certain categories in Base64.
   

### References
- [UpstageDocumentParseLoader](https://python.langchain.com/api_reference/upstage/document_parse/langchain_upstage.document_parse.UpstageDocumentParseLoader.html)
- [UpstageLayoutAnalysisLoader](https://python.langchain.com/api_reference/upstage/layout_analysis/langchain_upstage.layout_analysis.UpstageLayoutAnalysisLoader.html)
- [Upstage Migrate to Document Parse from Layout Analysis](https://console.upstage.ai/docs/capabilities/document-parse/migration-dp)

----

## Environment Setup
Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]** 

- ```langchain-opentutorial``` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials.
- You can checkout the [```langchain-opentutorial```](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.


### API Key Configuration
To use ```UpstageDocumentParseLoader``` , you need to [obtain a Upstage API key](https://console.upstage.ai/api-keys).

Once you have your API key, set it as the value for the variable ```UPSTAGE_API_KEY``` .


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
        "langchain_upstage",
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
        "UPSTAGE_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "12-UpstageDocumentParseLoader",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```UPSTAGE_API_KEY``` in .env file and load it.

[Note] This is not necessary if you've already set ```UPSTAGE_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



```python
import os
import nest_asyncio

# Allow async
nest_asyncio.apply()
```

## UpstageDocumentParseLoader Key Parameters

- ```file_path``` : Path(s) to the document(s) to be analyzed
- ```split``` : Document splitting mode [default: 'none', 'element', 'page']
- ```model``` : Model name for document parsing [default: 'document-parse']
- ```ocr``` : OCR mode ["force" (always OCR), "auto" (PDF-only)]
- ```output_format``` : Format of the analysis results [default: 'html', 'text', 'markdown']
- ```coordinates``` : Include OCR coordinates in the output [default: True]
- ```base64_encoding``` : List of element categories to be base64-encoded ['paragraph', 'table', 'figure', 'header', 'footer', 'list', 'chart', '...']

## Usage Example
Let's try running a code example here using ```UpstageDocumentParseLoader``` .

### Data Preparation

In this tutorial, we will use the following pdf file:

- Download Link: [Modular-RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/abs/2407.21059)
- File name: "2407.21059.pdf"
- File path: "./data/2407.21059.pdf"
 
After downloading the PDF file from the provided link, create a data folder in the current directory and save the PDF file into that folder.


```python
# Download and save sample PDF file to ./data directory
import requests


def download_pdf(url, save_path):
    """
    Downloads a PDF file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the PDF file to download.
        save_path (str): The full path (including file name) where the file will be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Save the file to the specified path
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"PDF downloaded and saved to: {save_path}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")


# Configuration for the PDF file
pdf_url = "https://arxiv.org/pdf/2407.21059"
file_path = "./data/2407.21059.pdf"

# Download the PDF
download_pdf(pdf_url, file_path)
```

<pre class="custom">PDF downloaded and saved to: ./data/2407.21059.pdf
</pre>

```python
# Set file path
FILE_PATH = "data/2407.21059.pdf"  # modify to your file path
```

```python
from langchain_upstage import UpstageDocumentParseLoader

# Configure the document loader
loader = UpstageDocumentParseLoader(
    FILE_PATH,
    output_format="html",
    split="page",
    ocr="auto",
    coordinates=True,
    base64_encoding=["chart"],
)

# Load the document
docs = loader.load()

# Print the results
for doc in docs[:2]:
    print(doc)
```

<pre class="custom">page_content='<p id='0' data-category='paragraph' style='font-size:14px'>1</p> <h1 id='1' style='font-size:20px'>Modular RAG: Transforming RAG Systems into<br>LEGO-like Reconfigurable Frameworks</h1> <br><p id='2' data-category='paragraph' style='font-size:18px'>Yunfan Gao, Yun Xiong, Meng Wang, Haofen Wang</p> <p id='3' data-category='paragraph' style='font-size:16px'>Abstract—Retrieval-augmented Generation (RAG) has<br>markedly enhanced the capabilities of Large Language Models<br>(LLMs) in tackling knowledge-intensive tasks. The increasing<br>demands of application scenarios have driven the evolution<br>of RAG, leading to the integration of advanced retrievers,<br>LLMs and other complementary technologies, which in turn<br>has amplified the intricacy of RAG systems. However, the rapid<br>advancements are outpacing the foundational RAG paradigm,<br>with many methods struggling to be unified under the process<br>of “retrieve-then-generate”. In this context, this paper examines<br>the limitations of the existing RAG paradigm and introduces<br>the modular RAG framework. By decomposing complex RAG<br>systems into independent modules and specialized operators, it<br>facilitates a highly reconfigurable framework. Modular RAG<br>transcends the traditional linear architecture, embracing a<br>more advanced design that integrates routing, scheduling, and<br>fusion mechanisms. Drawing on extensive research, this paper<br>further identifies prevalent RAG patterns—linear, conditional,<br>branching, and looping—and offers a comprehensive analysis<br>of their respective implementation nuances. Modular RAG<br>presents innovative opportunities for the conceptualization<br>and deployment of RAG systems. Finally, the paper explores<br>the potential emergence of new operators and paradigms,<br>establishing a solid theoretical foundation and a practical<br>roadmap for the continued evolution and practical deployment<br>of RAG technologies.</p> <br><p id='4' data-category='paragraph' style='font-size:16px'>Index Terms—Retrieval-augmented generation, large language<br>model, modular system, information retrieval</p> <p id='5' data-category='paragraph' style='font-size:18px'>I. INTRODUCTION</p> <br><header id='6' style='font-size:22px'>2024<br>Jul<br>26<br>[cs.CL]<br>arXiv:2407.21059v1</header> <br><p id='7' data-category='paragraph' style='font-size:18px'>L remarkable capabilities, yet they still face numerous<br>ARGE Language Models (LLMs) have demonstrated<br>challenges, such as hallucination and the lag in information up-<br>dates [1]. Retrieval-augmented Generation (RAG), by access-<br>ing external knowledge bases, provides LLMs with important<br>contextual information, significantly enhancing their perfor-<br>mance on knowledge-intensive tasks [2]. Currently, RAG, as<br>an enhancement method, has been widely applied in various<br>practical application scenarios, including knowledge question<br>answering, recommendation systems, customer service, and<br>personal assistants. [3]–[6]</p> <br><p id='8' data-category='paragraph' style='font-size:18px'>During the nascent stages of RAG , its core framework is<br>constituted by indexing, retrieval, and generation, a paradigm<br>referred to as Naive RAG [7]. However, as the complexity<br>of tasks and the demands of applications have escalated, the</p> <p id='9' data-category='footnote' style='font-size:14px'>Yunfan Gao is with Shanghai Research Institute for Intelligent Autonomous<br>Systems, Tongji University, Shanghai, 201210, China.<br>Yun Xiong is with Shanghai Key Laboratory of Data Science, School of<br>Computer Science, Fudan University, Shanghai, 200438, China.<br>Meng Wang and Haofen Wang are with College of Design and Innovation,<br>Tongji University, Shanghai, 20092, China. (Corresponding author: Haofen<br>Wang. E-mail: carter.whfcarter@gmail.com)</p> <br><p id='10' data-category='paragraph' style='font-size:18px'>limitations of Naive RAG have become increasingly apparent.<br>As depicted in Figure 1, it predominantly hinges on the<br>straightforward similarity of chunks, result in poor perfor-<br>mance when confronted with complex queries and chunks with<br>substantial variability. The primary challenges of Naive RAG<br>include: 1) Shallow Understanding of Queries. The semantic<br>similarity between a query and document chunk is not always<br>highly consistent. Relying solely on similarity calculations<br>for retrieval lacks an in-depth exploration of the relationship<br>between the query and the document [8]. 2) Retrieval Re-<br>dundancy and Noise. Feeding all retrieved chunks directly<br>into LLMs is not always beneficial. Research indicates that<br>an excess of redundant and noisy information may interfere<br>with the LLM’s identification of key information, thereby<br>increasing the risk of generating erroneous and hallucinated<br>responses. [9]</p> <br><p id='11' data-category='paragraph' style='font-size:18px'>To overcome the aforementioned limitations, Advanced<br>RAG paradigm focuses on optimizing the retrieval phase,<br>aiming to enhance retrieval efficiency and strengthen the<br>utilization of retrieved chunks. As shown in Figure 1 ,typical<br>strategies involve pre-retrieval processing and post-retrieval<br>processing. For instance, query rewriting is used to make<br>the queries more clear and specific, thereby increasing the<br>accuracy of retrieval [10], and the reranking of retrieval results<br>is employed to enhance the LLM’s ability to identify and<br>utilize key information [11].</p> <br><p id='12' data-category='paragraph' style='font-size:18px'>Despite the improvements in the practicality of Advanced<br>RAG, there remains a gap between its capabilities and real-<br>world application requirements. On one hand, as RAG tech-<br>nology advances, user expectations rise, demands continue to<br>evolve, and application settings become more complex. For<br>instance, the integration of heterogeneous data and the new<br>demands for system transparency, control, and maintainability.<br>On the other hand, the growth in application demands has<br>further propelled the evolution of RAG technology.</p> <br><p id='13' data-category='paragraph' style='font-size:18px'>As shown in Figure 2, to achieve more accurate and efficient<br>task execution, modern RAG systems are progressively inte-<br>grating more sophisticated function, such as organizing more<br>refined index base in the form of knowledge graphs, integrat-<br>ing structured data through query construction methods, and<br>employing fine-tuning techniques to enable encoders to better<br>adapt to domain-specific documents.</p> <br><p id='14' data-category='paragraph' style='font-size:18px'>In terms of process design, the current RAG system has<br>surpassed the traditional linear retrieval-generation paradigm.<br>Researchers use iterative retrieval [12] to obtain richer con-<br>text, recursive retrieval [13] to handle complex queries, and<br>adaptive retrieval [14] to provide overall autonomy and flex-<br>ibility. This flexibility in the process significantly enhances</p>' metadata={'page': 1, 'base64_encodings': [], 'coordinates': [[{'x': 0.9137, 'y': 0.0321}, {'x': 0.9206, 'y': 0.0321}, {'x': 0.9206, 'y': 0.0418}, {'x': 0.9137, 'y': 0.0418}], [{'x': 0.1037, 'y': 0.0715}, {'x': 0.8961, 'y': 0.0715}, {'x': 0.8961, 'y': 0.1385}, {'x': 0.1037, 'y': 0.1385}], [{'x': 0.301, 'y': 0.149}, {'x': 0.6988, 'y': 0.149}, {'x': 0.6988, 'y': 0.1673}, {'x': 0.301, 'y': 0.1673}], [{'x': 0.0785, 'y': 0.2203}, {'x': 0.4943, 'y': 0.2203}, {'x': 0.4943, 'y': 0.5498}, {'x': 0.0785, 'y': 0.5498}], [{'x': 0.0785, 'y': 0.5566}, {'x': 0.4926, 'y': 0.5566}, {'x': 0.4926, 'y': 0.5837}, {'x': 0.0785, 'y': 0.5837}], [{'x': 0.2176, 'y': 0.6044}, {'x': 0.3518, 'y': 0.6044}, {'x': 0.3518, 'y': 0.6205}, {'x': 0.2176, 'y': 0.6205}], [{'x': 0.0254, 'y': 0.2747}, {'x': 0.0612, 'y': 0.2747}, {'x': 0.0612, 'y': 0.7086}, {'x': 0.0254, 'y': 0.7086}], [{'x': 0.0764, 'y': 0.625}, {'x': 0.4947, 'y': 0.625}, {'x': 0.4947, 'y': 0.7904}, {'x': 0.0764, 'y': 0.7904}], [{'x': 0.0774, 'y': 0.7923}, {'x': 0.4942, 'y': 0.7923}, {'x': 0.4942, 'y': 0.8539}, {'x': 0.0774, 'y': 0.8539}], [{'x': 0.0773, 'y': 0.8701}, {'x': 0.4946, 'y': 0.8701}, {'x': 0.4946, 'y': 0.9447}, {'x': 0.0773, 'y': 0.9447}], [{'x': 0.5068, 'y': 0.221}, {'x': 0.9234, 'y': 0.221}, {'x': 0.9234, 'y': 0.4605}, {'x': 0.5068, 'y': 0.4605}], [{'x': 0.5074, 'y': 0.4636}, {'x': 0.9243, 'y': 0.4636}, {'x': 0.9243, 'y': 0.6131}, {'x': 0.5074, 'y': 0.6131}], [{'x': 0.5067, 'y': 0.6145}, {'x': 0.9234, 'y': 0.6145}, {'x': 0.9234, 'y': 0.7483}, {'x': 0.5067, 'y': 0.7483}], [{'x': 0.5071, 'y': 0.7504}, {'x': 0.9236, 'y': 0.7504}, {'x': 0.9236, 'y': 0.8538}, {'x': 0.5071, 'y': 0.8538}], [{'x': 0.5073, 'y': 0.8553}, {'x': 0.9247, 'y': 0.8553}, {'x': 0.9247, 'y': 0.9466}, {'x': 0.5073, 'y': 0.9466}]]}
    page_content='<header id='15' style='font-size:14px'>2</header> <figure id='16'><img style='font-size:16px' alt="Fig. 1. Cases of Naive RAG and Advanced RAG.When faced with complex
    questions, both encounter limitations and struggle to provide satisfactory
    answers. Despite the fact that Advanced RAG improves retrieval accuracy
    through hierarchical indexing, pre-retrieval, and post-retrieval processes, these
    relevant documents have not been used correctly." data-coord="top-left:(99,107); bottom-right:(1168,1179)" /></figure> <br><p id='17' data-category='paragraph' style='font-size:20px'>the expressive power and adaptability of RAG systems, en-<br>abling them to better adapt to various application scenarios.<br>However, this also makes the orchestration and scheduling of<br>workflows more complex, posing greater challenges to system<br>design. Specifically, RAG currently faces the following new<br>challenges:</p> <br><p id='18' data-category='paragraph' style='font-size:20px'>Complex data sources integration. RAG are no longer<br>confined to a single type of unstructured text data source but<br>have expanded to include various data types, such as semi-<br>structured data like tables and structured data like knowledge<br>graphs [15]. Access to heterogeneous data from multiple<br>sources can provide the system with a richer knowledge<br>background, and more reliable knowledge verification capa-<br>bilities [16].</p> <br><caption id='19' style='font-size:16px'>Fig. 2. Case of current Modular RAG.The system integrates diverse data<br>and more functional components. The process is no longer confined to linear<br>but is controlled by multiple control components for retrieval and generation,<br>making the entire system more flexible and complex.</caption> <br><p id='20' data-category='paragraph' style='font-size:20px'>New demands for system interpretability, controllability,</p>' metadata={'page': 2, 'base64_encodings': [], 'coordinates': [[{'x': 0.9113, 'y': 0.0313}, {'x': 0.9227, 'y': 0.0313}, {'x': 0.9227, 'y': 0.0417}, {'x': 0.9113, 'y': 0.0417}], [{'x': 0.0777, 'y': 0.0649}, {'x': 0.9163, 'y': 0.0649}, {'x': 0.9163, 'y': 0.7149}, {'x': 0.0777, 'y': 0.7149}], [{'x': 0.0775, 'y': 0.7174}, {'x': 0.4936, 'y': 0.7174}, {'x': 0.4936, 'y': 0.8062}, {'x': 0.0775, 'y': 0.8062}], [{'x': 0.0762, 'y': 0.8089}, {'x': 0.4923, 'y': 0.8089}, {'x': 0.4923, 'y': 0.9293}, {'x': 0.0762, 'y': 0.9293}], [{'x': 0.5077, 'y': 0.8791}, {'x': 0.9248, 'y': 0.8791}, {'x': 0.9248, 'y': 0.926}, {'x': 0.5077, 'y': 0.926}], [{'x': 0.0911, 'y': 0.9311}, {'x': 0.4948, 'y': 0.9311}, {'x': 0.4948, 'y': 0.9462}, {'x': 0.0911, 'y': 0.9462}]]}
</pre>
