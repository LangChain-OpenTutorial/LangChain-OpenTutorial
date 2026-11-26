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

# LlamaParse

- Author: [syshin0116](https://github.com/syshin0116)
- Peer Review: [JoonHo Kim](https://github.com/jhboyo), [Jaemin Hong](https://github.com/geminii01), [leebeanbin](https://github.com/leebeanbin), [Taylor(Jihyun Kim)](https://github.com/Taylor0819), [Dooil Kwak](https://github.com/back2zion)
- Proofread : [JaeJun Shim](https://github.com/kkam-dragon)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/12-LlamaParse.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/06-DocumentLoader/12-LlamaParse.ipynb)

## Overview

```LlamaParse``` is a document parsing service developed by **LlamaIndex** , specifically designed for large language models (LLMs).

Key Features:

- Support for various document formats, such as PDF, Word, PowerPoint, and Excel
- Customized output formats through natural language instructions
- Advanced table and image extraction capabilities
- Multilingual support
- Multiple output format support

```LlamaParse``` is available as a standalone API and is also integrated into the LlamaCloud platform. This service aims to enhance the performance of LLM-based applications, such as RAG(Retrieval-Augmented Generation), by parsing and refining documents.

Users can process up to 1,000 pages per day for free, with additional capacity available through paid plans. ```LlamaParse``` is currently offered in public beta and is continuously expanding its features.

- Link: [https://cloud.llamaindex.ai](https://cloud.llamaindex.ai)


### Table of Contents
- [Overview](#overview)
- [Environement Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [LlamaParse Parameters](#llamaparse-parameters)
- [Simple Parsing](#simple-parsing)
- [Multomodal Model Parsing](#multimodal-model-parsing)
- [Custom Parsing Instructions](#custom-parsing-instructions)

### References

- [LlamaParse: Using in Python](https://docs.cloud.llamaindex.ai/llamaparse/getting_started/python)
- [LlamaParse: Getting Started](https://docs.cloud.llamaindex.ai/llamaparse/getting_started)

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
        "llama-index-core",
        "llama-parse",
        "llama-index-readers-file",
    ],
    verbose=False,
    upgrade=False,
)
```

<pre class="custom">
    [notice] A new release of pip is available: 24.2 -> 24.3.1
    [notice] To update, run: pip install --upgrade pip
</pre>

### API Key Configuration

To use ```LlamaParse``` , you need to [obtain a Llama Cloud API key](https://docs.cloud.llamaindex.ai/llamaparse/getting_started/get_an_api_key).

```python
# Set environment variables
from langchain_opentutorial import set_env

set_env(
    {
        "OPENAI_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "LlamaParse",
        "LLAMA_CLOUD_API_KEY": "",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as ```OPENAI_API_KEY``` in a ```.env``` file and load them.

**[Note]** This is not necessary if you've already set the required API keys in previous steps.

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

## Data Preparation

In this tutorial, we will use the following pdf file:

- Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- Download Link: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- File name: "1706.03762v7.pdf"
- File path: "data/1706.03762v7.pdf"

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
pdf_url = "https://arxiv.org/pdf/1706.03762v7"
file_path = "./data/1706.03762v7.pdf"

# Download the PDF
download_pdf(pdf_url, file_path)
```

<pre class="custom">PDF downloaded and saved to: ./data/1706.03762v7.pdf
</pre>

```python
# Set file path
FILE_PATH = "./data/1706.03762v7.pdf"  # modify to your file path
```

## LlamaParse Parameters

### Key Parameters

These are the core settings that most users will configure:

| Parameter        | Description                                                                   | Default Value      |
| ---------------- | ----------------------------------------------------------------------------- | ------------------ |
| ```api_key```        | A string representing the API key for authenticating with the **LlamaParse API** | Required           |
| ```base_url```       | The base URL for the **LlamaParse API**                                         | "DEFAULT_BASE_URL" |
| ```check_interval``` | Specifies the time (in seconds) between checks for the parsing job status     | 1                  |
| ```ignore_errors```  | Boolean indicating whether to skip errors during parsing                      | True               |
| ```max_timeout```    | Maximum time (in seconds) to wait for the parsing job to finish               | 2000               |
| ```num_workers```    | Number of parallel workers for API requests (Range: 1-9)                      | 4                  |
| ```result_type```    | Format of the parsing result (e.g., "text", "markdown", "json", "structured") | "text"             |
| ```show_progress```  | Displays progress for multi-file parsing                                      | True               |
| ```split_by_page```  | Splits the output by pages                                                    | True               |
| ```language```       | Specifies the language of the document text                                   | "en"               |
| ```verbose```        | Enables verbose output to show detailed parsing progress                      | True               |


### Advanced Parameters

For specialized use cases, consider these options:

**Parsing Modes and Enhancements**

|Parameter|Description|Default Value|
|---|---|---|
|```auto_mode```|Automatically selects the optimal parsing mode|False|
|```auto_mode_trigger_on_image_in_page```|Upgrades pages with images to premium mode (if ```auto_mode``` is enabled)|False|
|```auto_mode_trigger_on_table_in_page```|Upgrades pages with tables to premium mode (if ```auto_mode``` is enabled)|False|
|```auto_mode_trigger_on_text_in_page```|Upgrades pages with specific text to premium mode (if ```auto_mode``` is enabled)|None|
|```auto_mode_trigger_on_regexp_in_page```|Upgrades pages matching a regex to premium mode (if ```auto_mode``` is enabled)|None|
|```premium_mode```|Uses the most advanced parsing capabilities|False|
|```fast_mode```|Enables faster parsing by skipping OCR and table reconstruction|False|



**Content Extraction**

|Parameter|Description|Default Value|
|---|---|---|
|```disable_ocr```|Disables OCR, extracting only selectable text|False|
|```disable_image_extraction```|Prevents image extraction to speed up the parsing process|False|
|```extract_charts```|Extracts or tags charts in the document|False|
|```extract_layout```|Includes layout information in the parsed output|False|
|```annotate_links```|Annotates links in the document for URL extraction|False|
|```continuous_mode```|Improves parsing quality for documents with multi-page tables|False|
|```guess_xlsx_sheet_names```|Infers sheet names when parsing Excel files|False|



**Output Customization**

| Parameter                            | Description                                        | Default Value |
| ------------------------------------ | -------------------------------------------------- | ------------- |
| ```page_separator```                     | Specifies a custom string to separate parsed pages | None          |
| ```structured_output```                  | Outputs data in structured formats (e.g., JSON)    | False         |
| ```structured_output_json_schema```      | JSON schema for formatting structured output       | None          |
| ```structured_output_json_schema_name``` | Predefined schema name for formatting output       | None          |
| ```parsing_instruction```                | Custom instructions for parsing behavior           | ""            |



**Targeting and Filtering**

| Parameter                                            | Description                                                                                    | Default Value |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------- |
| ```target_pages```                                       | Comma-separated list of page numbers to parse                                                  | None          |
| ```max_pages```                                          | Limits the number of pages to parse                                                            | None          |
| ```bbox_top```, ```bbox_bottom```, ```bbox_left```, ```bbox_right``` | Defines margins for bounding boxes (0–1 range) for extracting specific regions of the document | None          |
| ```skip_diagonal_text```                                 | Ignores text that appears diagonally (non-standard text rotations)                             | False         |



**Integration and Webhooks**

|Parameter|Description|Default Value|
|---|---|---|
|```webhook_url```|URL to be called upon completion of the parsing job|None|
|```output_s3_path_prefix```|S3 path for uploading parsed output|None|
|```custom_client```|Custom HTTPX client for sending requests|None|
|```invalidate_cache```|Ignores cached documents, forcing re-parsing|False|
|```do_not_cache```|Prevents caching of parsed documents|False|


## Simple Parsing

The default usage of ```LlamaParse``` demonstrates how to parse documents using its core functionality. This mode is optimized for simplicity and works well for standard document types.

```python
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Configure the LlamaParse instance
parser = LlamaParse(
    result_type="markdown",  # Output format ("text", "markdown", "json", or "structured")
    num_workers=8,
    verbose=True,
    language="en",
    show_progress=True,
)

# Define a file extractor mapping file extensions to parsers
file_extractor = {".pdf": parser}

# Use SimpleDirectoryReader to parse the specified PDF file
documents = SimpleDirectoryReader(
    input_files=[FILE_PATH],  # List of files to process
    file_extractor=file_extractor,
).load_data()
```

<pre class="custom">Started parsing the file under job_id a1da411e-8d11-468f-98b9-5c846fccc4a0
</pre>

```python
# Display the number of documents parsed
len(documents)
```




<pre class="custom">15</pre>



```python
# Display the first document
documents[0]
```




<pre class="custom">Document(id_='801f0227-e66c-424b-b82b-4fefc746ec96', embedding=None, metadata={'file_path': 'data/1706.03762v7.pdf', 'file_name': '1706.03762v7.pdf', 'file_type': 'application/pdf', 'file_size': 2215244, 'creation_date': '2025-01-08', 'last_modified_date': '2025-01-08'}, excluded_embed_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], excluded_llm_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text_resource=MediaResource(embeddings=None, data=None, text='Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.\n\n# Attention Is All You Need\n\narXiv:1706.03762v7  [cs.CL]  2 Aug 2023\n\nAshish Vaswani∗                   Noam Shazeer∗                   Niki Parmar∗               Jakob Uszkoreit∗\n\nGoogle Brain\n\nGoogle Brain\n\nGoogle Research\n\nGoogle Research\n\navaswani@google.com                 noam@google.com               nikip@google.com                usz@google.com\n\nLlion Jones∗                     Aidan N. Gomez∗ †                              Łukasz Kaiser∗\n\nGoogle Research\n\nUniversity of Toronto\n\nGoogle Brain\n\nllion@google.com                   aidan@cs.toronto.edu                     lukaszkaiser@google.com\n\nIllia Polosukhin∗ ‡\n\nillia.polosukhin@gmail.com\n\n# Abstract\n\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.\n\n∗Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.\n\n†Work performed while at Google Brain.\n\n‡Work performed while at Google Research.\n\n31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.', path=None, url=None, mimetype=None), image_resource=None, audio_resource=None, video_resource=None, text_template='{metadata_str}\n\n{content}')</pre>



### Conversion to LangChain Documents

The parsed documents are converted to the ```LangChain``` document format for further processing.

```python
# Convert LlamaIndex documents to LangChain document format
docs = [doc.to_langchain_format() for doc in documents]
```

```python
# Display the content of a specific document (e.g., the 6th document)
print(docs[5].page_content)
```

<pre class="custom"># Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.
    
    |Layer Type|Complexity per Layer|Sequential Operations|Maximum Path Length|
    |---|---|---|---|
    |Self-Attention|O(n2 · d)|O(1)|O(1)|
    |Recurrent|O(k · n · d)|O(n)|O(logk(n))|
    |Convolutional|O(r · n · d)|O(1)|O(n/r)|
    
    # 3.5 Positional Encoding
    
    Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].
    
    In this work, we use sine and cosine functions of different frequencies:
    
    PE(pos, 2i) = sin(pos/100002i/dmodel)
    
    PE(pos, 2i+1) = cos(pos/100002i/dmodel)
    
    where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE(pos+k) can be represented as a linear function of PE(pos).
    
    We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.
    
    # 4 Why Self-Attention
    
    In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈ Rd, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.
    
    One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required. The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.
    
    As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence
</pre>

```python
# Display metadata of the first document
docs[0].metadata
```




<pre class="custom">{'file_path': 'data/1706.03762v7.pdf',
     'file_name': '1706.03762v7.pdf',
     'file_type': 'application/pdf',
     'file_size': 2215244,
     'creation_date': '2025-01-08',
     'last_modified_date': '2025-01-08'}</pre>



## MultiModal Model Parsing

**Multimodal parsing** in ```LlamaParse``` uses external AI models to process documents with complex content. Instead of extracting text directly, it processes screenshots of each page and generates a structured output based on visual interpretation. This method is particularly effective for non-standard layouts, scanned documents, or documents with embedded media.

### Key Features:

- Visual Processing: Operates on page screenshots, not raw text, to interpret document content.
- Advanced Models: Integrates with AI models like ```OpenAI``` 's GPT-4o and others for enhanced document analysis.
- Customizable: Supports various models and optional API key usage for flexibility.

### Procedure:

1. Screenshot Generation: A screenshot is taken for each page of the document.
2. Model Processing: The page screenshots are sent to the selected multimodal model with instructions to process them visually.
3. Result Compilation: The model outputs the page content (e.g., as Markdown), which is then consolidated into the final result.

### Key Parameters

|Parameter|Description|Example Value|
|---|---|---|
|```use_vendor_multimodal_model```|Specifies whether to use an external vendor's multimodal model. Setting this to True enables multimodal parsing.|True|
|```vendor_multimodal_model_name```|Specifies the name of the multimodal model to use. In this case, "openai-gpt4o" is selected.|"openai-gpt4o"|
|```vendor_multimodal_api_key```|Sets the API key for the multimodal model. The **OpenAI API key** is retrieved from an environment variable.|"OPENAI_API_KEY"|
|```result_type```|Specifies the format of the parsing result. Here, it is set to "markdown", so the results are returned in Markdown format.|"markdown"|
|```language```|Specifies the language of the document to be parsed. |"en"|
|```skip_diagonal_text```|Determines whether to skip diagonal text during parsing.|True|
|```page_separator```|Specifies a custom page separator.|None|

```python
# Configure the LlamaParse instance to use the vendor multimodal model
multimodal_parser = LlamaParse(
    use_vendor_multimodal_model=True,
    # vendor_multimodal_model_name="openai-gpt4o", # uncomment to specify a model
    # vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"], # uncomment and use your OpenAI API key(uses less llama cloud credits)
    result_type="markdown",
    language="en",
)
```

```python
# Parse the PDF file using the multimodal parser
parsed_docs = multimodal_parser.load_data(file_path=FILE_PATH)
```

<pre class="custom">Started parsing the file under job_id 14e6f376-d585-400a-b3ad-235c9d8070db
</pre>

```python
# Convert to langchain document format
docs = [doc.to_langchain_format() for doc in parsed_docs]
docs
```




<pre class="custom">[Document(id='7328336e-649e-4089-b304-99930e2491b7', metadata={}, page_content='# Attention Is All You Need\n\nAshish Vaswani\\*  \nGoogle Brain  \navaswani@google.com  \n\nNoam Shazeer\\*  \nGoogle Brain  \nnoam@google.com  \n\nNiki Parmar\\*  \nGoogle Research  \nnikip@google.com  \n\nJakob Uszkoreit\\*  \nGoogle Research  \nusz@google.com  \n\nLlion Jones\\*  \nGoogle Research  \nllion@google.com  \n\nAidan N. Gomez\\* †  \nUniversity of Toronto  \naidan@cs.toronto.edu  \n\nŁukasz Kaiser\\*  \nGoogle Brain  \nlukaszkaiser@google.com  \n\nIlia Polosukhin\\* ‡  \nillia.polosukhin@gmail.com  \n\n## Abstract\n\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.\n\n----\n\n\\*Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.\n\n†Work performed while at Google Brain.  \n‡Work performed while at Google Research.\n\n31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.'),
     Document(id='644b622e-05e8-43c1-8210-3d970eabc5ec', metadata={}, page_content='# 1 Introduction\n\nRecurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].\n\nRecurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states \\( h_t \\), as a function of the previous hidden state \\( h_{t-1} \\) and the input for position \\( t \\). This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.\n\nAttention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.\n\nIn this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.\n\n# 2 Background\n\nThe goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.\n\nSelf-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].\n\nEnd-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].\n\nTo the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].\n\n# 3 Model Architecture\n\nMost competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations \\( (x_1, \\ldots, x_n) \\) to a sequence of continuous representations \\( z = (z_1, \\ldots, z_n) \\). Given \\( z \\), the decoder then generates an output sequence \\( (y_1, \\ldots, y_m) \\) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.'),
     Document(id='5a78c15c-e71d-4015-b7a4-59452a7c3c43', metadata={}, page_content='# The Transformer Model Architecture\n\nThe Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.\n\n## 3.1 Encoder and Decoder Stacks\n\n**Encoder:**  \nThe encoder is composed of a stack of \\( N = 6 \\) identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [1] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is \\( \\text{LayerNorm}(x + \\text{Sublayer}(x)) \\), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension \\( d_{\\text{model}} = 512 \\).\n\n**Decoder:**  \nThe decoder is also composed of a stack of \\( N = 6 \\) identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position \\( i \\) can depend only on the known outputs at positions less than \\( i \\).\n\n## 3.2 Attention\n\nAn attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values.\n\n!Figure 1: The Transformer - model architecture.'),
     Document(id='671374b8-3c8b-40b5-b5a5-396027bc727a', metadata={}, page_content='!Scaled Dot-Product Attention and Multi-Head Attention\n\nFigure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.\n\nof the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.\n\n### 3.2.1 Scaled Dot-Product Attention\n\nWe call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension \\(d_k\\), and values of dimension \\(d_v\\). We compute the dot products of the query with all keys, divide each by \\(\\sqrt{d_k}\\), and apply a softmax function to obtain the weights on the values.\n\nIn practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix \\(Q\\). The keys and values are also packed together into matrices \\(K\\) and \\(V\\). We compute the matrix of outputs as:\n\n\\[\n\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V\n\\]\n\nThe two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of \\(\\frac{1}{\\sqrt{d_k}}\\). Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.\n\nWhile for small values of \\(d_k\\) the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of \\(d_k\\) [3]. We suspect that for large values of \\(d_k\\), the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by \\(\\frac{1}{\\sqrt{d_k}}\\).\n\n### 3.2.2 Multi-Head Attention\n\nInstead of performing a single attention function with \\(d_{\\text{model}}\\)-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values \\(h\\) times with different, learned linear projections to \\(d_k\\), \\(d_k\\) and \\(d_v\\) dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding \\(d_v\\)-dimensional\n\n----\n\nTo illustrate why the dot products get large, assume that the components of \\(q\\) and \\(k\\) are independent random variables with mean 0 and variance 1. Then their dot product, \\(q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i\\), has mean 0 and variance \\(d_k\\).'),
     Document(id='cc2a128e-75de-45b4-97ba-ddca253816fb', metadata={}, page_content='output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.\n\nMulti-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.\n\n\\[ \\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O \\]\n\nwhere \\(\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\\)\n\nWhere the projections are parameter matrices \\(W_i^Q \\in \\mathbb{R}^{d_{model} \\times d_k}, W_i^K \\in \\mathbb{R}^{d_{model} \\times d_k}, W_i^V \\in \\mathbb{R}^{d_{model} \\times d_v},\\) and \\(W^O \\in \\mathbb{R}^{hd_v \\times d_{model}}\\).\n\nIn this work we employ \\(h = 8\\) parallel attention layers, or heads. For each of these we use \\(d_k = d_v = d_{model}/h = 64\\). Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.\n\n### 3.2.3 Applications of Attention in our Model\n\nThe Transformer uses multi-head attention in three different ways:\n\n- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].\n\n- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.\n\n- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to \\(-\\infty\\)) all values in the input of the softmax which correspond to illegal connections. See Figure 2.\n\n### 3.3 Position-wise Feed-Forward Networks\n\nIn addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.\n\n\\[ \\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2 \\]\n\nWhile the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is \\(d_{model} = 512\\), and the inner-layer has dimensionality \\(d_{ff} = 2048\\).\n\n### 3.4 Embeddings and Softmax\n\nSimilarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension \\(d_{model}\\). We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by \\(\\sqrt{d_{model}}\\).'),
     Document(id='2910e9e7-c58c-4555-83ac-411a71a42f49', metadata={}, page_content='Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. \\( n \\) is the sequence length, \\( d \\) is the representation dimension, \\( k \\) is the kernel size of convolutions and \\( r \\) the size of the neighborhood in restricted self-attention.\n\n| Layer Type              | Complexity per Layer | Sequential Operations | Maximum Path Length |\n|-------------------------|----------------------|-----------------------|---------------------|\n| Self-Attention          | \\( O(n^2 \\cdot d) \\) | \\( O(1) \\)            | \\( O(1) \\)          |\n| Recurrent               | \\( O(n \\cdot d^2) \\) | \\( O(n) \\)            | \\( O(n) \\)          |\n| Convolutional           | \\( O(k \\cdot n \\cdot d^2) \\) | \\( O(1) \\)    | \\( O(\\log_k(n)) \\)  |\n| Self-Attention (restricted) | \\( O(r \\cdot n \\cdot d) \\) | \\( O(1) \\) | \\( O(n/r) \\)        |\n\n## 3.5 Positional Encoding\n\nSince our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension \\( d_{\\text{model}} \\) as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].\n\nIn this work, we use sine and cosine functions of different frequencies:\n\n\\[\nPE_{\\text{(pos, 2i)}} = \\sin(\\text{pos}/10000^{2i/d_{\\text{model}}})\n\\]\n\n\\[\nPE_{\\text{(pos, 2i+1)}} = \\cos(\\text{pos}/10000^{2i/d_{\\text{model}}})\n\\]\n\nwhere pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from \\( 2\\pi \\) to \\( 10000 \\cdot 2\\pi \\). We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset \\( k \\), \\( PE_{\\text{pos}+k} \\) can be represented as a linear function of \\( PE_{\\text{pos}} \\).\n\nWe also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.\n\n## 4 Why Self-Attention\n\nIn this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations \\((x_1, ..., x_n)\\) to another sequence of equal length \\((z_1, ..., z_n)\\), with \\( x_i, z_i \\in \\mathbb{R}^d \\), such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.\n\nOne is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.\n\nThe third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.\n\nAs noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires \\( O(n) \\) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence'),
     Document(id='e0584970-d549-4882-99aa-09d9d3f37aaa', metadata={}, page_content='# 5 Training\n\nThis section describes the training regime for our models.\n\n## 5.1 Training Data and Batching\n\nWe trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.\n\n## 5.2 Hardware and Schedule\n\nWe trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models (described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).\n\n## 5.3 Optimizer\n\nWe used the Adam optimizer [20] with \\(\\beta_1 = 0.9\\), \\(\\beta_2 = 0.98\\) and \\(\\epsilon = 10^{-9}\\). We varied the learning rate over the course of training, according to the formula:\n\n\\[\nlrate = d_{\\text{model}}^{-0.5} \\cdot \\min(\\text{step\\_num}^{-0.5}, \\text{step\\_num} \\cdot \\text{warmup\\_steps}^{-1.5})\n\\]\n\nThis corresponds to increasing the learning rate linearly for the first \\(\\text{warmup\\_steps}\\) training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used \\(\\text{warmup\\_steps} = 4000\\).\n\n## 5.4 Regularization\n\nWe employ three types of regularization during training:'),
     Document(id='f42704dd-b47c-41d5-b670-4bd1919e1a78', metadata={}, page_content='Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.\n\n| Model                          | BLEU       | Training Cost (FLOPs) |\n|-------------------------------|------------|-----------------------|\n|                               | EN-DE | EN-FR | EN-DE | EN-FR |\n| ByteNet [18]                  | 23.75      | 39.2       | 1.0 · 10^20  |\n| Deep-Att + PosUnk [39]        |            |            |               |\n| GNMT + RL [38]                | 24.6       | 39.92      | 2.3 · 10^19   | 1.4 · 10^20  |\n| ConvS2S [9]                   | 25.16      | 40.46      | 9.6 · 10^18   | 1.5 · 10^20  |\n| MoE [32]                      | 26.03      | 40.56      | 2.0 · 10^19   | 1.2 · 10^20  |\n| Deep-Att + PosUnk Ensemble [39]|            | 40.4       |               | 8.0 · 10^20  |\n| GNMT + RL Ensemble [38]       | 26.30      | 41.16      | 1.8 · 10^20   | 1.1 · 10^21  |\n| ConvS2S Ensemble [9]          | 26.36      | 41.29      | 7.7 · 10^19   | 1.2 · 10^21  |\n| Transformer (base model)      | 27.3       | 38.1       | 3.3 · 10^18   |               |\n| Transformer (big)             | 28.4       | 41.8       | 2.3 · 10^19   |               |\n\n**Residual Dropout** We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of \\( P_{drop} = 0.1 \\).\n\n**Label Smoothing** During training, we employed label smoothing of value \\( \\epsilon_{ls} = 0.1 \\) [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.\n\n## 6 Results\n\n### 6.1 Machine Translation\n\nOn the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.\n\nOn the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate \\( P_{drop} = 0.1 \\), instead of 0.3.\n\nFor the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].\n\nTable 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU ⁵.\n\n### 6.2 Model Variations\n\nTo evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the\n\n⁵We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.'),
     Document(id='f235b946-168e-4ada-bbcb-be9a7b3ef33e', metadata={}, page_content='Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.\n\n|     | \\(N\\) | \\(d_{\\text{model}}\\) | \\(d_{\\text{ff}}\\) | \\(h\\) | \\(d_k\\) | \\(d_v\\) | \\(P_{\\text{drop}}\\) | \\(\\epsilon_{ls}\\) | train steps | PPL (dev) | BLEU (dev) | params \\(\\times 10^6\\) |\n|-----|-------|----------------------|-------------------|-------|---------|---------|---------------------|-------------------|-------------|-----------|------------|-----------------------|\n| base| 6     | 512                  | 2048              | 8     | 64      | 64      | 0.1                 | 0.1               | 100K        | 4.92      | 25.8       | 65                    |\n| (A) |       | 1                    | 512               | 512   |         |         |                     |                   |             | 5.29      | 24.9       |                       |\n|     |       | 4                    | 128               | 128   |         |         |                     |                   |             | 5.00      | 25.5       |                       |\n|     |       | 16                   | 32                | 32    |         |         |                     |                   |             | 4.91      | 25.8       |                       |\n|     |       | 32                   | 16                | 16    |         |         |                     |                   |             | 5.01      | 25.4       |                       |\n| (B) |       |                      | 16                |       |         |         |                     |                   |             | 5.16      | 25.1       | 58                    |\n|     |       |                      | 32                |       |         |         |                     |                   |             | 5.01      | 25.4       | 60                    |\n| (C) | 2     |                      |                   |       |         |         |                     |                   |             | 6.11      | 23.7       | 36                    |\n|     | 4     |                      |                   |       |         |         |                     |                   |             | 5.19      | 25.3       | 50                    |\n|     | 8     |                      |                   |       |         |         |                     |                   |             | 4.88      | 25.5       | 80                    |\n|     |       | 256                  | 32                | 32    |         |         |                     |                   |             | 5.75      | 24.5       | 28                    |\n|     |       | 1024                 | 128               | 128   |         |         |                     |                   |             | 4.66      | 26.0       | 168                   |\n|     |       | 1024                 |                   |       |         |         |                     |                   |             | 5.12      | 25.4       | 53                    |\n|     |       | 4096                 |                   |       |         |         |                     |                   |             | 4.75      | 26.2       | 90                    |\n| (D) |       |                      |                   |       |         |         | 0.0                 |                   |             | 5.77      | 24.6       |                       |\n|     |       |                      |                   |       |         |         | 0.2                 |                   |             | 4.95      | 25.5       |                       |\n|     |       |                      |                   |       |         |         | 0.0                 |                   |             | 4.67      | 25.3       |                       |\n|     |       |                      |                   |       |         |         | 0.2                 |                   |             | 5.47      | 25.7       |                       |\n| (E) |       |                      |                   |       |         |         | positional embedding instead of sinusoids | | | 4.92 | 25.7 | |\n| big | 6     | 1024                 | 4096              | 16    | 0.3     | 0.3     | 0.3                 | 0.3               | 300K        | 4.33      | 26.4       | 213                   |\n\ndevelopment set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.\n\nIn Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.\n\nIn Table 3 rows (B), we observe that reducing the attention key size \\(d_k\\) hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, especially, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.\n\n### 6.3 English Constituency Parsing\n\nTo evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].\n\nWe trained a 4-layer transformer with \\(d_{\\text{model}} = 1024\\) on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.\n\nWe performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we'),
     Document(id='b373249e-1769-411d-aacd-0b456d8bd240', metadata={}, page_content='## Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)\n\n| Parser                              | Training                | WSJ 23 F1 |\n|-------------------------------------|-------------------------|-----------|\n| Vinyals & Kaiser et al. (2014) [37] | WSJ only, discriminative| 88.3      |\n| Petrov et al. (2006) [29]           | WSJ only, discriminative| 90.4      |\n| Zhu et al. (2013) [40]              | WSJ only, discriminative| 90.4      |\n| Dyer et al. (2016) [8]              | WSJ only, discriminative| 91.7      |\n| Transformer (4 layers)              | WSJ only, discriminative| 91.3      |\n| Zhu et al. (2013) [40]              | semi-supervised         | 91.3      |\n| Huang & Harper (2009) [14]          | semi-supervised         | 91.3      |\n| McClosky et al. (2006) [26]         | semi-supervised         | 92.1      |\n| Vinyals & Kaiser et al. (2014) [37] | semi-supervised         | 92.1      |\n| Transformer (4 layers)              | semi-supervised         | 92.7      |\n| Luong et al. (2015) [23]            | multi-task              | 93.0      |\n| Dyer et al. (2016) [8]              | generative              | 93.3      |\n\nIncreased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3 for both WSJ only and the semi-supervised setting.\n\nOur results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].\n\nIn contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the Berkeley-Parser [29] even when training only on the WSJ training set of 40K sentences.\n\n## 7 Conclusion\n\nIn this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.\n\nFor translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.\n\nWe are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.\n\nThe code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.\n\n## Acknowledgements\n\nWe are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.\n\n## References\n\n[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.\n\n[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *CoRR, abs/1409.0473*, 2014.\n\n[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. *CoRR, abs/1703.03906*, 2017.\n\n[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. *arXiv preprint arXiv:1601.06733*, 2016.'),
     Document(id='5238b57c-c7af-403d-a8c6-0ffe3b04c1d7', metadata={}, page_content='```\n[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, \nand Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical \nmachine translation. CoRR, abs/1406.1078, 2014.\n\n[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv \npreprint arXiv:1610.02357, 2016.\n\n[7] Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation \nof gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.\n\n[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural \nnetwork grammars. In Proc. of NAACL, 2016.\n\n[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolu- \ntional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.\n\n[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint \narXiv:1308.0850, 2013.\n\n[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im- \nage recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern \nRecognition, pages 770–778, 2016.\n\n[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in \nrecurrent nets: the difficulty of learning long-term dependencies, 2001.\n\n[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, \n9(8):1735–1780, 1997.\n\n[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations \nacross languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural \nLanguage Processing, pages 832–841, ACL, August 2009.\n\n[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring \nthe limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.\n\n[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural \nInformation Processing Systems, (NIPS), 2016.\n\n[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference \non Learning Representations (ICLR), 2016.\n\n[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko- \nray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, \n2017.\n\n[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. \nIn International Conference on Learning Representations, 2017.\n\n[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.\n\n[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint \narXiv:1703.10722, 2017.\n\n[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen \nZhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint \narXiv:1703.03130, 2017.\n\n[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task \nsequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.\n\n[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention- \nbased neural machine translation. arXiv preprint arXiv:1508.04025, 2015.\n```'),
     Document(id='4b2f4906-9dab-404e-a21a-c6e5acb6562f', metadata={}, page_content='```\n[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated \ncorpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.\n\n[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In \nProceedings of the Human Language Technology Conference of the NAACL, Main Conference, \npages 152–159. ACL, June 2006.\n\n[27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention \nmodel. In Empirical Methods in Natural Language Processing, 2016.\n\n[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive \nsummarization. arXiv preprint arXiv:1705.04304, 2017.\n\n[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, \nand interpretable tree annotation. In Proceedings of the 21st International Conference on \nComputational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July \n2006.\n\n[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv \npreprint arXiv:1608.05859, 2016.\n\n[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words \nwith subword units. arXiv preprint arXiv:1508.07909, 2015.\n\n[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, \nand Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts \nlayer. arXiv preprint arXiv:1701.06538, 2017.\n\n[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi- \nnov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine \nLearning Research, 15(1):1929–1958, 2014.\n\n[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory \nnetworks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, \nAdvances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates, \nInc., 2015.\n\n[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural \nnetworks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.\n\n[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. \nRethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.\n\n[37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In \nAdvances in Neural Information Processing Systems, 2015.\n\n[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang \nMacherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine \ntranslation system: Bridging the gap between human and machine translation. arXiv preprint \narXiv:1609.08144, 2016.\n\n[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with \nfast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.\n\n[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate \nshift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume \n1: Long Papers), pages 434–443. ACL, August 2013.\n```'),
     Document(id='c2da1709-b684-42e8-b2a7-02210fd4fcc4', metadata={}, page_content='# Attention Visualizations\n\n!Attention Visualizations\n\nFigure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.'),
     Document(id='a27715b4-f56c-4baf-8f58-d6c16eb2cb0f', metadata={}, page_content='NO_CONTENT_HERE'),
     Document(id='f8bf4ec2-efc2-4acb-9484-efc551db3915', metadata={}, page_content='NO_CONTENT_HERE')]</pre>



```python
# Display the content of the first document
print(docs[0].page_content)
```

<pre class="custom"># Attention Is All You Need
    
    Ashish Vaswani\*  
    Google Brain  
    avaswani@google.com  
    
    Noam Shazeer\*  
    Google Brain  
    noam@google.com  
    
    Niki Parmar\*  
    Google Research  
    nikip@google.com  
    
    Jakob Uszkoreit\*  
    Google Research  
    usz@google.com  
    
    Llion Jones\*  
    Google Research  
    llion@google.com  
    
    Aidan N. Gomez\* †  
    University of Toronto  
    aidan@cs.toronto.edu  
    
    Łukasz Kaiser\*  
    Google Brain  
    lukaszkaiser@google.com  
    
    Ilia Polosukhin\* ‡  
    illia.polosukhin@gmail.com  
    
    ## Abstract
    
    The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
    
    ----
    
    \*Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.
    
    †Work performed while at Google Brain.  
    ‡Work performed while at Google Research.
    
    31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
</pre>

## Custom Parsing Instructions

You can also specify custom instructions for parsing. This allows you to fine-tune the parser’s behavior to meet specific requirements.

```python
# Configure parsing instruction
parsing_instruction = "You are parsing a research paper. Summarize content of each page in markdown format."

# LlamaParse configuration
parser = LlamaParse(
    use_vendor_multimodal_model=True,
    # vendor_multimodal_model_name="openai-gpt4o",
    # vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"],
    result_type="markdown",
    language="en",
    parsing_instruction=parsing_instruction,
)

# Parse pdf file
parsed_docs = parser.load_data(file_path=FILE_PATH)

# Convert to langchain documents
docs = [doc.to_langchain_format() for doc in parsed_docs]
```

<pre class="custom">Started parsing the file under job_id 63423449-c6ff-41ff-9191-da14819b04a0
</pre>

```python
# Display the content of the first document
print(docs[0].page_content)
```

<pre class="custom">**Title:** Attention Is All You Need
    
    **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
    
    **Abstract Summary:**
    
    The paper introduces the Transformer, a novel neural network architecture based solely on attention mechanisms, eliminating the need for recurrence and convolutions. This model is designed for sequence transduction tasks, such as machine translation. The Transformer demonstrates superior performance, achieving 28.4 BLEU on the WMT 2014 English-to-German translation task and setting a new state-of-the-art BLEU score of 41.8 on the WMT 2014 English-to-French task. The model is more efficient, requiring less training time and resources, and generalizes well to other tasks, including English constituency parsing.
</pre>
