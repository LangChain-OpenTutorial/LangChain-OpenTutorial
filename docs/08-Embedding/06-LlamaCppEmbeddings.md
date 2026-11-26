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

# LlamaCpp Embeddings With Langchain

- Author: [Yongdam Kim](https://github.com/dancing-with-coffee/)
- Peer Review : [Pupba](https://github.com/pupba), [Teddy Lee](https://github.com/teddylee777)
- Proofread : [Youngjun cho](https://github.com/choincnp)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/06-LlamaCppEmbeddings.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/08-Embedding/06-LlamaCppEmbeddings.ipynb)
## Overview

This tutorial covers how to perform **Text Embedding** using **Llama-cpp** and **Langchain**.

**Llama-cpp** is an open-source package implemented in C++ that allows you to use LLMs such as llama very efficiently locally.

In this tutorial, we will create a simple example to measure similarity between ```Documents``` and an input ```Query``` using **Llama-cpp** and **Langchain**.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Llama-cpp Installation and Model Serving](#llama-cpp-installation-and-model-serving)
- [Identify Supported Embedding Models and Serving Model](#identify-supported-embedding-models-and-serving-model)
- [Model Load and Embedding](#model-load-and-embedding)
- [The similarity calculation results](#the-similarity-calculation-results)

### References

- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Llama-cpp Python GitHub](https://github.com/abetlen/llama-cpp-python)
- [LangChain Documentation](https://langchain.readthedocs.io/en/latest/)
- [Cosine Similarity - Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- [CompendiumLabs/bge-large-en-v1.5-gguf - Hugging Face](https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/tree/main)
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
        "langchain_community",
        "llama-cpp-python",
        "scikit-learn",
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
        "LANGCHAIN_API_KEY": "",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "LlamaCpp-Embeddings-With-Langchain",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set ```LANGCHAIN_API_KEY``` in ```.env``` file and load it. 

[Note] This is not necessary if you've already set ```LANGCHAIN_API_KEY``` in previous steps.

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">False</pre>



## Llama-cpp Installation and Model Serving

Llama-cpp is an open-source project that makes it easy to run large language models (LLMs) locally. It allows you to download and run various LLMs on your own computer, giving you the freedom to experiment with AI models.

To install **llama-cpp-python**:
```bash
pip install llama-cpp-python
```

1. Make sure you have the required environment for C++ compilation (e.g., on Linux or macOS). 
2. Download or specify your chosen embedding model file (e.g., ```CompendiumLabs/bge-large-en-v1.5-gguf```).
3. Here, we use ```bge-large-en-v1.5-q8_0.gguf``` as an example and you can download it from [CompendiumLabs/bge-large-en-v1.5-gguf - Hugging Face](https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/tree/main).
4. Check that ```llama-cpp-python``` can find the model path.

Below, we will demonstrate how to serve a LLaMA model using Llama-cpp. You can follow the official [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python) for more details.

## Identify Supported Embedding Models and Serving Model

You can find a variety of embedding models, which typically come in different quantizations (e.g., q4_0, q4_1, q5_0, q8_0, etc.).

**1. Search models**
- You can look for models on Hugging Face or other community websites.

**2. Download or Pull a Model**
- For instance, you could download from Hugging Face if the model is hosted.

**3. Verify the Model**
- Check that the ```.bin``` (or ```.gguf```) file is accessible to your environment.


## Model Load and Embedding

Now that you have installed ```llama-cpp-python``` and have downloaded a model, let's see how to load it and use it for text embedding.

Below, we define a ```Query``` or some ```Documents``` to embed using ```Llama-cpp``` within LangChain.

```python
from langchain_community.embeddings import LlamaCppEmbeddings

# Example query and documents
query = "What is LangChain?"
docs = [
    "LangChain is an open-source framework designed to facilitate the development of applications powered by large language models (LLMs). It provides tools and components to build end-to-end workflows for tasks like document retrieval, chatbots, summarization, data analysis, and more.",
    "Spaghetti Carbonara is a traditional Italian pasta dish made with eggs, cheese, pancetta, and pepper. It's simple yet incredibly delicious. Typically served with spaghetti, but can also be enjoyed with other pasta types.",
    "The tropical island of Bali offers stunning beaches, volcanic mountains, lush forests, and vibrant coral reefs. Travelers often visit for surfing, yoga retreats, and the unique Balinese Hindu culture.",
    "C++ is a high-performance programming language widely used in system/software development, game programming, and real-time simulations. It supports both procedural and object-oriented paradigms.",
    "In astronomy, the Drake Equation is a probabilistic argument used to estimate the number of active, communicative extraterrestrial civilizations in the Milky Way galaxy. It takes into account factors such as star formation rate and fraction of habitable planets.",
]
```

### Load the Embedding Model

Below is how you can initialize the ```LlamaCppEmbeddings``` class by specifying the path to your LLaMA model file (```model_path```).

For example, you might have a downloaded model path: ```./bge-large-en-v1.5-q8_0.gguf```.

We demonstrate how to instantiate the embeddings class and then embed queries and documents using Llama-cpp.

```python
# Load the Llama-cpp Embedding Model
model_path = "data/bge-large-en-v1.5-q8_0.gguf"  # example path

embedder = LlamaCppEmbeddings(model_path=model_path, n_gpu_layers=-1)
print("Embedding model has been successfully loaded.")
```

<pre class="custom">llama_load_model_from_file: using device Metal (Apple M3) - 16383 MiB free
    llama_model_loader: loaded meta data with 24 key-value pairs and 389 tensors from data/bge-large-en-v1.5-q8_0.gguf (version GGUF V3 (latest))
    llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
    llama_model_loader: - kv   0:                       general.architecture str              = bert
    llama_model_loader: - kv   1:                               general.name str              = bge-large-en-v1.5
    llama_model_loader: - kv   2:                           bert.block_count u32              = 24
    llama_model_loader: - kv   3:                        bert.context_length u32              = 512
    llama_model_loader: - kv   4:                      bert.embedding_length u32              = 1024
    llama_model_loader: - kv   5:                   bert.feed_forward_length u32              = 4096
    llama_model_loader: - kv   6:                  bert.attention.head_count u32              = 16
    llama_model_loader: - kv   7:          bert.attention.layer_norm_epsilon f32              = 0.000000
    llama_model_loader: - kv   8:                          general.file_type u32              = 7
    llama_model_loader: - kv   9:                      bert.attention.causal bool             = false
    llama_model_loader: - kv  10:                          bert.pooling_type u32              = 2
    llama_model_loader: - kv  11:            tokenizer.ggml.token_type_count u32              = 2
    llama_model_loader: - kv  12:                tokenizer.ggml.bos_token_id u32              = 101
    llama_model_loader: - kv  13:                tokenizer.ggml.eos_token_id u32              = 102
    llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = bert
    llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,30522]   = ["[PAD]", "[unused0]", "[unused1]", "...
    llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,30522]   = [-1000.000000, -1000.000000, -1000.00...
    llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,30522]   = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
    llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 100
    llama_model_loader: - kv  19:          tokenizer.ggml.seperator_token_id u32              = 102
    llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 0
    llama_model_loader: - kv  21:                tokenizer.ggml.cls_token_id u32              = 101
    llama_model_loader: - kv  22:               tokenizer.ggml.mask_token_id u32              = 103
    llama_model_loader: - kv  23:               general.quantization_version u32              = 2
    llama_model_loader: - type  f32:  244 tensors
    llama_model_loader: - type q8_0:  145 tensors
    llm_load_vocab: control token:    100 '[UNK]' is not marked as EOG
    llm_load_vocab: control token:    101 '[CLS]' is not marked as EOG
    llm_load_vocab: control token:      0 '[PAD]' is not marked as EOG
    llm_load_vocab: control token:    102 '[SEP]' is not marked as EOG
    llm_load_vocab: control token:    103 '[MASK]' is not marked as EOG
    llm_load_vocab: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
    llm_load_vocab: special tokens cache size = 5
    llm_load_vocab: token to piece cache size = 0.2032 MB
    llm_load_print_meta: format           = GGUF V3 (latest)
    llm_load_print_meta: arch             = bert
    llm_load_print_meta: vocab type       = WPM
    llm_load_print_meta: n_vocab          = 30522
    llm_load_print_meta: n_merges         = 0
    llm_load_print_meta: vocab_only       = 0
    llm_load_print_meta: n_ctx_train      = 512
    llm_load_print_meta: n_embd           = 1024
    llm_load_print_meta: n_layer          = 24
    llm_load_print_meta: n_head           = 16
    llm_load_print_meta: n_head_kv        = 16
    llm_load_print_meta: n_rot            = 64
    llm_load_print_meta: n_swa            = 0
    llm_load_print_meta: n_embd_head_k    = 64
    llm_load_print_meta: n_embd_head_v    = 64
    llm_load_print_meta: n_gqa            = 1
    llm_load_print_meta: n_embd_k_gqa     = 1024
    llm_load_print_meta: n_embd_v_gqa     = 1024
    llm_load_print_meta: f_norm_eps       = 1.0e-12
    llm_load_print_meta: f_norm_rms_eps   = 0.0e+00
    llm_load_print_meta: f_clamp_kqv      = 0.0e+00
    llm_load_print_meta: f_max_alibi_bias = 0.0e+00
    llm_load_print_meta: f_logit_scale    = 0.0e+00
    llm_load_print_meta: n_ff             = 4096
    llm_load_print_meta: n_expert         = 0
    llm_load_print_meta: n_expert_used    = 0
    llm_load_print_meta: causal attn      = 0
    llm_load_print_meta: pooling type     = 2
    llm_load_print_meta: rope type        = 2
    llm_load_print_meta: rope scaling     = linear
    llm_load_print_meta: freq_base_train  = 10000.0
    llm_load_print_meta: freq_scale_train = 1
    llm_load_print_meta: n_ctx_orig_yarn  = 512
    llm_load_print_meta: rope_finetuned   = unknown
    llm_load_print_meta: ssm_d_conv       = 0
    llm_load_print_meta: ssm_d_inner      = 0
    llm_load_print_meta: ssm_d_state      = 0
    llm_load_print_meta: ssm_dt_rank      = 0
    llm_load_print_meta: ssm_dt_b_c_rms   = 0
    llm_load_print_meta: model type       = 335M
    llm_load_print_meta: model ftype      = Q8_0
    llm_load_print_meta: model params     = 334.09 M
    llm_load_print_meta: model size       = 340.90 MiB (8.56 BPW) 
    llm_load_print_meta: general.name     = bge-large-en-v1.5
    llm_load_print_meta: BOS token        = 101 '[CLS]'
    llm_load_print_meta: EOS token        = 102 '[SEP]'
    llm_load_print_meta: UNK token        = 100 '[UNK]'
    llm_load_print_meta: SEP token        = 102 '[SEP]'
    llm_load_print_meta: PAD token        = 0 '[PAD]'
    llm_load_print_meta: CLS token        = 101 '[CLS]'
    llm_load_print_meta: MASK token       = 103 '[MASK]'
    llm_load_print_meta: LF token         = 0 '[PAD]'
    llm_load_print_meta: EOG token        = 102 '[SEP]'
    llm_load_print_meta: max token length = 21
    llm_load_tensors: tensor 'token_embd.weight' (q8_0) (and 4 others) cannot be used with preferred buffer type CPU_AARCH64, using CPU instead
    ggml_backend_metal_log_allocated_size: allocated buffer, size =   307.23 MiB, (  307.31 / 16384.02)
    llm_load_tensors: offloading 24 repeating layers to GPU
    llm_load_tensors: offloading output layer to GPU
    llm_load_tensors: offloaded 25/25 layers to GPU
    llm_load_tensors: Metal_Mapped model buffer size =   307.23 MiB
    llm_load_tensors:   CPU_Mapped model buffer size =    33.69 MiB
    ................................................................................
    llama_new_context_with_model: n_seq_max     = 1
    llama_new_context_with_model: n_ctx         = 512
    llama_new_context_with_model: n_ctx_per_seq = 512
    llama_new_context_with_model: n_batch       = 512
    llama_new_context_with_model: n_ubatch      = 512
    llama_new_context_with_model: flash_attn    = 0
    llama_new_context_with_model: freq_base     = 10000.0
    llama_new_context_with_model: freq_scale    = 1
    ggml_metal_init: allocating
    ggml_metal_init: found device: Apple M3
    ggml_metal_init: picking default device: Apple M3
    ggml_metal_init: using embedded metal library
    ggml_metal_init: GPU name:   Apple M3
    ggml_metal_init: GPU family: MTLGPUFamilyApple9  (1009)
    ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
    ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
    ggml_metal_init: simdgroup reduction   = true
    ggml_metal_init: simdgroup matrix mul. = true
    ggml_metal_init: has bfloat            = true
    ggml_metal_init: use bfloat            = false
    ggml_metal_init: hasUnifiedMemory      = true
    ggml_metal_init: recommendedMaxWorkingSetSize  = 17179.89 MB
    ggml_metal_init: loaded kernel_add                                    0x104d0f860 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_add_row                                0x104e4cd60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sub                                    0x121712be0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sub_row                                0x1078faf00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul                                    0x110765820 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_row                                0x104d0e830 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_div                                    0x110765070 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_div_row                                0x122f5e620 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_f32                             0x110766540 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_f16                             0x106fff000 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_i32                             0x106ffe710 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_i16                             0x110765e10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_scale                                  0x121804080 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_scale_4                                0x1078f78d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_clamp                                  0x1078fbe10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_tanh                                   0x106ffdd80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_relu                                   0x106fffa80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sigmoid                                0x121806040 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu                                   0x1078fc5a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu_4                                 0x121712380 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu_quick                             0x121806a80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu_quick_4                           0x121806600 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_silu                                   0x1078fc860 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_silu_4                                 0x1078fea90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_elu                                    0x1078fdcf0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f16                           0x104e4d190 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f16_4                         0x1078ff3c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f32                           0x121806d80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f32_4                         0x1078ffc00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_diag_mask_inf                          0x110768220 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_diag_mask_inf_8                        0x110768f60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_f32                           0x121713670 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_f16                           0x1078feec0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_get_rows_bf16                     (not supported)
    ggml_metal_init: loaded kernel_get_rows_q4_0                          0x122105900 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q4_1                          0x122105f80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q5_0                          0x1107689a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q5_1                          0x110766dc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q8_0                          0x11076a280 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q2_K                          0x121713c00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q3_K                          0x121713fd0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q4_K                          0x121715a80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q5_K                          0x121716210 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q6_K                          0x121808650 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq2_xxs                       0x1217166a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq2_xs                        0x121715650 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq3_xxs                       0x122106920 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq3_s                         0x122106f40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq2_s                         0x1218070d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq1_s                         0x11076a790 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq1_m                         0x121807fe0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq4_nl                        0x1218089d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq4_xs                        0x122107700 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_i32                           0x122107d40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rms_norm                               0x122108a30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_group_norm                             0x121717b10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_norm                                   0x12180a920 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_ssm_conv_f32                           0x121717710 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_ssm_scan_f32                           0x121717060 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f32_f32                         0x122108cf0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mv_bf16_f32                   (not supported)
    ggml_metal_init: skipping kernel_mul_mv_bf16_f32_1row              (not supported)
    ggml_metal_init: skipping kernel_mul_mv_bf16_f32_l4                (not supported)
    ggml_metal_init: skipping kernel_mul_mv_bf16_bf16                  (not supported)
    ggml_metal_init: loaded kernel_mul_mv_f16_f32                         0x121723680 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f16_f32_1row                    0x121723940 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f16_f32_l4                      0x1217249a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f16_f16                         0x1217244b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q4_0_f32                        0x11076ba80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q4_1_f32                        0x1217186e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q5_0_f32                        0x1217189a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q5_1_f32                        0x12180af90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q8_0_f32                        0x12180b530 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_2                0x12180b7f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_3                0x122109270 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_4                0x1217253f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_5                0x11076d140 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_2               0x122109e90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_3               0x121725cf0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_4               0x121718e60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_5               0x121726b30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_2               0x121727430 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_3               0x121726480 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_4               0x12210a620 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_5               0x12180bb30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_2               0x121727bc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_3               0x121727f70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_4               0x12180c2d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_5               0x11076c120 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_2               0x12210aac0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_3               0x12180cc70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_4               0x11076b4c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_5               0x121728e10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_2               0x1217296e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_3               0x122109a10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_4               0x122f5e9d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_5               0x12172a000 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_2               0x12180dbc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_3               0x12172a8a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_4               0x1217287f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_5               0x121829ec0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_2               0x12210bf40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_3               0x121829670 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_4               0x12172b1c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_5               0x12172baa0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_2               0x121829930 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_3               0x12210c820 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_4               0x12172c8d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_5               0x11076ea00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_2             0x11076d990 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_3             0x12210ad80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_4             0x12182b620 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_5             0x12210cf70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q2_K_f32                        0x12172c1d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q3_K_f32                        0x11076fb20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q4_K_f32                        0x11076f620 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q5_K_f32                        0x12210da20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q6_K_f32                        0x12172c490 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq2_xxs_f32                     0x12172d0a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq2_xs_f32                      0x12172ea40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq3_xxs_f32                     0x12182c4e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq3_s_f32                       0x1107702a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq2_s_f32                       0x110770b60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq1_s_f32                       0x12172f170 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq1_m_f32                       0x1107722a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq4_nl_f32                      0x110772b40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq4_xs_f32                      0x110773450 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_f32_f32                      0x12172f520 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_f16_f32                      0x12182ada0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mv_id_bf16_f32                (not supported)
    ggml_metal_init: loaded kernel_mul_mv_id_q4_0_f32                     0x110773e00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q4_1_f32                     0x110774720 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q5_0_f32                     0x12172fbc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q5_1_f32                     0x110775000 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q8_0_f32                     0x12182cbb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q2_K_f32                     0x12210ed20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q3_K_f32                     0x121730b50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q4_K_f32                     0x110771af0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q5_K_f32                     0x12172fe80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q6_K_f32                     0x12210e5b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq2_xxs_f32                  0x1107766f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq2_xs_f32                   0x1217312b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq3_xxs_f32                  0x121731980 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq3_s_f32                    0x1240133e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq2_s_f32                    0x12210e1f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq1_s_f32                    0x122111110 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq1_m_f32                    0x12182d340 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq4_nl_f32                   0x12210f940 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq4_xs_f32                   0x122111880 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_f32_f32                         0x12182d020 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_f16_f32                         0x110771400 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mm_bf16_f32                   (not supported)
    ggml_metal_init: loaded kernel_mul_mm_q4_0_f32                        0x110778380 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q4_1_f32                        0x122112f90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q5_0_f32                        0x12182d600 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q5_1_f32                        0x121731c40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q8_0_f32                        0x12182de30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q2_K_f32                        0x110778f00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q3_K_f32                        0x110777090 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q4_K_f32                        0x122113b40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q5_K_f32                        0x122113470 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q6_K_f32                        0x122114200 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq2_xxs_f32                     0x1217326c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq2_xs_f32                      0x121732b30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq3_xxs_f32                     0x1221126c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq3_s_f32                       0x121732fa0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq2_s_f32                       0x110779da0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq1_s_f32                       0x1221150c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq1_m_f32                       0x122114980 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq4_nl_f32                      0x121733480 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq4_xs_f32                      0x11077a520 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_f32_f32                      0x11077b330 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_f16_f32                      0x12182f380 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mm_id_bf16_f32                (not supported)
    ggml_metal_init: loaded kernel_mul_mm_id_q4_0_f32                     0x12182f870 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q4_1_f32                     0x121733740 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q5_0_f32                     0x1107795c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q5_1_f32                     0x121830e70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q8_0_f32                     0x121831ae0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q2_K_f32                     0x110777cc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q3_K_f32                     0x121832530 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q4_K_f32                     0x1218330b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q5_K_f32                     0x11077bc50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q6_K_f32                     0x122116ac0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq2_xxs_f32                  0x121832cc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq2_xs_f32                   0x122f5ee60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq3_xxs_f32                  0x11077c570 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq3_s_f32                    0x11077df30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq2_s_f32                    0x122116330 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq1_s_f32                    0x11077ce80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq1_m_f32                    0x121734190 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq4_nl_f32                   0x121832050 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq4_xs_f32                   0x11077d510 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_norm_f32                          0x121833cf0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_norm_f16                          0x121834c60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_neox_f32                          0x11077e6e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_neox_f16                          0x1221174d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_f16                             0x121734860 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_f32                             0x1217353a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_ext_f16                         0x122118330 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_ext_f32                         0x11077edf0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_conv_transpose_1d_f32_f32              0x121834590 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_conv_transpose_1d_f16_f32              0x11077fa60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_upscale_f32                            0x11077ffe0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pad_f32                                0x110780550 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pad_reflect_1d_f32                     0x110780c70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_timestep_embedding_f32                 0x121736820 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_arange_f32                             0x104e4c870 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_argsort_f32_i32_asc                    0x1218358e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_argsort_f32_i32_desc                   0x121835240 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_leaky_relu_f32                         0x1218363a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h64                 0x121737dc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h80                 0x121837d30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h96                 0x121736e70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h112                0x122119210 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h128                0x121838640 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h256                0x1218393c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h64           (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h80           (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h96           (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h112          (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h128          (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h256          (not supported)
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h64                0x1107832e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h80                0x121738f80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h96                0x121839b00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h112               0x121839f90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h128               0x12183a520 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h256               0x12211a060 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h64                0x12211a3f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h80                0x12211aff0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h96                0x12183b3d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h112               0x12211c200 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h128               0x12211bc60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h256               0x1107837f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h64                0x121737240 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h80                0x1217395e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h96                0x12173a0e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h112               0x12183a820 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h128               0x110782710 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h256               0x12183aae0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h64                0x12173ad90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h80                0x12173b310 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h96                0x12211cae0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h112               0x12173b5d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h128               0x12173b890 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h256               0x12173bb50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h64                0x110783d40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h80                0x12173c9a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h96                0x12211a730 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h112               0x12211ce60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h128               0x12173d0d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h256               0x12183cb20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_f16_h128            0x12183cde0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h128      (not supported)
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_0_h128           0x12173e6d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_1_h128           0x12173ece0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_0_h128           0x12183d170 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_1_h128           0x12211d560 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q8_0_h128           0x12211e460 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_f16_h256            0x12211dc80 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h256      (not supported)
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_0_h256           0x12183eab0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_1_h256           0x12183dac0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_0_h256           0x12183de30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_1_h256           0x110784770 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q8_0_h256           0x12173f210 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_set_f32                                0x110786cb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_set_i32                                0x12211df40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_f32                            0x110786450 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_f16                            0x12211ff10 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_cpy_f32_bf16                      (not supported)
    ggml_metal_init: loaded kernel_cpy_f16_f32                            0x122120c50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f16_f16                            0x12211f6d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_cpy_bf16_f32                      (not supported)
    ggml_metal_init: skipping kernel_cpy_bf16_bf16                     (not supported)
    ggml_metal_init: loaded kernel_cpy_f32_q8_0                           0x12183f920 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q4_0                           0x12183f2a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q4_1                           0x122121380 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q5_0                           0x1221221d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q5_1                           0x122122a80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_iq4_nl                         0x12173fe30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_concat                                 0x121740ae0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sqr                                    0x110785c10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sqrt                                   0x1107873f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sin                                    0x110788120 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cos                                    0x110789070 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sum_rows                               0x12183ff00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_argmax                                 0x110789b10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pool_2d_avg_f32                        0x121741210 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pool_2d_max_f32                        0x122121d00 | th_max = 1024 | th_width =   32
    llama_kv_cache_init:      Metal KV buffer size =    48.00 MiB
    llama_new_context_with_model: KV self size  =   48.00 MiB, K (f16):   24.00 MiB, V (f16):   24.00 MiB
    llama_new_context_with_model:        CPU  output buffer size =     0.00 MiB
    llama_new_context_with_model:      Metal compute buffer size =    25.00 MiB
    llama_new_context_with_model:        CPU compute buffer size =     5.01 MiB
    llama_new_context_with_model: graph nodes  = 849
    llama_new_context_with_model: graph splits = 2
    Metal : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | ACCELERATE = 1 | AARCH64_REPACK = 1 | 
</pre>

    Embedding model has been successfully loaded.
    

    Model metadata: {'tokenizer.ggml.cls_token_id': '101', 'tokenizer.ggml.padding_token_id': '0', 'tokenizer.ggml.seperator_token_id': '102', 'tokenizer.ggml.unknown_token_id': '100', 'general.quantization_version': '2', 'tokenizer.ggml.token_type_count': '2', 'general.file_type': '7', 'tokenizer.ggml.eos_token_id': '102', 'bert.context_length': '512', 'bert.pooling_type': '2', 'tokenizer.ggml.bos_token_id': '101', 'bert.attention.head_count': '16', 'bert.feed_forward_length': '4096', 'tokenizer.ggml.mask_token_id': '103', 'tokenizer.ggml.model': 'bert', 'bert.attention.causal': 'false', 'general.name': 'bge-large-en-v1.5', 'bert.block_count': '24', 'bert.attention.layer_norm_epsilon': '0.000000', 'bert.embedding_length': '1024', 'general.architecture': 'bert'}
    Using fallback chat format: llama-2
    

### Embedding Queries and Documents

Now let's embed both the ```query``` and the ```documents```. We will verify the dimension of the output vectors.

However, there is currently one issue that cannot be resolved when using the latest model with ```LlamaCppEmbeddings```. I will post the link to the issue below, so please check it out and if it is resolved in the latest version, you can use it as instructed in the original langchain official tutorial.

- Issue link : https://github.com/langchain-ai/langchain/issues/22532

```python
# from langchain tutorial

"""
embedded_query = llama_embeddings.embed_query(query)
embedded_docs = llama_embeddings.embed_documents(docs)

print(f"Embedding Dimension Output (Query): {len(embedded_query)}")
print(f"Embedding Dimension Output (Docs): {len(embedded_docs[0])}")
"""

# Overridden version of the LlamaCppEmbeddings class
from typing import List
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings


class CustomLlamaCppEmbeddings(LlamaCppEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed([text])[0] for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)[0]
        return list(map(float, embedding))


c_embedder = CustomLlamaCppEmbeddings(model_path=model_path, n_gpu_layers=-1)
embedded_query = c_embedder.embed_query([query])
embedded_docs = c_embedder.embed_documents(docs)
```

<pre class="custom">llama_load_model_from_file: using device Metal (Apple M3) - 15997 MiB free
    llama_model_loader: loaded meta data with 24 key-value pairs and 389 tensors from data/bge-large-en-v1.5-q8_0.gguf (version GGUF V3 (latest))
    llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
    llama_model_loader: - kv   0:                       general.architecture str              = bert
    llama_model_loader: - kv   1:                               general.name str              = bge-large-en-v1.5
    llama_model_loader: - kv   2:                           bert.block_count u32              = 24
    llama_model_loader: - kv   3:                        bert.context_length u32              = 512
    llama_model_loader: - kv   4:                      bert.embedding_length u32              = 1024
    llama_model_loader: - kv   5:                   bert.feed_forward_length u32              = 4096
    llama_model_loader: - kv   6:                  bert.attention.head_count u32              = 16
    llama_model_loader: - kv   7:          bert.attention.layer_norm_epsilon f32              = 0.000000
    llama_model_loader: - kv   8:                          general.file_type u32              = 7
    llama_model_loader: - kv   9:                      bert.attention.causal bool             = false
    llama_model_loader: - kv  10:                          bert.pooling_type u32              = 2
    llama_model_loader: - kv  11:            tokenizer.ggml.token_type_count u32              = 2
    llama_model_loader: - kv  12:                tokenizer.ggml.bos_token_id u32              = 101
    llama_model_loader: - kv  13:                tokenizer.ggml.eos_token_id u32              = 102
    llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = bert
    llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,30522]   = ["[PAD]", "[unused0]", "[unused1]", "...
    llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,30522]   = [-1000.000000, -1000.000000, -1000.00...
    llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,30522]   = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
    llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 100
    llama_model_loader: - kv  19:          tokenizer.ggml.seperator_token_id u32              = 102
    llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 0
    llama_model_loader: - kv  21:                tokenizer.ggml.cls_token_id u32              = 101
    llama_model_loader: - kv  22:               tokenizer.ggml.mask_token_id u32              = 103
    llama_model_loader: - kv  23:               general.quantization_version u32              = 2
    llama_model_loader: - type  f32:  244 tensors
    llama_model_loader: - type q8_0:  145 tensors
    llm_load_vocab: control token:    100 '[UNK]' is not marked as EOG
    llm_load_vocab: control token:    101 '[CLS]' is not marked as EOG
    llm_load_vocab: control token:      0 '[PAD]' is not marked as EOG
    llm_load_vocab: control token:    102 '[SEP]' is not marked as EOG
    llm_load_vocab: control token:    103 '[MASK]' is not marked as EOG
    llm_load_vocab: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
    llm_load_vocab: special tokens cache size = 5
    llm_load_vocab: token to piece cache size = 0.2032 MB
    llm_load_print_meta: format           = GGUF V3 (latest)
    llm_load_print_meta: arch             = bert
    llm_load_print_meta: vocab type       = WPM
    llm_load_print_meta: n_vocab          = 30522
    llm_load_print_meta: n_merges         = 0
    llm_load_print_meta: vocab_only       = 0
    llm_load_print_meta: n_ctx_train      = 512
    llm_load_print_meta: n_embd           = 1024
    llm_load_print_meta: n_layer          = 24
    llm_load_print_meta: n_head           = 16
    llm_load_print_meta: n_head_kv        = 16
    llm_load_print_meta: n_rot            = 64
    llm_load_print_meta: n_swa            = 0
    llm_load_print_meta: n_embd_head_k    = 64
    llm_load_print_meta: n_embd_head_v    = 64
    llm_load_print_meta: n_gqa            = 1
    llm_load_print_meta: n_embd_k_gqa     = 1024
    llm_load_print_meta: n_embd_v_gqa     = 1024
    llm_load_print_meta: f_norm_eps       = 1.0e-12
    llm_load_print_meta: f_norm_rms_eps   = 0.0e+00
    llm_load_print_meta: f_clamp_kqv      = 0.0e+00
    llm_load_print_meta: f_max_alibi_bias = 0.0e+00
    llm_load_print_meta: f_logit_scale    = 0.0e+00
    llm_load_print_meta: n_ff             = 4096
    llm_load_print_meta: n_expert         = 0
    llm_load_print_meta: n_expert_used    = 0
    llm_load_print_meta: causal attn      = 0
    llm_load_print_meta: pooling type     = 2
    llm_load_print_meta: rope type        = 2
    llm_load_print_meta: rope scaling     = linear
    llm_load_print_meta: freq_base_train  = 10000.0
    llm_load_print_meta: freq_scale_train = 1
    llm_load_print_meta: n_ctx_orig_yarn  = 512
    llm_load_print_meta: rope_finetuned   = unknown
    llm_load_print_meta: ssm_d_conv       = 0
    llm_load_print_meta: ssm_d_inner      = 0
    llm_load_print_meta: ssm_d_state      = 0
    llm_load_print_meta: ssm_dt_rank      = 0
    llm_load_print_meta: ssm_dt_b_c_rms   = 0
    llm_load_print_meta: model type       = 335M
    llm_load_print_meta: model ftype      = Q8_0
    llm_load_print_meta: model params     = 334.09 M
    llm_load_print_meta: model size       = 340.90 MiB (8.56 BPW) 
    llm_load_print_meta: general.name     = bge-large-en-v1.5
    llm_load_print_meta: BOS token        = 101 '[CLS]'
    llm_load_print_meta: EOS token        = 102 '[SEP]'
    llm_load_print_meta: UNK token        = 100 '[UNK]'
    llm_load_print_meta: SEP token        = 102 '[SEP]'
    llm_load_print_meta: PAD token        = 0 '[PAD]'
    llm_load_print_meta: CLS token        = 101 '[CLS]'
    llm_load_print_meta: MASK token       = 103 '[MASK]'
    llm_load_print_meta: LF token         = 0 '[PAD]'
    llm_load_print_meta: EOG token        = 102 '[SEP]'
    llm_load_print_meta: max token length = 21
    llm_load_tensors: tensor 'token_embd.weight' (q8_0) (and 4 others) cannot be used with preferred buffer type CPU_AARCH64, using CPU instead
    ggml_backend_metal_log_allocated_size: allocated buffer, size =   307.23 MiB, (  693.44 / 16384.02)
    llm_load_tensors: offloading 24 repeating layers to GPU
    llm_load_tensors: offloading output layer to GPU
    llm_load_tensors: offloaded 25/25 layers to GPU
    llm_load_tensors: Metal_Mapped model buffer size =   307.23 MiB
    llm_load_tensors:   CPU_Mapped model buffer size =    33.69 MiB
    ................................................................................
    llama_new_context_with_model: n_seq_max     = 1
    llama_new_context_with_model: n_ctx         = 512
    llama_new_context_with_model: n_ctx_per_seq = 512
    llama_new_context_with_model: n_batch       = 512
    llama_new_context_with_model: n_ubatch      = 512
    llama_new_context_with_model: flash_attn    = 0
    llama_new_context_with_model: freq_base     = 10000.0
    llama_new_context_with_model: freq_scale    = 1
    ggml_metal_init: allocating
    ggml_metal_init: found device: Apple M3
    ggml_metal_init: picking default device: Apple M3
    ggml_metal_init: using embedded metal library
    ggml_metal_init: GPU name:   Apple M3
    ggml_metal_init: GPU family: MTLGPUFamilyApple9  (1009)
    ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
    ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
    ggml_metal_init: simdgroup reduction   = true
    ggml_metal_init: simdgroup matrix mul. = true
    ggml_metal_init: has bfloat            = true
    ggml_metal_init: use bfloat            = false
    ggml_metal_init: hasUnifiedMemory      = true
    ggml_metal_init: recommendedMaxWorkingSetSize  = 17179.89 MB
    ggml_metal_init: loaded kernel_add                                    0x104541960 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_add_row                                0x1218415b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sub                                    0x121841870 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sub_row                                0x121841b30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul                                    0x1107e7c70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_row                                0x1218408e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_div                                    0x121735910 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_div_row                                0x1107e74c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_f32                             0x1107e7780 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_f16                             0x12210d300 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_i32                             0x1107e8e30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_repeat_i16                             0x12187e120 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_scale                                  0x104ff3320 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_scale_4                                0x104e4d4a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_clamp                                  0x1107e9920 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_tanh                                   0x122122e50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_relu                                   0x1107ea300 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sigmoid                                0x122123110 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu                                   0x1221233d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu_4                                 0x12187e660 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu_quick                             0x1107ea950 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_gelu_quick_4                           0x12187ebb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_silu                                   0x12187f0f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_silu_4                                 0x12187f630 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_elu                                    0x12187fb70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f16                           0x1218800b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f16_4                         0x104ff35e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f32                           0x121880370 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_soft_max_f32_4                         0x122123690 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_diag_mask_inf                          0x121880630 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_diag_mask_inf_8                        0x1107ebd20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_f32                           0x1107ead60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_f16                           0x1218808f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_get_rows_bf16                     (not supported)
    ggml_metal_init: loaded kernel_get_rows_q4_0                          0x121880bb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q4_1                          0x121880e70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q5_0                          0x1107ecac0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q5_1                          0x121881130 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q8_0                          0x104ff38a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q2_K                          0x1218813f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q3_K                          0x104ff3b60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q4_K                          0x1107ed960 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q5_K                          0x1218816b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_q6_K                          0x121881970 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq2_xxs                       0x121881c30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq2_xs                        0x1107edc80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq3_xxs                       0x1107edf40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq3_s                         0x1107ee2f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq2_s                         0x121881ef0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq1_s                         0x1218821b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq1_m                         0x121882470 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq4_nl                        0x122123950 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_iq4_xs                        0x121882730 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_get_rows_i32                           0x1218829f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rms_norm                               0x121882cb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_group_norm                             0x1107ef9d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_norm                                   0x1107ee810 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_ssm_conv_f32                           0x121882f70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_ssm_scan_f32                           0x104ff4590 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f32_f32                         0x1107efdc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mv_bf16_f32                   (not supported)
    ggml_metal_init: skipping kernel_mul_mv_bf16_f32_1row              (not supported)
    ggml_metal_init: skipping kernel_mul_mv_bf16_f32_l4                (not supported)
    ggml_metal_init: skipping kernel_mul_mv_bf16_bf16                  (not supported)
    ggml_metal_init: loaded kernel_mul_mv_f16_f32                         0x1107f00d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f16_f32_1row                    0x121883670 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f16_f32_l4                      0x104ff4850 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_f16_f16                         0x121883c30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q4_0_f32                        0x104ff5920 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q4_1_f32                        0x122123c10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q5_0_f32                        0x121883ef0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q5_1_f32                        0x1107f04d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q8_0_f32                        0x1107f1c10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_2                0x1107f1690 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_3                0x1218841b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_4                0x122123ed0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_f16_f32_r1_5                0x122124190 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_2               0x121884620 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_3               0x1107f2060 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_4               0x121884d30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_0_f32_r1_5               0x122124450 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_2               0x1107f28e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_3               0x1107f3a20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_4               0x12170d600 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_1_f32_r1_5               0x1107f3220 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_2               0x1107f34e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_3               0x121885310 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_4               0x1107f4b40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_0_f32_r1_5               0x121886150 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_2               0x1107f58a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_3               0x12170d8c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_4               0x1107f6690 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_1_f32_r1_5               0x122124710 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_2               0x1221249d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_3               0x121886890 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_4               0x121887770 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q8_0_f32_r1_5               0x12170db80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_2               0x122124c90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_3               0x1218859a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_4               0x122124f50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q4_K_f32_r1_5               0x122125210 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_2               0x1221254d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_3               0x122125790 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_4               0x12170e010 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q5_K_f32_r1_5               0x121888bc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_2               0x122125a50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_3               0x1218880c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_4               0x122126090 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_q6_K_f32_r1_5               0x1221269b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_2             0x121888540 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_3             0x1221272c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_4             0x1107f6b30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_ext_iq4_nl_f32_r1_5             0x1107f7440 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q2_K_f32                        0x122127b70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q3_K_f32                        0x121889ca0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q4_K_f32                        0x12188a680 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q5_K_f32                        0x122127f40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_q6_K_f32                        0x122f0cdd0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq2_xxs_f32                     0x1240140c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq2_xs_f32                      0x122f0c910 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq3_xxs_f32                     0x122128990 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq3_s_f32                       0x12188aa70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq2_s_f32                       0x12188bd00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq1_s_f32                       0x122128310 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq1_m_f32                       0x12170ed50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq4_nl_f32                      0x1107f7b60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_iq4_xs_f32                      0x12188b5b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_f32_f32                      0x12170e760 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_f16_f32                      0x12170f640 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mv_id_bf16_f32                (not supported)
    ggml_metal_init: loaded kernel_mul_mv_id_q4_0_f32                     0x12188b870 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q4_1_f32                     0x12188d260 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q5_0_f32                     0x12188dbc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q5_1_f32                     0x12188c6c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q8_0_f32                     0x12188cbb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q2_K_f32                     0x122129110 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q3_K_f32                     0x1107f89b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q4_K_f32                     0x12170ff20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q5_K_f32                     0x12188ee30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_q6_K_f32                     0x1107f7f30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq2_xxs_f32                  0x1107f9140 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq2_xs_f32                   0x122129770 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq3_xxs_f32                  0x12188fc70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq3_s_f32                    0x1107f9f60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq2_s_f32                    0x1218904c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq1_s_f32                    0x121712640 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq1_m_f32                    0x121710970 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq4_nl_f32                   0x12212b310 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mv_id_iq4_xs_f32                   0x121890c50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_f32_f32                         0x121891370 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_f16_f32                         0x121891790 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mm_bf16_f32                   (not supported)
    ggml_metal_init: loaded kernel_mul_mm_q4_0_f32                        0x12212a6d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q4_1_f32                        0x121892420 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q5_0_f32                        0x12212bd10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q5_1_f32                        0x1107fae50 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q8_0_f32                        0x12212cb30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q2_K_f32                        0x121711b70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q3_K_f32                        0x12212d350 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q4_K_f32                        0x12212dba0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q5_K_f32                        0x1107fb6e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_q6_K_f32                        0x1107f96c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq2_xxs_f32                     0x12212c0c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq2_xs_f32                      0x1107fc3b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq3_xxs_f32                     0x12212e430 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq3_s_f32                       0x12212f060 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq2_s_f32                       0x1107fcb40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq1_s_f32                       0x121741bb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq1_m_f32                       0x121711580 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq4_nl_f32                      0x1107fc670 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_iq4_xs_f32                      0x121892720 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_f32_f32                      0x122130920 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_f16_f32                      0x121893770 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_mul_mm_id_bf16_f32                (not supported)
    ggml_metal_init: loaded kernel_mul_mm_id_q4_0_f32                     0x12212eb10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q4_1_f32                     0x1217423e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q5_0_f32                     0x122131950 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q5_1_f32                     0x121894230 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q8_0_f32                     0x1107fe260 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q2_K_f32                     0x1107fd550 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q3_K_f32                     0x122132240 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q4_K_f32                     0x122132a10 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q5_K_f32                     0x121711170 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_q6_K_f32                     0x121743ca0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq2_xxs_f32                  0x1221333e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq2_xs_f32                   0x122133d00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq3_xxs_f32                  0x1107fef00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq3_s_f32                    0x1218949f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq2_s_f32                    0x121744430 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq1_s_f32                    0x1107fe940 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq1_m_f32                    0x122132cd0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq4_nl_f32                   0x121894f60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_mul_mm_id_iq4_xs_f32                   0x121895b40 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_norm_f32                          0x1107fdac0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_norm_f16                          0x122134410 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_neox_f32                          0x121745310 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_rope_neox_f16                          0x12170b210 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_f16                             0x1107ffbc0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_f32                             0x1221346d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_ext_f16                         0x124404150 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_im2col_ext_f32                         0x12170b640 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_conv_transpose_1d_f32_f32              0x122135100 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_conv_transpose_1d_f16_f32              0x122f0d2b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_upscale_f32                            0x1240146e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pad_f32                                0x124404850 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pad_reflect_1d_f32                     0x121895640 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_timestep_embedding_f32                 0x1221357b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_arange_f32                             0x1244051a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_argsort_f32_i32_asc                    0x1218965d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_argsort_f32_i32_desc                   0x124405890 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_leaky_relu_f32                         0x121896c30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h64                 0x1221370b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h80                 0x1217459d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h96                 0x121897db0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h112                0x124407ec0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h128                0x124408180 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_f16_h256                0x124408950 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h64           (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h80           (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h96           (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h112          (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h128          (not supported)
    ggml_metal_init: skipping kernel_flash_attn_ext_bf16_h256          (not supported)
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h64                0x124409820 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h80                0x124409220 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h96                0x122137860 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h112               0x12440a140 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h128               0x122137e80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_0_h256               0x122136a60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h64                0x122138730 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h80                0x122139c70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h96                0x121898570 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h112               0x12440aa70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h128               0x12440be00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q4_1_h256               0x122139310 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h64                0x12213ada0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h80                0x1218994f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h96                0x12440b440 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h112               0x121899e00 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h128               0x12440b800 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_0_h256               0x12440cb90 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h64                0x12170cb60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h80                0x12170bb60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h96                0x12189adb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h112               0x121747070 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h128               0x1217479b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q5_1_h256               0x12213ba30 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h64                0x12213a900 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h80                0x12440db70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h96                0x12213c970 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h112               0x12440e480 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h128               0x12213d270 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_q8_0_h256               0x12440cf20 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_f16_h128            0x12213db70 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h128      (not supported)
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_0_h128           0x121747fd0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_1_h128           0x12189b9d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_0_h128           0x12189b580 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_1_h128           0x12213e4d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q8_0_h128           0x12189a960 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_f16_h256            0x12440e7c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_flash_attn_ext_vec_bf16_h256      (not supported)
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_0_h256           0x12213eda0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q4_1_h256           0x12189d4d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_0_h256           0x12440f6d0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q5_1_h256           0x12440eca0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_flash_attn_ext_vec_q8_0_h256           0x12170c550 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_set_f32                                0x12440ff80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_set_i32                                0x12189ced0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_f32                            0x124410240 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_f16                            0x1217483e0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_cpy_f32_bf16                      (not supported)
    ggml_metal_init: loaded kernel_cpy_f16_f32                            0x12213f1c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f16_f16                            0x12213fae0 | th_max = 1024 | th_width =   32
    ggml_metal_init: skipping kernel_cpy_bf16_f32                      (not supported)
    ggml_metal_init: skipping kernel_cpy_bf16_bf16                     (not supported)
    ggml_metal_init: loaded kernel_cpy_f32_q8_0                           0x12189ddd0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q4_0                           0x121749900 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q4_1                           0x124410500 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q5_0                           0x1217493a0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_q5_1                           0x124411b70 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cpy_f32_iq4_nl                         0x1221404f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_concat                                 0x12189e8c0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sqr                                    0x12189c5f0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sqrt                                   0x121748e80 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sin                                    0x1244113b0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_cos                                    0x12174acf0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_sum_rows                               0x124412e60 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_argmax                                 0x12174bbb0 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pool_2d_avg_f32                        0x124412130 | th_max = 1024 | th_width =   32
    ggml_metal_init: loaded kernel_pool_2d_max_f32                        0x1244145f0 | th_max = 1024 | th_width =   32
    llama_kv_cache_init:      Metal KV buffer size =    48.00 MiB
    llama_new_context_with_model: KV self size  =   48.00 MiB, K (f16):   24.00 MiB, V (f16):   24.00 MiB
    llama_new_context_with_model:        CPU  output buffer size =     0.00 MiB
    llama_new_context_with_model:      Metal compute buffer size =    25.00 MiB
    llama_new_context_with_model:        CPU compute buffer size =     5.01 MiB
    llama_new_context_with_model: graph nodes  = 849
    llama_new_context_with_model: graph splits = 2
    Metal : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | ACCELERATE = 1 | AARCH64_REPACK = 1 | Metal : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | ACCELERATE = 1 | AARCH64_REPACK = 1 | 
    Model metadata: {'tokenizer.ggml.cls_token_id': '101', 'tokenizer.ggml.padding_token_id': '0', 'tokenizer.ggml.seperator_token_id': '102', 'tokenizer.ggml.unknown_token_id': '100', 'general.quantization_version': '2', 'tokenizer.ggml.token_type_count': '2', 'general.file_type': '7', 'tokenizer.ggml.eos_token_id': '102', 'bert.context_length': '512', 'bert.pooling_type': '2', 'tokenizer.ggml.bos_token_id': '101', 'bert.attention.head_count': '16', 'bert.feed_forward_length': '4096', 'tokenizer.ggml.mask_token_id': '103', 'tokenizer.ggml.model': 'bert', 'bert.attention.causal': 'false', 'general.name': 'bge-large-en-v1.5', 'bert.block_count': '24', 'bert.attention.layer_norm_epsilon': '0.000000', 'bert.embedding_length': '1024', 'general.architecture': 'bert'}
    Using fallback chat format: llama-2
    llama_perf_context_print:        load time =      59.73 ms
    llama_perf_context_print: prompt eval time =       0.00 ms /     8 tokens (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:       total time =      59.76 ms /     9 tokens
    llama_perf_context_print:        load time =      59.73 ms
    llama_perf_context_print: prompt eval time =       0.00 ms /    62 tokens (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:       total time =      29.66 ms /    63 tokens
    llama_perf_context_print:        load time =      59.73 ms
    llama_perf_context_print: prompt eval time =       0.00 ms /    47 tokens (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:       total time =      28.45 ms /    48 tokens
    llama_perf_context_print:        load time =      59.73 ms
    llama_perf_context_print: prompt eval time =       0.00 ms /    40 tokens (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:       total time =      31.09 ms /    41 tokens
    llama_perf_context_print:        load time =      59.73 ms
    llama_perf_context_print: prompt eval time =       0.00 ms /    40 tokens (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:       total time =      30.95 ms /    41 tokens
    llama_perf_context_print:        load time =      59.73 ms
    llama_perf_context_print: prompt eval time =       0.00 ms /    54 tokens (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
    llama_perf_context_print:       total time =      28.12 ms /    55 tokens
</pre>

### Check custom embeddings

- To check whether the embedding results are output as expected, I output the dimensions of each embedding vector.

```python
print("Query embedding dimension:", len(embedded_query))
print("Document embedding dimension:", len(embedded_docs[0]))
```

<pre class="custom">Query embedding dimension: 1024
    Document embedding dimension: 1024
</pre>

## The similarity calculation results

We can use the vector representations of the query and documents to calculate similarity.
Here, we use the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) provided by scikit-learn.


```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Calculate Cosine Similarity
similarities = cosine_similarity([embedded_query], embedded_docs)[0]
print(similarities)

# Sort indices in ascending order.
sorted_indices = np.argsort(similarities)[::-1]

print(f"[Query] {query}\n====================================")
for i, idx in enumerate(sorted_indices):
    print(f"[{i}] similarity: {similarities[idx]:.3f} | {docs[idx]}")
    print()
```

<pre class="custom">[0.87119322 0.37936658 0.29399348 0.49741913 0.38813166]
    [Query] What is LangChain?
    ====================================
    [0] similarity: 0.871 | LangChain is an open-source framework designed to facilitate the development of applications powered by large language models (LLMs). It provides tools and components to build end-to-end workflows for tasks like document retrieval, chatbots, summarization, data analysis, and more.
    
    [1] similarity: 0.497 | C++ is a high-performance programming language widely used in system/software development, game programming, and real-time simulations. It supports both procedural and object-oriented paradigms.
    
    [2] similarity: 0.388 | In astronomy, the Drake Equation is a probabilistic argument used to estimate the number of active, communicative extraterrestrial civilizations in the Milky Way galaxy. It takes into account factors such as star formation rate and fraction of habitable planets.
    
    [3] similarity: 0.379 | Spaghetti Carbonara is a traditional Italian pasta dish made with eggs, cheese, pancetta, and pepper. It's simple yet incredibly delicious. Typically served with spaghetti, but can also be enjoyed with other pasta types.
    
    [4] similarity: 0.294 | The tropical island of Bali offers stunning beaches, volcanic mountains, lush forests, and vibrant coral reefs. Travelers often visit for surfing, yoga retreats, and the unique Balinese Hindu culture.
    
</pre>

----
This concludes the **Llama-cpp Embeddings With Langchain** tutorial in the style of the original reference notebook.
