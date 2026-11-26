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

# Using Various LLM Models

- Author: [eunhhyy](https://github.com/eunhhyy)
- Peer Review : [Wooseok Jeong](https://github.com/jeong-wooseok)
- Proofread : [Chaeyoon Kim](https://github.com/chaeyoonyunakim)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/01-Models.ipynb)[![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/04-Model/01-Models.ipynb)
## Overview

This tutorial provides a comprehensive guide to major ```Large Language Models (LLMs)``` in the AI Market.

### Table of Contents

- [Overview](#overview)
- [OpenAI GPT Series](#openai-gpt-series)
- [Meta Llama Series](#meta-llama-series)
- [Anthropic Claude Series](#anthropic-claude-series)
- [Google Gemini Series](#google-gemini-series)
- [Mistral AI models Series](#mistral-ai-models-series)
- [Alibaba Qwen Series](#alibaba-qwen-series)


### References
- [OpenAI's models overview](https://platform.openai.com/docs/models#models-overview).
- [Meta's models overview](https://www.llama.com/).
- [Anthropic's models overview](https://docs.anthropic.com/en/docs/intro-to-claude).
- [Google’s models overview](https://ai.google.dev/gemini-api/docs/models/gemini).
- [Mistral's models overview](https://mistral.ai/technology/#models).
- [Alibaba Cloud’s models overview](https://mistral.ai/technology/#models).

----

## OpenAI - GPT Series

GPT models by OpenAI are advanced transformer-based language models designed for tasks like text generation, summarization, translation, and Q&A. Offered primarily as a cloud-based API, they let developers use the models without hosting them. While not open-source, GPT provides pre-trained models with fine-tuning capabilities.

### Model Variants

1. **GPT-4o Series (Flagship Models)**
   - **GPT-4o**: High-reliability model with improved speed over Turbo
   - **GPT-4-turbo**: Latest model with vision, JSON, and function calling capabilities
   - **GPT-4o-mini**: Entry-level model surpassing GPT-3.5 Turbo performance

2. **O1 Series (Reasoning Specialists)**
   - **O1**: Advanced reasoning model for complex problem-solving
   - **O1-mini**: Fast, cost-effective model for specialized tasks

3. **GPT-4o Multimedia Series (Beta)**
   - **GPT-4o-realtime**: Real-time audio and text processing model
   - **GPT-4o-audio-preview**: Specialized audio input/output model

### GPT-4o Overview

**Core Features**
- Most advanced GPT-4 model with enhanced reliability
- Faster processing compared to GPT-4-turbo variant
- Extensive 128,000-token context window
- 16,384-token maximum output capacity

**Performance**
- Superior reliability and consistency in responses
- Enhanced reasoning capabilities across diverse tasks
- Optimized speed for real-time applications
- Balanced efficiency for resource utilization

**Use Cases**
- Complex analysis and problem-solving
- Long-form content generation
- Detailed technical documentation
- Advanced code generation and review

**Technical Specifications**
- Latest GPT architecture optimizations
- Improved response accuracy
- Built-in safety measures
- Enhanced context retention

For more detailed information, please refer to [OpenAI's official documentation](https://platform.openai.com/docs/models#models-overview).

## Meta - Llama Series

Meta's Llama AI series offers open-source models that allow fine-tuning, distillation, and flexible deployment.

### Model Variants

1. **Llama 3.1 (Multilingual)**
   - **8B**: Light-weight, ultra-fast model for mobile and edge devices
   - **405B**: Flagship foundation model for diverse use cases

2. **Llama 3.2 (Lightweight and Multimodal)**
   - **1B and 3B**: Efficient models for on-device processing
   - **11B and 90B**: Multimodal models with high-resolution image reasoning

3. **Llama 3.3 (Multilingual)**
   - **70B**: Multilingual support with enhanced performance

### Llama 3.3 Overview

**Safety Features**
- Incorporates alignment techniques for safe responses

**Performance**
- Comparable to larger models with fewer resources

**Efficiency**
- Optimized for common GPUs, reducing hardware needs

**Language Support**
- Supports eight languages, including English and Spanish

**Training**
- Pre-trained on 15 trillion tokens
- Fine-tuned through Supervised Fine-tuning (SFT) and RLHF

   > **Supervised Fine-tuning** : Supervised fine-tuning is a process of improving an existing AI model's performance by training it with labeled data. For example, if you want to teach the model text summarization, you provide pairs of 'original text' and 'summarized text' as training data. Through this training with correct answer pairs, the model can enhance its performance on specific tasks.
   >
   > **Reinforcement Learning with Human Feedback (RLHF)** : RLHF is a method where AI models learn to generate better responses through human feedback. When the AI generates responses, humans evaluate them, and the model improves based on these evaluations. Just like a student improves their skills through teacher feedback, AI develops to provide more ethical and helpful responses through human feedback.
   
**Use Cases**  

For more detailed information, please refer to [Meta's official documentation](https://www.llama.com/).


## Anthropic - Claude Series

Claude models by Anthropic are advanced language models with cloud-based APIs for diverse NLP tasks. These models balance performance, safety, and real-time responsiveness.

### Model Variants

1. **Claude 3 Series (Flagship Models)**
   - **Claude 3 Haiku**: Near-instant responsiveness
   - **Claude 3 Sonnet**: Balanced intelligence and speed
   - **Claude 3 Opus**: Strong performance for complex tasks

2. **Claude 3.5 Series (Enhanced Models)**
   - **Claude 3.5 Haiku**: Enhanced real-time responses
   - **Claude 3.5 Sonnet**: Advanced research and analysis capabilities

### Claude 3 Opus Overview

**Core Features**
- Handles highly complex tasks such as math and coding
- Extensive context window for detailed document processing

**Performance**
- Superior reliability and consistency
- Optimized for real-time applications

**Use Cases**
- Long-form content generation
- Detailed technical documentation
- Advanced code generation and review

For more detailed information, please refer to [Anthropic's official documentation](https://docs.anthropic.com/en/docs/intro-to-claude).


## Google - Gemini

Google's Gemini models prioritize efficiency and scalability, designed for a wide range of advanced applications.

### Model Variants

1. **Gemini 1.5 Flash**: Offers a 1 million-token context window  
2. **Gemini 1.5 Pro**: Offers a 2 million-token context window  
3. **Gemini 2.0 Flash (Experimental)**: Next-generation model with enhanced speed and performance  

### Gemini 2.0 Flash Overview

**Core Features**
- Supports multimodal live APIs for real-time vision and audio streaming applications  
- Enhanced spatial understanding and native image generation capabilities  
- Integrated tool usage and improved agent functionalities  

**Performance**
- Provides faster speeds and improved performance compared to previous models  

**Use Cases**
- Real-time streaming applications  
- Reasoning tasks for complex problem-solving  
- Image and text generation  

For more detailed information, refer to [Google's Gemini documentation](https://ai.google.dev/gemini-api/docs/models/gemini).

## Mistral AI Models Overview

Mistral AI provides commercial and open-source models for diverse NLP tasks, including specialized solutions.

### Model Variants

**Commercial Models**
- Mistral Large 24.11: Multilingual with a 128k context window
- Codestral: Coding specialist with 80+ language support
- Ministral Series: Lightweight models for low-latency applications

**Open Source Models**
- Mathstral: Mathematics-focused
- Codestral Mamba: 256k context for coding tasks

For more detailed information, please refer to [Mistral's official documentation](https://mistral.ai/technology/#models).


## Alibaba - Qwen

Alibaba’s Qwen models offer open-source and commercial variants optimized for diverse industries and tasks.

### Model Variants

1. **Qwen 2.5**: Advanced multilingual model
2. **Qwen-VL**: Multimodal text and image capabilities
3. **Qwen-Audio**: Specialized in audio transcription and analysis
4. **Qwen-Coder**: Optimized for coding tasks
5. **Qwen-Math**: Designed for advanced math problem-solving

### Key Features

- Leading performance on various benchmarks
- Easy deployment with Alibaba Cloud’s platform
- Applications in generative AI, such as writing, image generation, and audio analysis

For more detailed information, visit [Alibaba Cloud’s official Qwen page](https://mistral.ai/technology/#models).
