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

# Heuristic Evaluation

- Author: [Sunworl Kim](https://github.com/sunworl)
- Design:
- Peer Review:
- Proofread:
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

## Overview

Heuristic evaluation is a quick and simple method of inference when insufficient time or information makes it impossible to make a perfectly reasonable judgment.

This tutorial describes a heuristic evaluation for text generation and search augmentation generation (RAG) systems, and helps to understand through examples.

The main components covered include:

1. For work tokenization, tokenization is performed using NLTK's main token function.

2. Perform a heuristic evaluation based on Rouge, BLEU, METOR, and SemScore.

    - ROUGE : Used to evaluate the quality of automatic summaries and machine translations.
    - BLEU : Mainly used for machine translation evaluation. Measures how similar the generated text is to the reference text.
    - METEOR : An evaluation index developed to evaluate the quality of machine translation.
    - SemScore : Compares model outputs with gold standard responses using Semantic Textual Similarity (STS).

This guide is designed to help developers and researchers implement and understand these evaluation metrics for assessing the quality of text generation systems, particularly in the context of RAG applications.

### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Heuristic Evaluation Based on Rouge, BLEU, METOR, SemScore](#heuristic-evaluation-based-on-rouge-bleu-metor-semscore)
- [Function Definition for RAG Performance Testing](#function-definition-for-rag-performance-testing)
- [Word Tokenization Using NLTK](#word-tokenization-using-nltk)
- [ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Score](#rouge-recall-oriented-understudy-for-gisting-evaluation-score)
- [BLEU (Bilingual Evaluation Understudy) Score](#bleu-bilingual-evaluation-understudy-score)
- [METEOR (Metric for Evaluation of Translation with Explicit Ordering) Score](#meteormetric-for-evaluation-of-translation-with-explicit-ordering-score)
- [SemScore](#semscore)


### References

- [Langchain docs : Scoring Evaluator](https://python.langchain.com/v0.1/docs/guides/productionization/evaluation/string/scoring_eval_chain/)
- [LangSmith docs : Evaluate a RAG application](https://docs.smith.langchain.com/evaluation/tutorials/rag)
- [LangSmith docs : Evaluation concepts](https://docs.smith.langchain.com/evaluation/concepts)
----

## Environment Setup

Set up the environment. You may refer to [Environment Setup](https://wikidocs.net/257836) for more details.

**[Note]**
- `langchain-opentutorial` is a package that provides a set of easy-to-use environment setup, useful functions and utilities for tutorials. 
- You can checkout the [`langchain-opentutorial`](https://github.com/LangChain-OpenTutorial/langchain-opentutorial-pypi) for more details.

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
        "nltk",
        "sentence_transformers",
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
        "LANGCHAIN_PROJECT": "Heuristic-Evaluation"  
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

You can alternatively set API keys such as `OPENAI_API_KEY` in a `.env` file and load them.

[Note] This is not necessary if you've already set the required API keys in previous steps.

```python
# Load API keys from .env file
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Heuristic Evaluation Based on Rouge, BLEU, METOR, SemScore

Heuristic evaluation is a reasoning method that can be used quickly and easily when perfect rational judgment is not possible due to insufficient time or information.

(This also has the advantage of saving time and costs when using LLM as Judge.)

## Function Definition for RAG Performance Testing

Let's create a RAG system for testing.

```python
from myrag import PDFRAG
from langchain_openai import ChatOpenAI

# Create PDFRAG object

rag = PDFRAG(
    "data/Newwhitepaper_Agents2.pdf",
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# Create retriever
retriever = rag.create_retriever()

# Create chain
chain = rag.create_chain(retriever)

# Generate answer to question
chain.invoke("List up the name of the authors")
```




<pre class="custom">'The authors are Julia Wiesinger, Patrick Marlow, and Vladimir Vuskovic.'</pre>



Create a function named `ask_question`. It takes a dictionary named `inputs` as input and returns a dictionary named `answer`.


```python
# Create a function to answer questions
def ask_question(inputs: dict):
    return {"answer": chain.invoke(inputs["question"])}
```

## Word Tokenization Using NLTK 

Word tokenization is the process of splitting text into individual words or tokens. NLTK (Natural Language Toolkit) provides a robust word tokenization functionality through its word_tokenize function.
Main functions of morphological analyzer:

- `word_tokenize` : NLTK's main tokenization function
- `nltk.download('punkt')` : Downloads required tokenization models
- `split()` : Python's basic string splitting
- `word_tokenize()` : NLTK's advanced tokenization

```python
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (run once)
nltk.download('punkt_tab', quiet=True)

sent1 = "Hello. Nice meet you! My name is Chloe~~."
sent2 = "Hello. My name is Chloe~^^. Nice meet you!!"

# Basic string split
print(sent1.split())
print(sent2.split())

print("===" * 20)

# NLTK word tokenization
print(word_tokenize(sent1))
print(word_tokenize(sent2))
```

<pre class="custom">['Hello.', 'Nice', 'meet', 'you!', 'My', 'name', 'is', 'Chloe~~.']
    ['Hello.', 'My', 'name', 'is', 'Chloe~^^.', 'Nice', 'meet', 'you!!']
    ============================================================
    ['Hello', '.', 'Nice', 'meet', 'you', '!', 'My', 'name', 'is', 'Chloe~~', '.']
    ['Hello', '.', 'My', 'name', 'is', 'Chloe~^^', '.', 'Nice', 'meet', 'you', '!', '!']
</pre>

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Score

- It is an evaluation metric used to assess the quality of automatic summarization and machine translation.
- Measures how many important keywords from the reference text are included in the generated text.
- Calculated based on n-gram overlap.



  > Note: What is N-gram?

  ![image.png](assets/08-langsmith-heuristic-evaluation-01.png)



**Rouge-1**
- Measures the similarity at the word level.
- Evaluates the individual word matches between two sentences.

**Rouge-2**
- Measures the similarity based on overlapping consecutive word pairs (bigrams).
- Evaluates the matches of consecutive two-word pairs between two sentences.
  
**Rouge-L**
- Measures the similarity based on the Longest Common Subsequence (LCS).
- Evaluates the word order at the sentence level without requiring continuous matches.
- More flexible and ROUGE-N as it can capture longer distance word relationships.
- Naturally reflects sentence structure similarity by finding the longest sequence that preserves word order but allows gaps.


**Prectical Example**

Example sentences
- Original Sentence : "I met a cute dog while jogging in the park this morning."
- Generated Sentence : "I saw a little cute cat while taking a walk in the park this morning."

1. ROUGE-1 Analysis
   - Each word is compared individually.
   - Matching words : "I", "a", "cute", "while", "in", "the", "park", "this", "morning"
   - These words appear in both sentences, so they are reflected in the score.


2. ROUGE-2 Analysis
   - Compares sequences of consecutive word pairs.
   - Matching phrases : "in the", "the park", "park this", "this morning"
   - These two-word combinations appear in both sentences, so they are reflected in the score.


3. ROUGE-L Analysis
   - Finds the longest common subsequence while maintaining word order.
   - Longest common subsequence : "I a cute while in the park this morning"
   - These words appear in the same order in both sentences, so this sequence is reflected in the ROUGE-L score.


This example demonstrates how each ROUGE metric captures different aspects of similarity:

- **ROUGE-1** captures basic content overlap through individual word matches.
- **ROUGE-2** identifies common phrases and local word order.
- **ROUGE-L** evaluates overall sentence structure while allowing for gaps between matched words.

```python
from rouge_score import rouge_scorer

sent1 = "I met a cute dog while jogging in the park this morning."
sent2 = "I saw a little cute cat while taking a walk in the park this morning."
sent3 = "I saw a little and cute dog on the park bench this morning."


# Define custom tokenizer class
class NLTKTokenizer:
    def tokenize(self, text):
        return word_tokenize(text.lower())


# Initialize RougeScorer with the NLTK tokenizer class
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], 
    use_stemmer=False, 
    tokenizer=NLTKTokenizer()
)

# Compare first pair of sentences
print(
    f"[1] {sent1}\n[2] {sent2}\n[rouge1] {scorer.score(sent1, sent2)['rouge1'].fmeasure:.5f}\n[rouge2] {scorer.score(sent1, sent2)['rouge2'].fmeasure:.5f}\n[rougeL] {scorer.score(sent1, sent2)['rougeL'].fmeasure:.5f}"
)
print("===" * 20)

# Compare second pair of sentences
print(
    f"[1] {sent1}\n[2] {sent3}\n[rouge1] {scorer.score(sent1, sent3)['rouge1'].fmeasure:.5f}\n[rouge2] {scorer.score(sent1, sent2)['rouge2'].fmeasure:.5f}\n[rougeL] {scorer.score(sent1, sent3)['rougeL'].fmeasure:.5f}"
)
```

<pre class="custom">[1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little cute cat while taking a walk in the park this morning.
    [rouge1] 0.68966
    [rouge2] 0.37037
    [rougeL] 0.68966
    ============================================================
    [1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little and cute dog on the park bench this morning.
    [rouge1] 0.66667
    [rouge2] 0.37037
    [rougeL] 0.66667
</pre>

## BLEU (Bilingual Evaluation Understudy) Score

BLEU is a metric used to evaluate text generation quality, particularly in machine translation. It measures the similarity between generated text and reference text by comparing the overlap of word sequences (n-grams).

### Key Features of BLEU

1. N-gram Precision
    - BLEU calculates precision from 1-gram (individual words) to 4-gram (sequences of 4 words)
    - The precision measures how many n-grams in the generated text match those in the reference text
    - Higher n-gram matches indicate better phrase-level similarity

2. Brevity Penalty
    - Imposes a penalty if the generated text is shorter than the reference text.
    - This prevents the system from achieving high precision by only generating short sentences.

3. Geometric Mean
    - The final BLEU score is the geometric mean of the n-gram precisions multiplied by the brevity penalty.
    - Results in a score between 0 and 1

### Example Anaysis

- Original Sentence : "I met a cute dog while jogging in the park this morning."
- Generated Sentence : "I saw a little cute cat while taking a walk in the park this morning."


1. 1-gram(Unigram) Analysis

    - Matching words: "I", "a", "cute", "in", "the", "park", "this", "morning"
    - Precision : 8 / 15 ≈ 0.5333

2. 2-gram(Bigram) Analysis
    - Matching pairs: "in the", "the park", "this morning"
    - Precision : 3 / 14 ≈ 0.2143

3. 3-gram(Trigram) Analysis
    - Matching sequences: "in the park", "the park this", "park this morning"
    - Precision : 3 / 13 ≈ 0.2308

4. 4-gram Analysis
    - Matching sequences: "in the park this", "the park this morning"
    - Precision : 2 / 12 ≈ 0.1667

4. Brevity Penalty
    - Since the lengths of the two sentences are similar, there is no penalty (1.0).

5. Final BLEU Score
    - Geometric mean (0.5333, 0.2143, 0.2308, 0.1667) * 1.0
    - The final BLEU score is 0.2531 or about 25.31%.

### Limitations

- Only checks for simple string matches without considering meaning.
- Does not distinguish the importance of words.

BLEU scores range from 0 to 1, with scores closer to 1 indicating higher quality. However, achieving a perfect score of 1 is very difficult in practice.

```python
from nltk.translate.bleu_score import sentence_bleu


sent1 = "I met a cute dog while jogging in the park this morning."
sent2 = "I saw a little cute cat while taking a walk in the park this morning."
sent3 = "I saw a little and cute dog on the park bench this morning."

# tokenization
reference = word_tokenize(sent1)
candidate1 = word_tokenize(sent2)
candidate2 = word_tokenize(sent3)

print("Sentence 1 tokens:", reference)
print("Sentence 2 tokens:", candidate1)
print("Sentence 3 tokens:", candidate2)
```

<pre class="custom">Sentence 1 tokens: ['I', 'met', 'a', 'cute', 'dog', 'while', 'jogging', 'in', 'the', 'park', 'this', 'morning', '.']
    Sentence 2 tokens: ['I', 'saw', 'a', 'little', 'cute', 'cat', 'while', 'taking', 'a', 'walk', 'in', 'the', 'park', 'this', 'morning', '.']
    Sentence 3 tokens: ['I', 'saw', 'a', 'little', 'and', 'cute', 'dog', 'on', 'the', 'park', 'bench', 'this', 'morning', '.']
</pre>

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Initialize smoothing function
smoothie = SmoothingFunction().method1

# Calculate and print BLEU score for first pair
bleu_score = sentence_bleu(
    [word_tokenize(sent1)],
    word_tokenize(sent2),
    smoothing_function=smoothie
)
print(f"[1] {sent1}\n[2] {sent2}\n[score] {bleu_score:.5f}")
print("===" * 20)

# Calculate and print BLEU score for second pair
bleu_score = sentence_bleu(
    [word_tokenize(sent1)],
    word_tokenize(sent3),
    smoothing_function=smoothie
)
print(f"[1] {sent1}\n[2] {sent3}\n[score] {bleu_score:.5f}")
```

<pre class="custom">[1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little cute cat while taking a walk in the park this morning.
    [score] 0.34235
    ============================================================
    [1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little and cute dog on the park bench this morning.
    [score] 0.11064
</pre>

## METEOR(Metric for Evaluation of Translation with Explicit Ordering) Score

A metric developed to evaluate the quality of machine translation and text generation.

### Key Features

1. Word Matching
   - Exact Matching : Identical words
   - Stem Matching : Words with the same root (e.g., "run" and "running")
   - Synonym Matching : Words with the same meaning (e.g., "quick" and "fast")
   - Paraphrase Matching : Phrase-level synonyms (commonly used in machine translation)

2. Precision and Recall Analysis
   - Precision : Proportion of words in the generated text that match the reference text
   - Recall : Proportion of words in the reference text that match the generated text
   - F-mean : Harmonic mean of precision and recall

3. Order Penalty
   - Evaluates word order similarity between texts.
   - Applies penalties for non-consecutive matches.
   - Ensures fluency and natural word ordering.

4. Weighted Evaluation
   - Assigns different weights to match types (exact, stem, synonym, paraphrase).
   - Allows customization based on evaluation needs.

### METEOR Score Calculation Process

1. Word Matching : Find all possible matches between texts.
2. Precision and Recall : Calculate based on matched words.
3. F-mean : Compute harmonic mean of precision and recall.
4. Order Penalty : Assess word order differences.
5. Final Score : F-mean × (1 - Order Penalty).

### Example
- Reference : "The cat is on the mat"
- Generated : "On the mat is a cat"

1. Word Matching : All content words match("the", "cat", "is", "on", "mat")
2. Precision & Recall = 1.0 (all words match)
3. F-mean = 1.0
4. Order Penalty : 0.1 (due to different word ordering)
5. Final METEOR Score = 1 * (1 - 0.1) = 0.9

### Advantages of METEOR

1. Recognizes synonyms and word variations.
2. Balances precision and recall.
3. Considers word order importance.
4. Effective with single reference text.
5. Correlates well with human judgment.

### METEOR vs BLEU vs ROUGE

- METEOR allows for more flexible evaluation by considering semantic similarity of words.
- It tends to match human judgment better than BLEU.
- Unlike ROUGE, we explicitly consider word order.
- METEOR can be more complicated and time consuming to calculate.

```python
import nltk
import warnings
from nltk.corpus import wordnet as wn

# Suppress NLTK download messages
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    nltk.download('wordnet', quiet=True)

# Import and ensure WordNet is loaded
wn.ensure_loaded()
```

```python
import nltk
import warnings
from nltk.translate import meteor_score

# Suppress NLTK download messages
with warnings.catch_warnings():
   warnings.filterwarnings("ignore")
   nltk.download('punkt_tab', quiet=True)

sent1 = "I met a cute dog while jogging in the park this morning."
sent2 = "I saw a little cute cat while taking a walk in the park this morning."
sent3 = "I saw a little and cute dog on the park bench this morning."

# Calculate METEOR score for first pair
meteor = meteor_score.meteor_score(
   [word_tokenize(sent1)],
   word_tokenize(sent2),
)

print(f"[1] {sent1}\n[2] {sent2}\n[score] {meteor:.5f}")
print("===" * 20)

# Calculate METEOR score for second pair
meteor = meteor_score.meteor_score(
   [word_tokenize(sent1)],
   word_tokenize(sent3),
)
print(f"[1] {sent1}\n[2] {sent3}\n[score] {meteor:.5f}")
```

<pre class="custom">[1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little cute cat while taking a walk in the park this morning.
    [score] 0.70489
    ============================================================
    [1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little and cute dog on the park bench this morning.
    [score] 0.62812
</pre>

## SemScore

- [SEMSCORE: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity](https://arxiv.org/pdf/2401.17072)

This research introduces SemScore, an efficient evaluation metric that uses semantic textual similarity (STS) to compare model outputs with reference responses. 

After evaluating 12 major instruction-tuned LLMs using 8 common text generation metrics, SemScore demonstrated the highest correlation with human evaluation.


### Key Features of SemScore

1. Semantic Textual Similarity (STS)
   - Measures the semantic similarity between the generated text and the reference text.
   - Considers the overall meaning of the sentences beyond simple word matching.

2. Utilization of Pre-trained Language Models
   - Uses pre-trained language models such as BERT or RoBERTa to generate sentence embeddings.
   - This allows for better capture of context and meaning.

3. Multiple Reference Handling
   - Can consider multiple reference answers.
   - This is particularly useful for open-ended questions or creative tasks.

4. Granular Evaluation
   - Evaluates not only the entire response but also parts of the response (e.g., sentence-level).

5. High Correlation with Human Evaluation
   - SemScore shows a high correlation with human evaluators' judgments.


### Calculation Process

1. Text Embedding Generation
   - Converts the generated text and reference text into vectors using pre-trained language models.

2. Similarity Computation
   - Calculates the cosine similarity between the embeddings of the generated text and the reference text.

3. Selection of Maximum Similarity
   - If there are multiple references, selects the highest similarity score.

4. Normalization
   - Normalizes the final score to a value between 0 and 1.

### Advantages of SemScore

1. Semantic Understanding
   - Considers the meaning of sentences beyond surface-level word matches.

2. Flexibility
   - Allows for various forms of answers, making it suitable for creative tasks or open-ended questions.

3. Context Consideration
   - Uses pre-trained language models to better understand the context of words and sentences.

4. Multilingual Support
   - Can evaluate multiple languages using multilingual models.

### SemScore vs Other Evaluation Metrics

- Unlike BLEU and ROUGE, it does not rely solely on simple n-gram matching.
- Measures more advanced semantic similarity compared to METEOR.
- Similar to BERTScore but specialized for instruction-based tasks.

Uses the `SentenceTransformer` model to generate sentence embeddings and calculates the cosine similarity between two sentences.
- The model used in the paper is `all-mpnet-base-v2`.



```python
from sentence_transformers import SentenceTransformer, util
import warnings
import logging
from tqdm import tqdm

# Suppress all warnings and logging messages
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)  # Suppress all warning types

# Disable tqdm warnings
import tqdm.autonotebook
tqdm.autonotebook.tqdm = tqdm.std.tqdm

# Configure logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)

sent1 = "I met a cute dog while jogging in the park this morning."
sent2 = "I saw a little cute cat while taking a walk in the park this morning."
sent3 = "I saw a little and cute dog on the park bench this morning."

# Load SentenceTransformer model silently
model = SentenceTransformer("all-mpnet-base-v2", use_auth_token=False)

# Encoding sentences
sent1_encoded = model.encode(sent1, convert_to_tensor=True, show_progress_bar=False)
sent2_encoded = model.encode(sent2, convert_to_tensor=True, show_progress_bar=False)
sent3_encoded = model.encode(sent3, convert_to_tensor=True, show_progress_bar=False)

# Calculate and print cosine similarity for first pair
cosine_similarity = util.pytorch_cos_sim(sent1_encoded, sent2_encoded).item()
print(f"[1] {sent1}\n[2] {sent2}\n[score] {cosine_similarity:.5f}")

print("===" * 20)

# Calculate and print cosine similarity for second pair
cosine_similarity = util.pytorch_cos_sim(sent1_encoded, sent3_encoded).item()
print(f"[1] {sent1}\n[2] {sent3}\n[score] {cosine_similarity:.5f}")

```

<pre class="custom">[1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little cute cat while taking a walk in the park this morning.
    [score] 0.69124
    ============================================================
    [1] I met a cute dog while jogging in the park this morning.
    [2] I saw a little and cute dog on the park bench this morning.
    [score] 0.76015
</pre>

The evaluator summarized above is as follows.


```python
from langsmith.schemas import Run, Example
import os

# Set tokenizer parallelization for HuggingFace models
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

class NLTKWrapper:
    @staticmethod
    def tokenize(text):
        return word_tokenize(text)
   

def rouge_evaluator(metric: str = "rouge1") -> dict:
    # Define wrapper function
    def _rouge_evaluator(run: Run, example: Example) -> dict:

        # Get generated output and reference answer
        student_answer = run.outputs.get("answer", "")
        reference_answer = example.outputs.get("answer", "")

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], 
            use_stemmer=True, 
            tokenizer=NLTKWrapper()
        )
        scores = scorer.score(reference_answer, student_answer)

        # Return ROUGE score
        rouge = scores[metric].fmeasure
        return {"key": "ROUGE", "score": rouge}
    return _rouge_evaluator


def bleu_evaluator(run: Run, example: Example) -> dict:
    # Get generated output and reference answer
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # Tokenization
    reference_tokens = word_tokenize(reference_answer)
    student_tokens = word_tokenize(student_answer)

    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], student_tokens)

    return {"key": "BLEU", "score": bleu_score}


def meteor_evaluator(run: Run, example: Example) -> dict:
    # Get generated output and reference answer
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # Tokenization
    reference_tokens = word_tokenize(reference_answer)
    student_tokens = word_tokenize(student_answer)

    # Calculate METEOR score
    meteor = meteor_score.meteor_score([reference_tokens], student_tokens)
    return {"key": "METEOR", "score": meteor}


def semscore_evaluator(run: Run, example: Example) -> dict:
    # Get generated output and reference answer
    student_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("answer", "")

    # Load SentenceTransformer model
    model = SentenceTransformer("all-mpnet-base-v2")

    # Generate sentence embeddings
    student_embedding = model.encode(student_answer, convert_to_tensor=True)
    reference_embedding = model.encode(reference_answer, convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_similarity = util.pytorch_cos_sim(
        student_embedding, reference_embedding
    ).item()

    return {"key": "sem_score", "score": cosine_similarity}
```

The evaluation is conducted using the Heuristic Evaluator.

```python
from langsmith.evaluation import evaluate
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Define Evaluator
heuristic_evalulators = [
    rouge_evaluator(metric="rougeL"),
    bleu_evaluator,
    meteor_evaluator,
    semscore_evaluator,
]

# Set Dataset Name
dataset_name = "RAG_EVAL_DATASET"


experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=heuristic_evalulators,
    experiment_prefix="Heuristic-EVAL",
    # Define Experimental Metadata
    metadata={
        "variant": "Evaluater Using Heuristic-EVAL (Rouge, BLEU, METEOR, SemScore)",
    },
)
```

<pre class="custom">View the evaluation results for experiment: 'Heuristic-EVAL-201c2ddf' at:
    https://smith.langchain.com/o/97d4ef95-7b86-4c82-9f4c-3f18e315c9b2/datasets/920886b5-0aa2-4f47-b23f-b3dfc33135ef/compare?selectedSessions=048bce87-ae35-4fd2-af16-6c738fc93762
    
    
</pre>


    0it [00:00, ?it/s]


Check the results.


  ![image.png](assets/08-langsmith-heuristic-evaluation-02.png)
