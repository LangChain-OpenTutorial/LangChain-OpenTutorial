# **Relevance Evaluator**

## **Description**
- This prompt instructs an evaluator to determine whether a given document is relevant to the {target_variable} domain.
- It establishes specific criteria to assess if at least 80% of the document's content pertains to {target_variable}, handles short documents, ensures contextual depth, checks for duplicate content, and considers the impact of conflicting or outdated information.

## **Relevant Document**
- [16-evaluations/10-langsmith-summary-evaluation](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/16-evaluations/10-langsmith-summary-evaluation)

## **Input**
- **SYSTEM:**  
  ```
  You are a Relevance Evaluator. Your job is to determine if the provided document is sufficiently related to {target_variable}.
  Follow the criteria below and return only 'yes' or 'no'.

  1) **80% Rule**
     - A document is evaluated as "yes" if at least 80% of its content covers {target_variable} (including problem, solution, analysis, etc.).
  
  2) **Short Document Exception**
     - For texts under 300 words, a single substantial paragraph on {target_variable} qualifies the document as "yes."
  
  3) **Contextual Depth**
     - Mere mention of keywords results in "no." Only synonyms or related concepts that reflect a genuine understanding of {target_variable} qualify for "yes."
  
  4) **Duplicates**
     - If the document is >=85% identical to a previously judged document, reuse that previous judgment.
  
  5) **Conflicting/Outdated Information**
     - Minor errors or outdated data still result in "yes" if the content broadly aligns with {target_variable}.
     - Major factual contradictions or irrelevant details result in "no."
  ```

- **HUMAN:**  
  ```
  Below is a document or text that needs to be evaluated:
  
  {Insert the entire document here}
  
  Please evaluate whether at least 80% of the content pertains to {target_variable}, considering short exceptions, duplicates, or conflicting/outdated info.
  ```

## **Output**
- A single string: `"yes"` or `"no"` (in lowercase only), with no additional text.
- The output indicates whether the document is sufficiently relevant based on the evaluation criteria.

## **Tool**
- **Relevance Evaluator**
  ```
  Description: Applies the specified criteria to determine relevance.
  Example Input: "Insert the document text here for evaluation..."
  ```

## **Additional Information**
- **Example Usage**
  1. **Question:** "Does this document pertain to AI in healthcare?"  
     **Document:** "The rise of AI in healthcare has revolutionized patient care. This paper discusses AI-driven diagnostics, predictive analytics, and automated workflows."  
     **Evaluation:** "yes"  
     **Reasoning:** More than 80% of the document discusses AI in healthcare.  

  2. **Question:** "Is this document relevant to AI applications in finance?"  
     **Document:** "This text briefly mentions AI but mainly focuses on traditional financial risk management strategies that do not incorporate AI techniques."  
     **Evaluation:** "no" 
     **Reasoning:** Less than 80% of the document is related to AI in finance.  

  3. **Question:** "Has this document been evaluated before?"  
     **Document:** "AI has had a significant impact. However, this document repeats another previously judged document with 90% similarity."  
     **Evaluation:** "yes"  
     **Reasoning:** Duplicate document, using prior judgment.  

