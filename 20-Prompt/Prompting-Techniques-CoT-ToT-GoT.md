# **Chain-of-Thought**, **Tree of Thoughts**, and **Graph of Thoughts**

## **Description**
- *Chain-of-Thought (CoT)* is a prompting technique that generates a series of intermediate steps for solving reasoning tasks. [^1]
- *Tree of Thoughts (ToT)* is a prompting technique that extends the *Chain-of-Thought* prompting. Instead of using a linear reasoning path, it uses a tree structure to explore multiple different reasoning paths. [^2]
- *Graph of Thoughts (GoT)* is a prompting technique extended from the *Chain-of-Thought* prompting and *Tree of Thoughts* prompting. By modeling the reasoning process as a graph structure, a more complex network of thoughts can be generated. [^3]

## Example Prompts from Relevant Papers
### Example 1: *Few-shot CoT* for Math Word Problems [^1]
```
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
Q: {question}
```

### Example 2: *Zero-shot CoT* [^4]
```
Q: {question}
A: Let's think step by step.
```

### Example 3: *ToT* for Creative Writing [^2] [^5]
- cot_prompt
    ```
    Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}

    Make a plan then write. Your output should be of the following format:

    Plan:
    Your plan here.

    Passage:
    Your passage here.
    ```
- vote_prompt
    ```
    Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
    ```
- compare_prompt
    ```
    Briefly analyze the coherency of the following two passages. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".
    ```
- score_prompt
    ```
    Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.
    ```
            
### Example 4: *GoT* in Document Merging [^3] [^6]
- merge prompt
    ```
    Merge the following 4 NDA documents <Doc1> - <Doc4> into a single NDA, maximizing retained information and minimizing redundancy. Output only the created NDA between the tags <Merged> and </Merged>, without any additional text.
    Here are NDAs <Doc1> - <Doc4>:
    <Doc1> {doc1} </Doc1>
    <Doc2> {doc2} </Doc2>
    <Doc3> {doc3} </Doc3>
    <Doc4> {doc4} </Doc4>
    ```

- score prompt: 
    ```
    The following NDA <S> merges NDAs <Doc1> - <Doc4>.
    Please score the merged NDA <S> in terms of how much redundant information is contained, independent of the original NDAs, as well as how much information is retained from the original NDAs.
    A score of 10 for redundancy implies that absolutely no information is redundant, while a score of 0 implies that at least half of the information is redundant (so everything is at least mentioned twice).
    A score of 10 for retained information implies that all information from the original NDAs is retained, while a score of 0 implies that no information is retained.
    You may provide reasoning for your scoring, but the final score for redundancy should be between the tags <Redundancy> and </Redundancy>, and the final score for retained information should be between the tags <Retained> and </Retained>, without any additional text within any of those tags.
    Here are NDAs <Doc1> - <Doc4>:
    <Doc1> {doc1} </Doc1>
    <Doc2> {doc2} </Doc2>
    <Doc3> {doc3} </Doc3>
    <Doc4> {doc4} </Doc4>
    Here is the merged NDA <S>:
    <S> {s} </S>
    ```

- aggregate prompt
    ```
    The following NDAs <S1> - <S{num_ndas_summaries}> each merge the initial NDAs <Doc1> - <Doc4>.
    Combine the merged NDAs <S1> - <S{num_ndas_summaries}> into a new one, maximizing their advantages and overall information retention, while minimizing redundancy.
    Output only the new NDA between the tags <Merged> and </Merged>, without any additional text.
    Here are the original NDAs <Doc1> - <Doc4>:
    <Doc1> {doc1} </Doc1>
    <Doc2> {doc2} </Doc2>
    <Doc3> {doc3} </Doc3>
    <Doc4> {doc4} </Doc4>
    Here are the merged NDAs <S1> - <S{num_ndas_summaries}>:
    <S1> {s1} </S1>
    ...
    <S{num_ndas_summaries}> {s{num_ndas_summaries}} </S{num_ndas_summaries}>
    ```

- improve prompt
    ```
    The following NDA <S> merges initial NDAs <Doc1> - <Doc4>.
    Please improve the merged NDA <S> by adding more information and removing redundancy. Output only the improved
    NDA, placed between the tags <Merged> and </Merged>, without any additional text.
    Here are NDAs <Doc1> - <Doc4>:
    <Doc1> {doc1} </Doc1>
    <Doc2> {doc2} </Doc2>
    <Doc3> {doc3} </Doc3>
    <Doc4> {doc4} </Doc4>
    Here is the merged NDA <S>:
    <S> {s} </S>
    ```

## Applications
### Example 1: CoT applied to a supervisor prompt in hierarchical multi-agent system
- **Related Usage** : [17-LangGraph/03-Use-Cases/08-Hierarchical-Multi-Agent-Teams](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/17-LangGraph/03-Use-Cases/08-Hierarchical-Multi-Agent-Teams.ipynb)

- The current tutorial features a hierarchical multi-agent structure, with a supervising agent in each team and a super-graph. The prompts for all three supervisors are the same, except for the members assigned to each supervisor. By defining members as a variable, the prompts can be shared among all three agents.
- SYSTEM: 
    ```
    You are a supervisor.
    The goal of your team is to write a {report} on the theme requested by the user.
    You are responsible for managing collaboration between your {members}.

    1. First, break down the task into logical steps. Enclose your thought for why each step is necessary within <think></think> tags.
    2. For each step, allocate a suitable worker from your {members}.
    3. Your {members} will respond to their assigned tasks. Evaluate their responses. Are they accurate and relevant?
    4. If any part is incomplete or insufficient, provide feedback and request revisions. Each chapter must have more than 5 sentences.
    5. When the task is fully completed, respond with "FINISH".
    
    Ensure that the final {report} maintains a professional tone and meets high-quality standards.
    ```
- Input variables: ["report", "members"]
    - `report` : Define the type of the report. (e.g. financial report, academic report, etc.)
    - `members` : A list of agents assigned to the supervisor.

### Example 2: ToT applied in math problem solving
- **Related Usage** : [17-LangGraph / 03-Use-Cases / 13-Tree-of-Thoughts](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/03-Use-Cases/13-Tree-of-Thoughts)

- SYSTEM:
    ```
    You are an expert in solving math puzzles.
    You will play a game called `Make 15`.
    The objective of the game is to create a mathematically valid equation that evaluates **exactly** to 15.

    You will be given four distinct numbers (ranging from 1 to 9).  
    Your reasoning process must follow a **tree structure**, branching off from the parent node.  
    At each step, you may **only select two of the remaining numbers** to compute a new number.

    [TREE STRUCTURE]
    1. Start with an empty root node.
    2. At each step, list the **remaining numbers**.
    3. Select **two numbers** from the remaining set and apply **one** of the four basic arithmetic operations (`+`, `-`, `*`, `/`).
    4. Replace the selected numbers with the computed result and **continue branching**.
    5. Each parent node can have up to {k} child nodes, meaning you can explore up to {k} different operations at each step.
    6. Evaluate whether the **final computed result** is **exactly 15**.
    7. Stop when only **one number remains**, and check if it equals **15**.

    [RULES]
    1. You must use **all four** numbers exactly **once**.
    2. Each number must be used **exactly once** in the final equation.
    3. Allowed arithmetic operations - **addition (+), subtraction (-), multiplication (*), and division (/).**
    4. Every number must have **an operator before or after it**.
    5. Use parentheses **explicitly** to clarify precedence when needed.
    6. If the final result is **not** 15, backtrack and explore alternative branches.

    [INPUT FORMAT]
    - The model will receive a list of **four distinct integers** from 1 to 9 (e.g., `[2, 3, 5, 7]`).
    - The model should output a structured breakdown of its **tree-based reasoning**, including:
    - **Each step of the tree**
    - **The selected numbers and the applied operation**
    - **The resulting new number**
    - **The final computed equation**
    - **Whether the equation successfully evaluates to 15**
    ```

- input_variables: ["k"]
    - `k` : Number of child nodes to suggest at each step.

### Example 3: GoT applied to document merging
- Merge_prompt:
    - SYSTEM:
        ```
        Merge the following {num} documents <Doc1> - <Doc{num}> into a single cohesive document.
  
        [GOAL]
            - Maximize information retention to preserve all unique information.
            - Minimize redundancy (avoid repeating similar content).
            - Maintain logical flow and readability.
            - Preserve the context and relationships between ideas.
        
        [IMPORTANT FORMATTING RULES]
            - The final merged document **must be placed between `<Merged>` and `</Merged>` tags.**
            - No extra text before `<Merged>` and no explanations after `</Merged>`.
            ```
            <Merged>
            [Final merged document here]
            </Merged>
            ```
              
        [PROCESS]
        1. Analysis:
            - Break each document into logical subparts.
            - Identify overlapping or complementary information and summarize it compactly.
            - If there is conflicting information, resolve it by prioritizing factual accuracy.
            - Extract all unique content.

        2. Organization:
            - Group related information across documents.
            - Structure the final document in a logical order:

        3. Merging:
            - Combine similar points without losing meaning.
            - The merged content must be shorter than the sum of original documents.

        4. Output:
            - The final document **must** be enclosed in `<Merged>` and `</Merged>` tags.
            - Ensure concise yet comprehensive content.

        5. Evaluation:
            - Compare the merged document with the original sources.
            - Scoring Guidelines:
            - Redundancy Score: Minimize repeated phrases.
            - Retention Score: Ensure all key details are preserved.
            - Readability Score: The final document should be clear and well-structured.
            - Consistency Check: Ensure no conflicting or contradictory statements remain.

        Here are documents <Doc1> - <Doc{num}>:
        {docs}
        ```
    - input_variables: ["num", "docs"]
        - `num` : The number of documents to be merged.
        - `docs` : Contents of the documents to be merged.

- Analyze_prompt:
    - SYSTEM:
        ```
        You are analyzing a merged document for redundancy and information retention.
        Your task is to provide two scores in a specific JSON format.
        
        Guidelines:
        1. Redundancy Score (0.0 to 1.0):
            - 0.0 means no redundant content
            - 1.0 means highly redundant content
            - Score strictly: 0.4+ indicates significant duplicate content
        
        2. Retention Score (0.0 to 1.0):
            - 0.0 means poor information retention
            - 1.0 means perfect information retention
            - Score strictly: below 0.7 means important information was lost
        
        Compare the original documents to the merged document and evaluate:
        - Are there any redundant content left?
        - Is the context and meaning maintained?
        - Is the merged document shorter than the sum of original documents?
        
        Original documents:
        {original_docs}
        
        Merged document:
        {merged_doc}
        
        Respond ONLY with a JSON object in this exact format:
        {{"redundancy_score": 0.XX, "retention_score": 0.XX}}
        ```
    - input_variables: ["original_docs", "merged_doc"]
        - `original_docs` : The contents of the original documents.
        - `merged_doc` : The merged document generated from the process.

## **Reference**
[^1]: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)

[^2]: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601)

[^3]: [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/pdf/2308.09687)

[^4]: [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916)

[^5]: [Github: Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm)

[^6]: [Github: Graph of Thoughts](https://github.com/spcl/graph-of-thoughts)