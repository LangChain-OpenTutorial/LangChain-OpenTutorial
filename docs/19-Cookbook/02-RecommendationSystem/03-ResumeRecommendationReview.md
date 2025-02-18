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

# ResumeRecommendationReview

- Author: [Ilgyun Jeong](https://github.com/johnny9210)
- Design:
- Peer Review: [Jaehun Choi](https://github.com/ash-hun), [Dooil Kwak](https://github.com/back2zion)
- This is a part of [LangChain Open Tutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/02-RecommendationSystem/03-ResumeRecommendationReview.ipynb) [![Open in GitHub](https://img.shields.io/badge/Open%20in%20GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/19-Cookbook/02-RecommendationSystem/03-ResumeRecommendationReview.ipynb)

## Overview

The ResumeRecommendationReview system is a comprehensive solution designed to simplify and enhance the job application process for individuals seeking corporate positions. The system is divided into two main components, each tailored to address key challenges faced by job seekers:

1) Company Recommendation
Using advanced matching algorithms, the system analyzes a user’s uploaded resume and compares it with job postings on LinkedIn. Based on this analysis, it identifies and recommends companies that align closely with the candidate’s qualifications, skills, and career aspirations.

2) Resume Evaluation and Enhancement
For the recommended companies, the system conducts a detailed evaluation of the user’s resume. It highlights strengths, identifies areas for improvement, and provides actionable suggestions for tailoring the resume to better fit the expectations of target roles. This ensures candidates can present their qualifications in the most impactful way possible.

By integrating these two components, the ResumeRecommendationReview system streamlines the job application journey, empowering users to:

- Discover job opportunities that best match their unique profile.
- Optimize their resumes for maximum impact, increasing their chances of securing interviews and job offers.

**Key Features**:

- **CV/Resume Upload**: 
  Users begin by uploading their existing CV or resume in a supported file format (e.g., PDF)
  The system extracts relevant keywords, experiences, and skill sets to build a user profile.

- **Job Matching with LinkedIn Postings**: 
  The platform automatically scans LinkedIn job listings (and potentially other job boards) for roles that align with the user’s skill set and career interests.
  A matching algorithm ranks and recommends a list of the most relevant companies and positions for the candidate to consider.

- **Comparison & Evaluation (LLM-as-a-Judge)**  
  The system leverages a Large Language Model (LLM) to analyze the uploaded resume and specific job requirements. 
  It evaluates the alignment between the user's experience and the job description, identifying strengths,  skill gaps, and areas in need of improvement.
  Additionally, the system evaluates the recommendation performance using **cosine similarity** to measure the semantic alignment and **NDCG (Normalized Discounted Cumulative Gain)** to assess the ranking quality of the recommendations.
  
- **Automated Resume Enhancement**: 
  Based on the LLM evaluation, the system provides a detailed report highlighting sections that need modification.
  Suggested edits may include restructuring experience points, emphasizing relevant skills, or adding keywords that match the job posting’s expectations.


### Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation and Preprocessing](#data-preparation-and-preprocessing)
- [Setting Up ChromaDB and Storing Data](#Setting-Up-ChromaDB-and-Storing-Data)
- [Company Recommendation System](#Company-Recommendation-System)
- [LLM-Based Resume Evaluation System](#LLM-Based-Resume-Evaluation-System)
- [LLM-Based Resume Revise System](#LLM-Based-Resume-Revise-System)

### References



---


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
        "chromadb",
        "langchain_chroma",
        "langchain_openai",
        "PyMuPDF",
        "pydantic",
        "pandas",
        "kagglehub",
        "langchain_community",
        "numpy",
        "ipywidgets",
        "chardet",
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
        "LANGCHAIN_PROJECT": "ResumeRecommendationReview",
        "UPSTAGE_API_KEY": "",
    }
)
```

<pre class="custom">Environment variables have been set successfully.
</pre>

```python
from dotenv import load_dotenv

load_dotenv(override=True)
```




<pre class="custom">True</pre>



## Data Preparation and Preprocessing

This section covers the data preparation and preprocessing steps required for the Resume Recommendation System. The key stages include:

- **Processing resume data (PDF)**  
- **Processing LinkedIn job postings**  

For the LinkedIn job postings data, this tutorial uses the dataset available on Kaggle: [arshkon/linkedin-job-postings](https://www.kaggle.com/arshkon/linkedin-job-postings).  

Using the raw data directly to build the recommendation system may lead to suboptimal performance. Therefore, the data is refined and preprocessed to focus specifically on recruitment-related information to enhance the accuracy and relevance of the recommendations.

Install and Import Required Libraries

```python
# Import required libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import fitz  # PyMuPDF
import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, List
import kagglehub
import json
import ipywidgets
```

Text Splitting Configuration

Set up configurations to divide the extracted text into manageable sizes, ensuring smooth processing:

Parameter Descriptions:
- `chunk_size`: The maximum length of each text chunk, ensuring the text is divided into manageable sections.
- `chunk_overlap`: The length of overlapping text between chunks, providing continuity and context for downstream tasks.
- `separators`: The delimiters used to split the text, such as line breaks or punctuation, to optimize the splitting process.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
)
```

Defining the Pydantic Model

In this section, we define a structured data model using Pydantic, which ensures validation and consistency in the data extracted from resumes. This model is critical for organizing key sections of a resume into a format that the system can analyze effectively.

```python
# Define the Pydantic model
class ResumeSection(BaseModel):
    skills: List[str] = Field(description="List of job-related technical skills")
    work_experience: List[Dict[str, str]] = Field(
        description="Work experience (role, description)"
    )
    projects: List[Dict[str, str]] = Field(
        description="Project experience (name, description)"
    )
    achievements: List[str] = Field(
        description="List of major achievements and accomplishments"
    )
    education: List[Dict[str, str]] = Field(
        description="Education information (name, description)"
    )


# Configure the PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=ResumeSection)
```

Analyzing Interests in Resumes

The `analyze_interests` function is designed to extract and summarize the key areas of interest and research focus from a resume. It uses a **Large Language Model (LLM)** to process the resume text and provide a concise summary, helping to identify the candidate's academic and professional interests effectively.

Purpose
- Extracts **main areas of interest** and **research focus** from the provided resume text.
- Generates a **brief summary** (2-3 sentences) that highlights the candidate's academic and career patterns.
- Focuses solely on interests and research areas to provide targeted insights.

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def analyze_interests(resume_text: str, llm) -> str:
    """Analyzes the complete resume text to identify key interest areas."""
    interest_prompt = """Analysis this resume text and provide a brief summary (2-3 sentences) 
    of the person's main areas of interest and research focus. Focus on their academic interests, 
    research topics, and career patterns.

    Resume Text:
    {text}

    Provide a concise summary focusing ONLY on their interests and research areas."""

    messages = [{"role": "user", "content": interest_prompt.format(text=resume_text)}]
    response = llm.invoke(messages)
    return response.content.strip()
```

Analyzing Career Fit in Resumes

The `analyze_career_fit` function evaluates a candidate's resume to recommend the most suitable job roles along with their respective fit scores. By leveraging a **Large Language Model (LLM)**, this function identifies key areas of expertise and rates the candidate's suitability for various technical positions.

Purpose
- Recommends **job roles** based on the candidate's skills, research background, and career trajectory.
- Assigns a **fit score** (0.0 to 1.0) for each role, reflecting the candidate's alignment with the position.

```python
def analyze_career_fit(resume_text: str, llm) -> Dict[str, float]:
    """Analyzes the resume to recommend suitable job roles and their fit scores."""
    career_prompt = """You are an expert technical recruiter. Analyze this resume and recommend the most suitable job roles.
    Focus on the candidate's expertise, research background, and career trajectory.
    
    Resume Text:
    {text}
    
    Based on their background, rate the candidate's fit (0.0 to 1.0) for different technical roles.
    Consider:
    - Technical expertise and depth
    - Research contributions
    - Project complexity
    - Educational background
    - Career progression
    
    Return ONLY a JSON object with role-fit pairs, like:
    {{"Research Scientist": 0.95, "Machine Learning Engineer": 0.9, "Algorithm Engineer": 0.85}}
    
    Include only roles with fit score > 0.7. Focus on senior/research level positions if appropriate."""

    messages = [{"role": "user", "content": career_prompt.format(text=resume_text)}]
    response = llm.invoke(messages)

    try:
        return json.loads(response.content.strip())
    except json.JSONDecodeError:
        return {}
```

Processing Resumes to Extract Key Job-Related Information

The `process_resume` function analyzes a resume file, extracting and processing key information relevant to job applications. It combines **text extraction**, **interest analysis**, and **career fit evaluation** to generate structured, weighted insights from the resume.

### Function Overview

Purpose
- Extract **key job-related information** from resumes in PDF format.
- Use **LLM analysis** to evaluate the candidate's skills, experience, projects, achievements, and education.
- Assign **weights** to each section based on relevance to the target job role.

```python
def process_resume(file_path, target_job_title=None):
    """Analyze a resume to extract key job-related information.

    Args:
        file_path (str): Path to the resume PDF file
        target_job_title (str, optional): Specific job title to target analysis

    Returns:
        List[Tuple[str, float]]: List of (content chunk, weight) pairs where:
            - content chunk is a section of the resume text
            - weight is the importance score (0-1) assigned to that section
    """
    # Extract text from PDF document
    doc = fitz.open(file_path)
    resume_text = ""
    for page in doc:
        resume_text += page.get_text()

    # Get initial analysis of interests and career fit
    interest_summary = analyze_interests(resume_text, llm)
    career_fit = analyze_career_fit(resume_text, llm)

    prompt_template = """You are a professional resume analyst specializing in research and technical roles.
    Analyze the resume in detail, focusing on the candidate's expertise level and research background.
    
    Target Job Title: {target_job_title}
    
    Resume Content:
    {resume_text}
    
    Extract the information in the following format:
    {format_instructions}
    
    Focus on extracting information most relevant to research and technical roles.
    Pay special attention to:
    - Research contributions and impact
    - Technical depth in each area
    - Project complexity and leadership
    - Academic achievements and specializations"""

    # Create the prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Format the messages
    messages = prompt.format_messages(
        target_job_title=target_job_title if target_job_title else "Not specified",
        resume_text=resume_text,
        format_instructions=parser.get_format_instructions(),
    )

    # Perform LLM analysis
    response = llm.invoke(messages)

    try:
        parsed_sections = parser.parse(response.content)
        print("Resume analysis completed.")
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"LLM response: {response.content}")
        return []

    # Apply weights to different sections based on importance
    weighted_content = []

    # Skills (Weight: 0.25)
    if parsed_sections.skills:
        skills_text = " ".join(parsed_sections.skills)
        weighted_content.append((skills_text, 0.25))

    # Work Experience (Weight: 0.3)
    if parsed_sections.work_experience:
        experience_text = "\n".join(
            [
                f"{exp.get('role', '')}: {exp.get('description', '')}"
                for exp in parsed_sections.work_experience
            ]
        )
        weighted_content.append((experience_text, 0.3))

    # Projects (Weight: 0.2)
    if parsed_sections.projects:
        projects_text = "\n".join(
            [
                f"{proj.get('name', '')}: {proj.get('description', '')}"
                for proj in parsed_sections.projects
            ]
        )
        weighted_content.append((projects_text, 0.2))

    # Achievements (Weight: 0.1)
    if parsed_sections.achievements:
        achievements_text = " ".join(parsed_sections.achievements)
        weighted_content.append((achievements_text, 0.1))

    # Education (Weight: 0.05)
    if parsed_sections.education:
        education_text = "\n".join(
            [
                f"{edu.get('name', '')}: {edu.get('description', '')}"
                for edu in parsed_sections.education
            ]
        )
        weighted_content.append((education_text, 0.05))

    # Add interest summary and career fit (combined weight: 0.1)
    if interest_summary or career_fit:
        analysis_text = (
            "Research Interests and Focus Areas: " + interest_summary + "\n\n"
        )
        if career_fit:
            analysis_text += "Recommended Roles:\n"
            for role, score in sorted(
                career_fit.items(), key=lambda x: x[1], reverse=True
            ):
                analysis_text += f"- {role}: {score:.2f}\n"

        weighted_content.append((analysis_text, 0.1))

        # Adjust other weights to maintain total of 1.0
        weighted_content = [
            (content, weight * 0.9) for content, weight in weighted_content[:-1]
        ] + [weighted_content[-1]]

    # Generate chunks for each section
    processed_chunks = []
    for content, weight in weighted_content:
        if content.strip():  # Process only non-empty strings
            chunks = text_splitter.split_text(content)
            processed_chunks.extend([(chunk, weight) for chunk in chunks])

    print(f"Number of extracted chunks: {len(processed_chunks)}")
    print("\nCareer Analysis Summary:")
    print("------------------------")
    print("Interests:", interest_summary)
    print("\nRecommended Roles:")
    for role, score in sorted(career_fit.items(), key=lambda x: x[1], reverse=True):
        print(f"- {role}: {score:.2f}")

    return processed_chunks
```

Resume Processing Example

Here's an example of how to use the `process_resume` function to extract structured data from a resume:

```python
process_resume("../data/joannadrummond-cv.pdf")
```

<pre class="custom">Resume analysis completed.
    Number of extracted chunks: 7
    
    Career Analysis Summary:
    ------------------------
    Interests: Joanna Drummond's main areas of interest and research focus are in computer science, particularly in algorithms, artificial intelligence, and stable matching problems. Her research has extensively explored topics such as preference elicitation, stable and approximately stable matching, and decision-making under uncertainty, with applications in multi-agent systems and educational technologies. She has also investigated the use of machine learning techniques for classifying student engagement and dialogue transitions in educational settings.
    
    Recommended Roles:
</pre>




    [('Python Java Julia R Matlab Unix Shell Scripting (bash) Linux Mac OSX Windows LATEX Weka',
      0.225),
     ('Research Intern: Microsoft Research, with Ian Kash and Peter Key, May 2016 to August 2016. Investigated simple pricing for cloud computing.\nResearch Assistant: University of Toronto, Department of Computer Science, August 2011 to Present. Investigated Bayes-Nash and ex-post equilibria for matching games with imperfect information, stable and approximately stable matching using multi-attribute preference information, and elicitation schemes using multi-attribute based queries.\nResearch Assistant: University of Pittsburgh Department of Computer Science, April 2008 to May 2011. Investigated the impact of different training set populations on accurately classifying student uncertainty while using a spoken intelligent physics tutor.\nResearch Assistant: DREU Program, Information Sciences Institute, University of Southern California, June 2010 to August 2010. Applied HMM’s and decision trees to students’ online forum data to categorize students’ posts.',
      0.27),
     ('Teaching Assistant: University of Toronto Dept. of Computer Science, September 2011 to Present. Developed assignments, created marking schemes, and marked exams and assignments for an upper-level Intro to AI course.\nTeaching Assistant: University of Pittsburgh Dept. of Mathematics, September 2007 to April 2008. Taught College Algebra Recitation, held office hours, graded homework.\nTutor: University of Pittsburgh Dept. of Mathematics, October 2006 to April 2007. Individual and Group Tutor, Subjects: College Algebra through Calculus III.',
      0.27),
     ('Stable Matching Problems: Investigated stable and approximately stable matching using multi-attribute preference information and elicitation schemes for the stable matching problem, including a scheme that found low interview-cost matchings.\nStudent Online Discussions: Analyzed and proved properties about an algorithm for dividing n indivisible objects among 2 people.',
      0.18000000000000002),
     ('Program Committee, CoopMAS 2017 Microsoft Research PhD Fellowship Program Finalist, 2016 Reviewer, Algorithmica, 2015 Reviewer, SAGT 2015 Reviewer, AAAI-15 Ontario Graduate Scholarship, 2014 Reviewer, COMSOC-2014 Microsoft Research Graduate Women’s Scholarship Recipient, 2012 Google Anita Borg Memorial Scholarship Finalist, 2012 Ontario Graduate Scholarship, 2012 Awardee of 2011 NSF Graduate Research Fellowship Program DREU Recipient, Chosen for Distributed Research Experience for Undergraduates Program Best Undergraduate Poster, University of Pittsburgh Department of Computer Science 10th Annual Computer Science Day',
      0.09000000000000001),
     ('PhD Computer Science: University of Toronto, (expected) Spring 2017. Co-advisors: Allan Borodin, Kate Larson. GPA: 3.83\nM.S. Computer Science: University of Toronto, Spring 2013. Advisor: Craig Boutilier. GPA: 3.93\nB.S. Computer Science and Mathematics: University of Pittsburgh, December 2010. Research Advisor: Diane Litman. Graduated Magna Cum Laude with Departmental Honors. GPA: 3.73',
      0.045000000000000005),
     ("Research Interests and Focus Areas: Joanna Drummond's main areas of interest and research focus are in computer science, particularly in algorithms, artificial intelligence, and stable matching problems. Her research has extensively explored topics such as preference elicitation, stable and approximately stable matching, and decision-making under uncertainty, with applications in multi-agent systems and educational technologies. She has also investigated the use of machine learning techniques for classifying student engagement and dialogue transitions in educational settings.",
      0.1)]



LinkedIn Data Preprocessing

This step involves loading job posting data and extracting only the necessary details. The dataset used for this tutorial is sourced from **Kaggle**: [arshkon/linkedin-job-postings](https://www.kaggle.com/arshkon/linkedin-job-postings).

- `company_name`: The name of the company offering the job posting.
- `title`: The title of the job being offered.
- `description`: A detailed description of the job, including responsibilities, qualifications, and expectations.
- `max_salary`: The maximum salary offered for the position.
- `med_salary`: The median salary for the position, providing an average range for the offered pay.
- `min_salary`: The minimum salary offered for the position.
- `skills_desc`: A list or summary of the required or preferred skills for the position.
- `work_type`: The type of work arrangement, such as full-time, part-time, remote, or hybrid.

Purpose of These Columns
These selected columns are essential for processing job posting data. They allow the system to:

- Extract relevant metadata for recommendation and filtering.
- Match resumes to job postings based on skills, and job details.
- Provide users with clear and actionable job-related information.
---
Efficient CSV Reading with Encoding Detection

This function provides a robust way to read CSV files by dynamically handling encoding issues. CSV files often come in various encodings, and incorrect encoding can cause errors when reading the file. The function attempts to read the file with the most common encodings and falls back to a detection library if necessary.

Function: `read_csv_with_encoding`

 **Purpose**
To efficiently read a CSV file while handling potential encoding issues, ensuring compatibility with a wide range of file formats.

---
**How It Works**
1. **Attempt to Read with UTF-8**:  
   The function first tries to read the file using UTF-8 encoding, which is the most commonly used encoding.
   - If successful, the function returns the loaded DataFrame.
   - If a `UnicodeDecodeError` occurs, it proceeds to the next step.

2. **Encoding Detection with `chardet`**:  
   If UTF-8 fails, the function uses the `chardet` library to detect the file's encoding:
   - Reads the first 10KB of the file for faster detection.
   - Extracts the detected encoding from the result.

3. **Retry with Detected Encoding**:  
   The function attempts to read the file again using the detected encoding:
   - If successful, the DataFrame is returned.
   - If another `UnicodeDecodeError` occurs, it falls back to a common encoding.

4. **Fallback to CP949**:  
   If both UTF-8 and the detected encoding fail, the function defaults to **CP949** encoding, commonly used for files in East Asian languages like Korean.

```python
import chardet


def read_csv_with_encoding(file_path):
    """Efficiently read CSV file with appropriate encoding"""
    # First try UTF-8 as it's most common
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        # If UTF-8 fails, then use chardet for detection
        with open(file_path, "rb") as file:
            raw_data = file.read(10000)  # Only read first 10KB for detection
            result = chardet.detect(raw_data)
            encoding = result["encoding"]

        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to a common encoding if detection fails
            return pd.read_csv(file_path, encoding="cp949")


path = kagglehub.dataset_download("arshkon/linkedin-job-postings", path="postings.csv")
df = read_csv_with_encoding(path)

selected_columns = [
    "company_name",
    "title",
    "description",
    "max_salary",
    "med_salary",
    "min_salary",
    "skills_desc",
    "work_type",
]
linkedin_df = df[selected_columns].copy()
```

Text Cleaning Function

Here’s a utility function designed to clean and preprocess text data for better consistency and quality:

If there are any `null` values in the company name field, those entries are excluded. (While other fields may also have `null` values, this step focuses only on excluding records with `null` in the company name.)

```python
def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", str(text))
    # Remove consecutive whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Remove rows where company_name is empty
linkedin_df = linkedin_df.dropna(subset=["company_name"])
# Alternative using boolean indexing:
# linkedin_df = linkedin_df[linkedin_df['company_name'].notna()]

# Clean text data
linkedin_df["description"] = linkedin_df["description"].apply(clean_text)
linkedin_df["skills_desc"] = linkedin_df["skills_desc"].apply(clean_text)
linkedin_df["title"] = linkedin_df["title"].apply(clean_text)

# Process salary information
for col in ["max_salary", "med_salary", "min_salary"]:
    linkedin_df[col] = pd.to_numeric(linkedin_df[col], errors="coerce")

# Handle missing values
linkedin_df["work_type"] = linkedin_df["work_type"].fillna("Not specified")
```

Processing Job Postings Data

The `process_job_postings` function integrates and processes job information from a LinkedIn dataset to create structured documents for analysis or recommendation purposes.

This function takes a DataFrame of LinkedIn job postings and processes each entry into a standardized format, combining relevant details like company name, job title, required skills, and salary information.

```python
def process_job_postings(linkedin_df):
    """Process and integrate job information"""
    job_documents = []

    # Integrate information for each job
    for _, row in linkedin_df.iterrows():
        # Format salary information
        salary_info = "No salary information"
        if pd.notna(row["min_salary"]) and pd.notna(row["max_salary"]):
            salary_info = f"{row['min_salary']:,.0f} - {row['max_salary']:,.0f}"
        elif pd.notna(row["med_salary"]):
            salary_info = f"Average {row['med_salary']:,.0f}"

        # Integrate job information
        job_text = f"""
        Company: {row['company_name']}
        Position: {row['title']}
        Work Type: {row['work_type']}
        Salary: {salary_info}
        
        Required Skills:
        {row['skills_desc']}
        
        Job Description:
        {row['description']}
        """

        # Store with metadata
        job_documents.append(
            {
                "content": job_text,
                "metadata": {
                    "company": row["company_name"],
                    "title": row["title"],
                    "work_type": row["work_type"],
                    "salary": salary_info,
                    "skills": row["skills_desc"],
                },
            }
        )

    return job_documents


# Usage example
job_documents = process_job_postings(linkedin_df)
```

## Setting Up ChromaDB and Storing Data

Using ChromaDB for Storing and Retrieving Resume and Job Posting Data
In this section, we will explore how to use ChromaDB to store resume and job posting data as vector representations and perform similarity-based searches.

What is `ChromaDB`?

`ChromaDB` is a vector database that allows text data to be stored as embeddings, enabling efficient similarity-based searches. In our Resume Recommendation System, ChromaDB is used for the following purposes:

- Vectorizing Text: Converting resume and job posting text into vector representations.
- Efficient Similarity Search: Performing fast searches based on the similarity of embeddings.
- Metadata-Based Search and Filtering: Enhancing search results with filters like job title, or company name.


Setup Steps
Preparing Required Libraries

Before starting, import the necessary libraries:

Roles of Each Library:

- `langchain_community.vectorstores`: Provides integration with ChromaDB.
- `langchain_openai`: Enables the use of OpenAI embedding models.
- `chromadb`: Provides vector database functionality.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
```

Initializing ChromaDB

Set up ChromaDB and create collections:

Why Use PersistentClient?
- `Permanent Data Storage`: Ensures that data is not lost when the application or session ends.
- `Data Persistence Across Sessions`: Allows the system to retain data for use in future queries without requiring re-upload or re-processing.
- `Ease of Backup and Recovery`: Provides a reliable way to save and restore data for robustness and fault tolerance.

```python
client = chromadb.PersistentClient(path="../data/chromadb")

# Create separate collections for resumes and job postings
resume_collection = client.create_collection("resumes")
job_collection = client.create_collection("jobs")
```

Storing Data
This step involves saving resume and job posting data into ChromaDB for efficient querying and management.
Origin data has too many data, so we use only 500 data

```python
# Prepare resume data
resume_file_path = "../data/joannadrummond-cv.pdf"  # Path to the resume PDF file
resume_chunks = process_resume(
    resume_file_path
)  # Using the previously defined process_resume function

resume_texts = [chunk[0] for chunk in resume_chunks]
resume_metadatas = [
    {"source": "resume", "type": "text", "weight": chunk[1]} for chunk in resume_chunks
]

resume_ids = [f"resume_chunk_{i}" for i in range(len(resume_chunks))]

# origin data has too many data, so we use only 500 data
job_documents_ = job_documents[:500]

# Prepare job posting data (same as before)
job_texts = [doc["content"] for doc in job_documents_]
job_metadatas = [doc["metadata"] for doc in job_documents_]
job_ids = [f"job_{i}" for i in range(len(job_documents_))]

# Generate and store embeddings
embeddings = OpenAIEmbeddings()

# Resume embeddings
resume_embeddings = embeddings.embed_documents(resume_texts)
resume_collection.add(
    embeddings=resume_embeddings,
    documents=resume_texts,
    metadatas=resume_metadatas,
    ids=resume_ids,
)

# Job posting embeddings
job_embeddings = embeddings.embed_documents(job_texts)
job_collection.add(
    embeddings=job_embeddings, documents=job_texts, metadatas=job_metadatas, ids=job_ids
)
```

<pre class="custom">Resume analysis completed.
    Number of extracted chunks: 10
    
    Career Analysis Summary:
    ------------------------
    Interests: Joanna Drummond's academic and research interests primarily lie in the fields of computer science and artificial intelligence, with a focus on algorithms, stable matching problems, and decision-making under uncertainty. Her research has extensively explored topics such as stable and approximately stable matching using multi-attribute preference information, preference elicitation, and the application of machine learning techniques to educational technologies and dialogue systems. Her work often involves investigating equilibrium concepts in matching games and developing algorithms for solving complex matching problems.
    
    Recommended Roles:
</pre>

Example of Job_documents_

```python
job_documents_[0]
```




<pre class="custom">{'content': '\n        Company: Corcoran Sawyer Smith\n        Position: Marketing Coordinator\n        Work Type: FULL_TIME\n        Salary: 17 - 20\n        \n        Required Skills:\n        Requirements: We are seeking a College or Graduate Student (can also be completed with school) with a focus in Planning, Architecture, Real Estate Development or Management or General Business. Must be able to work in an extremely fast paced environment and able to multitask and prioritize.\n        \n        Job Description:\n        Job descriptionA leading real estate firm in New Jersey is seeking an administrative Marketing Coordinator with some experience in graphic design. You will be working closely with our fun, kind, ambitious members of the sales team and our dynamic executive team on a daily basis. This is an opportunity to be part of a fast-growing, highly respected real estate brokerage with a reputation for exceptional marketing and extraordinary culture of cooperation and inclusion.Who you are:You must be a well-organized, creative, proactive, positive, and most importantly, kind-hearted person. Please, be responsible, respectful, and cool-under-pressure. Please, be proficient in Adobe Creative Cloud (Indesign, Illustrator, Photoshop) and Microsoft Office Suite. Above all, have fantastic taste and be a good-hearted, fun-loving person who loves working with people and is eager to learn.Role:Our office is a fast-paced environment. You’ll work directly with a Marketing team and communicate daily with other core staff and our large team of agents. This description is a brief overview, but your skills and interests will be considered in what you work on and as the role evolves over time.Agent Assistance- Receive & Organize Marketing Requests from Agents- Track Tasks & Communicate with Marketing team & Agents on Status- Prepare print materials and signs for open houses- Submit Orders to Printers & Communicate & Track DeadlinesGraphic Design & Branding- Managing brand strategy and messaging through website, social media, videos, online advertising, print placement and events- Receive, organize, and prioritize marketing requests from agents- Fulfill agent design requests including postcards, signs, email marketing and property brochures using pre-existing templates and creating custom designs- Maintain brand assets and generic filesEvents & Community- Plan and execute events and promotions- Manage Contacts & Vendors for Event Planning & SponsorshipsOur company is committed to creating a diverse environment and is proud to be an equal opportunity employer. All qualified applicants will receive consideration for employment without regard to race, color, religion, gender, gender identity or expression, sexual orientation, national origin, genetics, disability, age, or veteran status.Job Type: Full-time Pay: $18-20/hour Expected hours: 35 – 45 per week Benefits:Paid time offSchedule:8 hour shiftMonday to FridayExperience:Marketing: 1 year (Preferred)Graphic design: 2 years (Preferred)Work Location: In person\n        ',
     'metadata': {'company': 'Corcoran Sawyer Smith',
      'title': 'Marketing Coordinator',
      'work_type': 'FULL_TIME',
      'salary': '17 - 20',
      'skills': 'Requirements: We are seeking a College or Graduate Student (can also be completed with school) with a focus in Planning, Architecture, Real Estate Development or Management or General Business. Must be able to work in an extremely fast paced environment and able to multitask and prioritize.'}}</pre>



## Company Recommendation System

This section focuses on recommending companies that align with the candidate's resume and evaluates the recommendations using two key metrics:

1. **Cosine Similarity for Recommendation Evaluation**:  
   - Measures the similarity between the candidate's resume and the job posting.  
   - A higher cosine similarity score indicates a stronger match between the candidate's profile and the company's job requirements.

2. **NDCG (Normalized Discounted Cumulative Gain) for Recommendation Evaluation**:  
   - Assesses the quality of the ranking of recommended companies.  
   - A higher NDCG score signifies that the most relevant companies appear at the top of the recommendation list, reflecting better ranking performance.

### Understanding the Scores
- **High Scores**:  
   - Indicate a strong alignment between the resume and the recommended companies (Cosine Similarity).  
   - Demonstrate that the ranking system effectively prioritizes the most relevant companies (NDCG).  
- **Low Scores**:  
   - Suggest weaker matches between the resume and job postings or suboptimal ranking of recommendations.  

The goal is to achieve high scores in both metrics, ensuring accurate and effective company recommendations for the candidate.

Job Recommendation System with Weighted Similarity Search

This implementation utilizes a **Job Recommendation System** to match resumes with the most relevant job postings. By combining **cosine similarity** and **weighted scoring**, the system ensures accurate and tailored recommendations.


---
- **Personalized Matching**: Matches resumes to job postings with high accuracy.
- **Flexible Scoring**: Incorporates weighted factors to prioritize specific job attributes.
- **Enhanced Readability**: Formats job descriptions for easy review.

```python
import numpy as np
from typing import List, Dict, Union

# Retrieve resume and job posting collections
resume_collection = client.get_collection("resumes")
job_collection = client.get_collection("jobs")

# Retrieve stored resume data from ChromaDB
resume_results = resume_collection.get(
    include=["documents", "metadatas"]
)  # Use the get() method to fetch all data

# Combine resume texts
full_resume_text = " ".join(resume_results["documents"])

# Configure embeddings
embeddings = OpenAIEmbeddings()

# Convert the resume text into a query vector
query_embedding = embeddings.embed_query(full_resume_text)

# Use ChromaDB to search for the top 5 most similar job postings
job_results = job_collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

# List to store recommended jobs
recommended_jobs = []


class JobRecommender:
    def __init__(self, resume_collection, job_collection):
        self.resume_collection = resume_collection
        self.job_collection = job_collection
        self.embeddings = OpenAIEmbeddings()

    def get_resume_text(self) -> str:
        """Get combined resume text from collection"""
        resume_results = self.resume_collection.get(include=["documents", "metadatas"])
        return " ".join(resume_results["documents"])

    def get_query_embedding(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        return self.embeddings.embed_query(text)

    def weighted_similarity_search(
        self, query_embedding: List[float], method: str = "cosine", n_results: int = 5
    ) -> List[Dict]:
        """
        Search jobs using weighted similarity

        Args:
            query_embedding: The query embedding vector
            method: Similarity method ('cosine' or 'distance')
            n_results: Number of results to return
        """
        include_params = ["documents", "metadatas"]
        include_params.append("embeddings" if method == "cosine" else "distances")

        results = self.job_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2,  # Get more results for reranking
            include=include_params,
        )

        weighted_results = []
        for i in range(len(results["documents"][0])):
            weight = results["metadatas"][0][i].get("weight", 1.0)

            if method == "cosine":
                doc_embedding = results["embeddings"][0][i]
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
            else:  # distance
                distance = results["distances"][0][i]
                similarity = 1 - distance

            weighted_score = similarity * weight
            job_desc = self._clean_job_description(results["documents"][0][i])

            # Ensure consistent dictionary structure with search_jobs_by_distance
            weighted_results.append(
                {
                    "company": results["metadatas"][0][i].get("company", "Unknown"),
                    "title": results["metadatas"][0][i].get("title", "Unknown"),
                    "description": job_desc,
                    "similarity": weighted_score,  # Use weighted_score as the similarity
                    "metadata": results["metadatas"][0][i],
                }
            )

        # Sort by weighted score and get top results
        weighted_results.sort(key=lambda x: x["similarity"], reverse=True)
        return weighted_results[:n_results]

    def _clean_job_description(self, description: str) -> str:
        """Clean job description text"""
        return description.strip().replace("\n\n", "\n")

    def print_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Print job recommendations and return the results"""
        print("\n=== Similar Job Posting Search Results ===")  # Changed from Korean
        results = []

        for i, job in enumerate(recommendations, 1):
            print(f"\n\nJob Posting #{i}")
            print("=" * 80)
            print(f"Company: {job['company']}")
            print(f"Position: {job['title']}")
            print(f"Similarity Score: {job['similarity']:.2f}")
            print("\n[Job Description]")

            desc_lines = [
                line.strip() for line in job["description"].split("\n") if line.strip()
            ]
            for line in desc_lines:
                print(line)
            print("-" * 80)

            # Add results to list  # Changed from Korean
            results.append(
                {
                    "company": job["company"],
                    "title": job["title"],
                    "description": job["description"],
                    "similarity": job["similarity"],
                }
            )

        return results


# Execution section modification  # Changed from Korean
# Initialize collections
resume_collection = client.get_collection("resumes")
job_collection = client.get_collection("jobs")

# Create recommender instance
recommender = JobRecommender(resume_collection, job_collection)

# Get resume text and create query embedding
resume_text = recommender.get_resume_text()
query_embedding = recommender.get_query_embedding(resume_text)

# Get recommendations using different methods
weighted_recommendations = recommender.weighted_similarity_search(
    query_embedding, method="cosine"
)

# Print results and store them
print("\n=== Weighted Recommendations ===")
recommended_jobs = recommender.print_recommendations(weighted_recommendations)
```

<pre class="custom">
    === Weighted Recommendations ===
    
    === Similar Job Posting Search Results ===
    
    
    Job Posting #1
    ================================================================================
    Company: Symbolica AI
    Position: Senior Machine Learning Research Engineer
    Similarity Score: 0.77
    
    [Job Description]
    Company: Symbolica AI
    Position: Senior Machine Learning Research Engineer
    Work Type: FULL_TIME
    Salary: 150,000 - 350,000
    Required Skills:
    Job Description:
    Symbolica is building a new foundation for large-scale AI using structured, interpretable reasoning. We are expanding our team and seeking machine learning research engineers to contribute to the development of our cutting-edge code synthesis and theorem proving models. This is an opportunity to be part of a transformative project and make significant contributions to the field of AI. Responsibilities:Contribute to the design and implementation of machine learning architectures and algorithms for theorem proving, code synthesis, and text generationScale prototype models up using distributed training techniquesDevelop optimized GPU kernels to maximize model performanceIdentify performance bottlenecks using benchmarking and profiling toolsDesign and implement new mechanisms for model parallelismDesign and execute experiments to guide model development process while making effective use of compute budgetDevelop tools to gain insight into model behavior via fine-grained reporting and visualizationMaintain a deep understanding of current techniques in deep learning. Understand, implement, and improve on methods described in machine learning literatureCollaborate with a team of machine learning researchers and engineers to achieve project goals Qualifications:Proficiency with Python deep learning libraries such as PyTorch and JAXExperience with distributed training of large scale deep learning modelsFive years of experience in non-academic machine learning engineering roles, or two years with a relevant PhDNice to have: Proficiency with GPU kernel development using CUDA or Triton In-person in our Bay Area office is preferred, but we will be happy to consider exceptional candidates in other locations. We offer competitive compensation, including equity, health insurance, and 401k benefits. Salary and equity levels are commensurate with experience and location.
    --------------------------------------------------------------------------------
    
    
    Job Posting #2
    ================================================================================
    Company: Georgia Tech Research Institute
    Position: Field Office ISSM - Open Rank-RS-Albuquerque, NM
    Similarity Score: 0.77
    
    [Job Description]
    Company: Georgia Tech Research Institute
    Position: Field Office ISSM - Open Rank-RS-Albuquerque, NM
    Work Type: FULL_TIME
    Salary: No salary information
    Required Skills:
    Job Description:
    Overview The Georgia Tech Research Institute (GTRI) is the nonprofit, applied research division of the Georgia Institute of Technology (Georgia Tech). Founded in 1934 as the Engineering Experiment Station, GTRI has grown to more than 2,900 employees, supporting eight laboratories in over 20 locations around the country and performing more than $940 million of problem-solving research annually for government and industry. GTRI's renowned researchers combine science, engineering, economics, policy, and technical expertise to solve complex problems for the U.S. federal government, state, and industry. Georgia Tech's Mission and Values Georgia Tech's Mission Is To Develop Leaders Who Advance Technology And Improve The Human Condition. The Institute Has Nine Key Values That Are Foundational To Everything We Do Students are our top priority. We strive for excellence. We thrive on diversity. We celebrate collaboration. We champion innovation. We safeguard freedom of inquiry and expression. We nurture the wellbeing of our community. We act ethically. We are responsible stewards. Over the next decade, Georgia Tech will become an example of inclusive innovation, a leading technological research university of unmatched scale, relentlessly committed to serving the public good; breaking new ground in addressing the biggest local, national, and global challenges and opportunities of our time; making technology broadly accessible; and developing exceptional, principled leaders from all backgrounds ready to produce novel ideas and create solutions with real human impact. Project/Unit Description Cyber Security Division (CSD) is responsible for maintaining the overall security posture of classified systems at GTRI. CSD partners with government agencies to provide support for system accreditation and authorization to process classified information in both Collateral and Special (Special Access Programs (SAP) and Sensitive Compartment Information (SCI)) programs. In addition, CSD handles Communication Security (COMSEC) to ensure information is transmitted in a secure manner and in compliance with government regulations. Job Purpose ISSM is a contractually recognized role described in the National Industrial Security Program Operating Manual. The ISSM oversees the development, implementation, and evaluation of the GTRI information security program including insider threat awareness, facility management, personnel supporting information systems, user training and awareness, and others as appropriate. The ISSM develops, documents, monitors and reports the compliance of GTRI information security program in accordance with Cognizant Security Agency (CSA)-provided guidelines for management, operational, and technical controls. The ISSM leads self-inspections and implements corrective actions for all identified findings and vulnerabilities for information security program at the Field Office. The ISSM serves as the principal advisor on all matters, technical and otherwise, involving the security of classified systems at GTRI. They will coordinate and manage GTRI activities related to classified information systems requirements, assessment and authorization of classified information, classified information systems configuration management, and project management for the life cycle of classified information systems. The ISSM advises GTRI senior management and execute GTRI’s overall strategy for enterprise classified networks and systems to support GTRI’s current and future contractual requirements. Additionally, the ISSM researches policies and regulations, interacts with various agencies and levels of management, and contributes to establishing and maintaining accredited information systems to support GTRI contracts with the U.S. Government. The ISSM researches system vulnerabilities and threats to stay on top of the continuous threat against accredited information systems and networks. The Field Office ISSM is also the Assistant Facility Security Officer (AFSO) to assist the full-time Facility Security Officer (FSO) to ensure compliance with governmental regulations within the National Industrial Security Program Operating Manual (NISPOM), Intelligence Community Directives (ICD), Department of Defense (DoD) 5205.07, Volumes 1-4 and National Security Agency/Central Security Service (NSA/CSS) Policy Manual 3-16 and other regulations related to safeguarding and processing of classified information. The poistion will understand and execute requirements within the NISPOM for the management of Personnel Security, Physical and Environmental protections, Incident Handling, and Security Training and Awareness. Key Responsibilities Coordinate and manage the GTRI FO activities related to classified information systems requirements, assessment and authorization of classified information, classified information systems configuration management, and project management for the life cycle of classified information systems.Develop, maintain, and oversee policies, processes and procedures for the classified Information Systems (IS) security program for the Field Office Responsible for analyzing network security systems and/or information systems.Safeguard networks against unauthorized modification, destruction, or disclosure.Research, evaluate, design, test, recommend, communicate, and implement new security software or devices.Implement, enforce, communicate, and develop network or other information security policies or security plans for data, software applications, hardware, telecommunications, and computer installations.Interpret, research, and formalize Cyber Security policies, concepts, and measures when designing, procuring, adopting, and developing new IS to ensure compliance with Government policies, guidance, and orders.Research and advise Information Technology (IT) staff of technical security safeguards and operational security measures and provide technical support in implementing security controls.Define system security requirements, design system security architecture and develop detailed security designs.Manage system security requirements for GTRI’s accredited information systems and assure continuous system compliance.Establish strict program control processes to ensure mitigation of risks and supports obtaining certification and accreditation of systems. This includes process support, analysis support, coordination support, security certification test support, security documentation support, investigations, software research, hardware introduction and release, emerging technology research inspections and periodic audits.Responsible maintaining operational security posture for systems by enforcing established security policies, procedures, and standards.Work with Government security cognizant agencies to identify and manage security findings, risks and mitigations in Plan of Action and Milestones (POA&M).Perform continuous monitoring activities including system security audits and vulnerability scanning and remediation.Periodically conduct of a review of each system's audits and monitor corrective actions until all actions are closed.Ensure Configuration Management (CM) of all associated software, hardware, and security relevant functionsLead incident response process to include document and report to appropriate authorityResearch system vulnerabilities and threats to stay on top of the continuous threat against accredited systemsPrepare for and participate in self-inspection and Government security vulnerability assessments.Serve as secondary point of contact for all industrial security concerns. Assist FSO to manage and support the GTRI Field Office classified security programs.Assist FSO to develop and administer security education, training, and awareness programs for both cleared and non-cleared personnel.Assist FSO to maintain visitor control program Required Minimum Qualifications Must be able to obtain or have a current TS/SCI clearanceBachelor degree in Computer Engineering, Electrical Engineering, Computer Science, Information Assurance, Information Security or related fields.Must possess or be able to obtain CISSP, Security+ and/or other applicable certifications within six months of hire in compliance with DoD Directive 8140/8570, IAM Level II/III baseline certification requirements.Have experience with JSIG, RMF, ICD 503, NIST 800, NISPOM and DAAPMExperience with information systems Incident Response TeamExperience identifying system vulnerabilities and implementing mitigation strategies. Preferred Qualifications Active TS/SCI ClearanceIAM Level III compliance with CISSPExperience in an environment and culture steeped in teamwork and collaboration working on challenging technical projectsExperience working with XACTA/eMASS Travel Requirements <10% travel Education And Length Of Experience This position vacancy is an open-rank announcement. The final job offer will be dependent on candidate qualifications in alignment with Research Faculty Extension Professional ranks as outlined in section 3.2.1 of the Georgia Tech Faculty Handbook 0 years of related experience with a Bachelor’s degree in Computer Engineering, Electrical Engineering, Computer Science, Information Assurance, Information Security or related fields. U.S. Citizenship Requirements Due to our research contracts with the U.S. federal government, candidates for this position must be U.S. Citizens. Clearance Type Required Candidates must be able to obtain and maintain an active security clearance. Benefits At GTRI Comprehensive information on currently offered GTRI benefits, including Health & Welfare, Retirement Plans, Tuition Reimbursement, Time Off, and Professional Development, can be found through this link: https://hr.gatech.edu/benefits Equal Employment Opportunity The Georgia Institute of Technology (Georgia Tech) is an Equal Employment Opportunity Employer. The University is committed to maintaining a fair and respectful environment for all. To that end, and in accordance with federal and state law, Board of Regents policy, and University policy, Georgia Tech provides equal opportunity to all faculty, staff, students, and all other members of the Georgia Tech community, including applicants for admission and/or employment, contractors, volunteers, and participants in institutional programs, activities, or services. Georgia Tech complies with all applicable laws and regulations governing equal opportunity in the workplace and in educational activities. Georgia Tech prohibits discrimination, including discriminatory harassment, on the basis of race, ethnicity, ancestry, color, religion, sex (including pregnancy), sexual orientation, gender identity, gender expression, national origin, age, disability, genetics, or veteran status in its programs, activities, employment, and admissions. This prohibition applies to faculty, staff, students, and all other members of the Georgia Tech community, including affiliates, invitees, and guests. Further, Georgia Tech prohibits citizenship status, immigration status, and national origin discrimination in hiring, firing, and recruitment, except where such restrictions are required in order to comply with law, regulation, executive order, or Attorney General directive, or where they are required by Federal, State, or local government contract. All members of the USG community must adhere to the USG Statement of Core Values, which consists of Integrity, Excellence, Accountability, and Respect. These values shape and fundamentally support our University's work. Additionally, all faculty, staff, and administrators must also be aware of and comply with the Board of Regents and Georgia Institute of Technology's policies on Freedom of Expression and Academic Freedom. More information on these policies can be found here: Board of Regents Policy Manual | University System of Georgia (usg.edu).
    --------------------------------------------------------------------------------
    
    
    Job Posting #3
    ================================================================================
    Company: Georgia Tech Research Institute
    Position: Field Office ISSM - Open Rank-RS-Albuquerque, NM
    Similarity Score: 0.77
    
    [Job Description]
    Company: Georgia Tech Research Institute
    Position: Field Office ISSM - Open Rank-RS-Albuquerque, NM
    Work Type: FULL_TIME
    Salary: No salary information
    Required Skills:
    Job Description:
    Overview The Georgia Tech Research Institute (GTRI) is the nonprofit, applied research division of the Georgia Institute of Technology (Georgia Tech). Founded in 1934 as the Engineering Experiment Station, GTRI has grown to more than 2,900 employees, supporting eight laboratories in over 20 locations around the country and performing more than $940 million of problem-solving research annually for government and industry. GTRI's renowned researchers combine science, engineering, economics, policy, and technical expertise to solve complex problems for the U.S. federal government, state, and industry. Georgia Tech's Mission and Values Georgia Tech's Mission Is To Develop Leaders Who Advance Technology And Improve The Human Condition. The Institute Has Nine Key Values That Are Foundational To Everything We Do Students are our top priority. We strive for excellence. We thrive on diversity. We celebrate collaboration. We champion innovation. We safeguard freedom of inquiry and expression. We nurture the wellbeing of our community. We act ethically. We are responsible stewards. Over the next decade, Georgia Tech will become an example of inclusive innovation, a leading technological research university of unmatched scale, relentlessly committed to serving the public good; breaking new ground in addressing the biggest local, national, and global challenges and opportunities of our time; making technology broadly accessible; and developing exceptional, principled leaders from all backgrounds ready to produce novel ideas and create solutions with real human impact. Project/Unit Description Cyber Security Division (CSD) is responsible for maintaining the overall security posture of classified systems at GTRI. CSD partners with government agencies to provide support for system accreditation and authorization to process classified information in both Collateral and Special (Special Access Programs (SAP) and Sensitive Compartment Information (SCI)) programs. In addition, CSD handles Communication Security (COMSEC) to ensure information is transmitted in a secure manner and in compliance with government regulations. Job Purpose ISSM is a contractually recognized role described in the National Industrial Security Program Operating Manual. The ISSM oversees the development, implementation, and evaluation of the GTRI information security program including insider threat awareness, facility management, personnel supporting information systems, user training and awareness, and others as appropriate. The ISSM develops, documents, monitors and reports the compliance of GTRI information security program in accordance with Cognizant Security Agency (CSA)-provided guidelines for management, operational, and technical controls. The ISSM leads self-inspections and implements corrective actions for all identified findings and vulnerabilities for information security program at the Field Office. The ISSM serves as the principal advisor on all matters, technical and otherwise, involving the security of classified systems at GTRI. They will coordinate and manage GTRI activities related to classified information systems requirements, assessment and authorization of classified information, classified information systems configuration management, and project management for the life cycle of classified information systems. The ISSM advises GTRI senior management and execute GTRI’s overall strategy for enterprise classified networks and systems to support GTRI’s current and future contractual requirements. Additionally, the ISSM researches policies and regulations, interacts with various agencies and levels of management, and contributes to establishing and maintaining accredited information systems to support GTRI contracts with the U.S. Government. The ISSM researches system vulnerabilities and threats to stay on top of the continuous threat against accredited information systems and networks. The Field Office ISSM is also the Assistant Facility Security Officer (AFSO) to assist the full-time Facility Security Officer (FSO) to ensure compliance with governmental regulations within the National Industrial Security Program Operating Manual (NISPOM), Intelligence Community Directives (ICD), Department of Defense (DoD) 5205.07, Volumes 1-4 and National Security Agency/Central Security Service (NSA/CSS) Policy Manual 3-16 and other regulations related to safeguarding and processing of classified information. The poistion will understand and execute requirements within the NISPOM for the management of Personnel Security, Physical and Environmental protections, Incident Handling, and Security Training and Awareness. Key Responsibilities Coordinate and manage the GTRI FO activities related to classified information systems requirements, assessment and authorization of classified information, classified information systems configuration management, and project management for the life cycle of classified information systems.Develop, maintain, and oversee policies, processes and procedures for the classified Information Systems (IS) security program for the Field Office.Responsible for analyzing network security systems and/or information systems. Safeguard networks against unauthorized modification, destruction, or disclosure.Research, evaluate, design, test, recommend, communicate, and implement new security software or devices.Implement, enforce, communicate, and develop network or other information security policies or security plans for data, software applications, hardware, telecommunications, and computer installations.Interpret, research, and formalize Cyber Security policies, concepts, and measures when designing, procuring, adopting, and developing new IS to ensure compliance with Government policies, guidance, and orders.Research and advise Information Technology (IT) staff of technical security safeguards and operational security measures and provide technical support in implementing security controls.Perform examination and quality control inspections on Information Systems Security protections and safeguards to ensure compliance to Government requirements and standards.Define system security requirements, design system security architecture and develop detailed security designs.Assess information protection effectiveness and plan and manage technical efforts.Manage system security requirements for GTRI’s accredited information systems and assure continuous system compliance.Establish strict program control processes to ensure mitigation of risks and supports obtaining certification and accreditation of systems. This includes process support, analysis support, coordination support, security certification test support, security documentation support, investigations, software research, hardware introduction and release, emerging technology research inspections and periodic audits.Responsible maintaining operational security posture for systems by enforcing established security policies, procedures, and standards.Work with Government security cognizant agencies to identify and manage security findings, risks and mitigations in Plan of Action and Milestones (POA&M).Perform continuous monitoring activities including system security audits and vulnerability scanning and remediation.Periodically conduct of a review of each system's audits and monitor corrective actions until all actions are closed.Ensure Configuration Management (CM) of all associated software, hardware, and security relevant functionsLead incident response process to include document and report to appropriate authorityResearch policies and regulations, interact with various agencies and levels of management, and contribute to establishing and maintaining accredited information systemsResearch system vulnerabilities and threats to stay on top of the continuous threat against accredited systemsPrepare for and participate in self-inspection and Government security vulnerability assessments.Support the formal Security Test and Evaluation (ST&E) required by each government accrediting authority through pre-test preparations, participation in the tests, analysis of the results, and preparation of required reports.Serve as the secondary point of contact for all industrial security concerns.Assist the FSO to manage and support the GTRI Field Office classified security programs.Assist the FSO to develop and administer security education, training, and awareness programs for both cleared and non-cleared personnel.Assist the FSO to maintain visitor control program. Required Minimum Qualifications Must be able to obtain or have a current TS/SCI clearanceBachelor degree in Computer Engineering, Electrical Engineering, Computer Science, Information Assurance, Information Security or related fields.Must possess or be able to obtain CISSP, Security+ and/or other applicable certifications within six months of hire in compliance with DoD Directive 8140/8570, IAM Level II/III baseline certification requirements.Have experience with JSIG, RMF, ICD 503, NIST 800, NISPOM and DAAPMExperience with information systems Incident Response TeamExperience identifying system vulnerabilities and implementing mitigation strategies Preferred Qualifications Active TS/SCI Clearance IAM Level III compliance with CISSPExperience in an environment and culture steeped in teamwork and collaboration working on challenging technical projectsExperience working with XACTA/eMASS Travel Requirements <10% travel Education And Length Of Experience This position vacancy is an open-rank announcement. The final job offer will be dependent on candidate qualifications in alignment with Research Faculty Extension Professional ranks as outlined in section 3.2.1 of the Georgia Tech Faculty Handbook 2 years of related experience with a Bachelor’s degree in Computer Engineering, Electrical Engineering, Computer Science, Information Assurance, Information Security or related fields.0 years of related experience with a Masters’ degree in Computer Engineering, Electrical Engineering, Computer Science, Information Assurance, Information Security or related fields. U.S. Citizenship Requirements Due to our research contracts with the U.S. federal government, candidates for this position must be U.S. Citizens. Clearance Type Required Candidates must be able to obtain and maintain an active security clearance. Benefits At GTRI Comprehensive information on currently offered GTRI benefits, including Health & Welfare, Retirement Plans, Tuition Reimbursement, Time Off, and Professional Development, can be found through this link: https://hr.gatech.edu/benefits Equal Employment Opportunity The Georgia Institute of Technology (Georgia Tech) is an Equal Employment Opportunity Employer. The University is committed to maintaining a fair and respectful environment for all. To that end, and in accordance with federal and state law, Board of Regents policy, and University policy, Georgia Tech provides equal opportunity to all faculty, staff, students, and all other members of the Georgia Tech community, including applicants for admission and/or employment, contractors, volunteers, and participants in institutional programs, activities, or services. Georgia Tech complies with all applicable laws and regulations governing equal opportunity in the workplace and in educational activities. Georgia Tech prohibits discrimination, including discriminatory harassment, on the basis of race, ethnicity, ancestry, color, religion, sex (including pregnancy), sexual orientation, gender identity, gender expression, national origin, age, disability, genetics, or veteran status in its programs, activities, employment, and admissions. This prohibition applies to faculty, staff, students, and all other members of the Georgia Tech community, including affiliates, invitees, and guests. Further, Georgia Tech prohibits citizenship status, immigration status, and national origin discrimination in hiring, firing, and recruitment, except where such restrictions are required in order to comply with law, regulation, executive order, or Attorney General directive, or where they are required by Federal, State, or local government contract. All members of the USG community must adhere to the USG Statement of Core Values, which consists of Integrity, Excellence, Accountability, and Respect. These values shape and fundamentally support our University's work. Additionally, all faculty, staff, and administrators must also be aware of and comply with the Board of Regents and Georgia Institute of Technology's policies on Freedom of Expression and Academic Freedom. More information on these policies can be found here: Board of Regents Policy Manual | University System of Georgia (usg.edu).
    --------------------------------------------------------------------------------
    
    
    Job Posting #4
    ================================================================================
    Company: Azure Sky Management Consulting
    Position: Quantitative Researcher - Semi-Systematic Credit
    Similarity Score: 0.77
    
    [Job Description]
    Company: Azure Sky Management Consulting
    Position: Quantitative Researcher - Semi-Systematic Credit
    Work Type: FULL_TIME
    Salary: No salary information
    Required Skills:
    Job Description:
    Company Insight:The company is a world-leading algorithmic trading house that distinguishes itself from its competitors by its bespoke, bleeding-edge, technological systems which materialize a vast array of heavy return systematic quant-driven strategies. They are leaders in the fields of Mathematics, Computer Science and Engineering, ably processing petabytes of data to conceive complex and undiscovered strategies which range from short to long in holding periods. They are much more than a high-frequency trading firm. After a record year, the highly successful Fixed Income team is looking to invest more than ever before into brand-new systematic strategies across the Credit space. They are looking for a Quant Researcher to join their team in New York and come on board to help in the advancement of their current trading universe, as they seek to add Flow Credit products to their global (though mostly US-centric) systematic credit trading desk. Your Role:Investigate and implement game-changing ways to improve the quantitative analytics library, with a focus on credit flow productsConceive valuation strategies, build mathematical models and translate algorithms into impeccably clean codeApply statistical and predictive modeling techniques to process and analyze large and varied data setsBe open to projects in which traders require a specific piece of functionality, and others where a department-wide strategy needs implementing Experience/Skills Required:1-8 years experience within a front office MM credit quantitative research roleKnowledge of credit flow products (investment grade and/or high-yield corporate bonds, CDS and index pricing etc)Strong research agenda stemming from experience in a research-heavy credit flow MM team and from academic pedigreeReasonable experience in writing production level C++ and/or Python codeThe ability to work collaboratively with a diverse range of technological and quantitative individuals towards shared goals Pre-Application:Please do not apply if you're looking for a contract or remote work.Please ensure you meet the required experience section prior to applying.Allow 1-5 working days for a response to any job enquiry.Your application is subject to our privacy policy, found here: https://www.thurnpartners.com/privacy-policy
    --------------------------------------------------------------------------------
    
    
    Job Posting #5
    ================================================================================
    Company: University of Tehran
    Position: Researcher
    Similarity Score: 0.77
    
    [Job Description]
    Company: University of Tehran
    Position: Researcher
    Work Type: OTHER
    Salary: No salary information
    Required Skills:
    Job Description:
    Company Description The University of Tehran, established over seven centuries ago, is a renowned institution of higher education in Iran. It has evolved from a traditional religious school to a modern and academic structure. With campuses located in Tehran, Qom, Karaj, Kish, and Jolfa, the University offers 976 programs in over 500 fields across its 39 faculties and 120 departments. It is home to 15% of the country's Centers of Excellence and houses more than 40 research centers, 3,500 laboratories, and a leading press that publishes over 50 scientific peer-reviewed journals. Role Description This is a full-time on-site role for a Researcher at the University of Tehran's San Diego, CA location. The Researcher will be responsible for conducting research, collecting and analyzing data, preparing reports, and collaborating with other researchers. The role will involve staying up-to-date with the latest developments in the field, attending conferences and seminars, and publishing research findings. Qualifications Strong research skills, including data collection and analysisExcellent written and verbal communication skillsAbility to work effectively in a team and independentlyProficiency in conducting academic research and writing research reportsKnowledge of research methodologies and statistical analysisExperience with research software and toolsStrong organizational and time management skillsA PhD or Master's degree in a relevant field
    --------------------------------------------------------------------------------
</pre>

Resume and Job Recommendation Evaluation System

This implementation introduces a comprehensive evaluation system for job recommendations based on resumes.

 The system leverages **Discounted Cumulative Gain (DCG)** and **Normalized Discounted Cumulative Gain (NDCG)** to measure the quality of recommendations. Additionally, precision and recall metrics are calculated for further analysis.

```python
from typing import List, Dict
import numpy as np
import math


class ResumeProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    def process_resume(self, resume_texts: List[str]) -> str:
        """Process text using already refined resume_texts"""
        return " ".join(resume_texts)


class NDCGEvaluator:
    def __init__(self, model_name="gpt-4", temperature=0.2):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Ground truth generation prompt
        self.relevance_prompt = ChatPromptTemplate.from_template(
            """
        As an expert recruiter, evaluate the relevance between this resume and job posting.
        Consider technical skills, experience level, and overall fit.

        Resume:
        {resume_text}

        Job Posting:
        {job_text}

        Rate the relevance on a scale of 0 to 1, where:
        1.0: Perfect match
        0.8: Very good match
        0.6: Good match
        0.4: Moderate match
        0.2: Poor match
        0.0: No match

        Provide only the numerical score, nothing else.
        """
        )

    def calculate_dcg(self, relevance_scores: List[float], k: int = None) -> float:
        """Calculate Discounted Cumulative Gain for ranking evaluation.

        Formula: DCG = sum(rel_i / log2(i + 2)) where rel_i is the relevance of item i

        Args:
            relevance_scores (List[float]): List of relevance scores
            k (int, optional): Number of top items to consider

        Returns:
            float: DCG score
        """
        if k is None:
            k = len(relevance_scores)
        else:
            k = min(k, len(relevance_scores))

        dcg = 0.0
        for i in range(k):
            # 2^rel - 1 is commonly used for NDCG calculation to emphasize relevant items
            rel = 2 ** relevance_scores[i] - 1
            dcg += rel / math.log2(i + 2)
        return dcg

    def calculate_ndcg(
        self, predicted_scores: List[float], ideal_scores: List[float], k: int = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain
        NDCG = DCG / IDCG where IDCG is DCG of ideal ordering
        """
        if not predicted_scores or not ideal_scores:
            return 0.0

        # Limit to top k items if specified
        if k is None:
            k = len(predicted_scores)
        k = min(k, len(predicted_scores))

        # Sort ideal scores in descending order
        ideal_scores_sorted = sorted(ideal_scores, reverse=True)

        # Calculate DCG for predicted and ideal rankings
        dcg = self.calculate_dcg(predicted_scores[:k], k)
        idcg = self.calculate_dcg(ideal_scores_sorted[:k], k)

        # Avoid division by zero and ensure score is between 0 and 1
        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        # Ensure NDCG is between 0 and 1
        return max(0.0, min(1.0, ndcg))

    def generate_ground_truth(
        self, resume_text: str, job_postings: List[Dict]
    ) -> Dict[str, float]:
        """Generate ground truth relevance scores using LLM"""
        ground_truth = {}

        for job in job_postings:
            if job["company"] == "Unknown":
                continue

            messages = self.relevance_prompt.format_messages(
                resume_text=resume_text, job_text=job["description"]
            )

            response = self.llm.invoke(messages)
            try:
                score = float(response.content.strip())
                ground_truth[job["company"]] = score
            except ValueError:
                print(f"Error parsing score for company {job['company']}")
                ground_truth[job["company"]] = 0.0

        return ground_truth

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 for _ in scores]

        return [(score - min_score) / (max_score - min_score) for score in scores]

    def evaluate_recommendations(
        self, resume_text: str, recommended_jobs: List[Dict], k: int = None
    ) -> Dict[str, float]:
        """Evaluate recommendations using NDCG"""
        # Filter out Unknown companies
        valid_jobs = [job for job in recommended_jobs if job["company"] != "Unknown"]

        # Generate ground truth scores
        ground_truth = self.generate_ground_truth(resume_text, valid_jobs)

        # Get predicted scores and normalize them
        predicted_scores = [job["similarity"] for job in valid_jobs]
        predicted_scores = self.normalize_scores(predicted_scores)

        # Get ideal scores in the same order as predictions
        ideal_scores = [ground_truth[job["company"]] for job in valid_jobs]

        # Calculate NDCG
        ndcg_score = self.calculate_ndcg(predicted_scores, ideal_scores, k)

        # Additional metrics
        if k is None:
            k = len(valid_jobs)

        # Calculate precision and recall using threshold of 0.6 for relevance
        relevant_recommended = sum(1 for score in ideal_scores[:k] if score >= 0.6)
        total_relevant = sum(1 for score in ground_truth.values() if score >= 0.6)

        precision_at_k = relevant_recommended / k if k > 0 else 0
        recall_at_k = relevant_recommended / total_relevant if total_relevant > 0 else 0

        return {
            "ndcg": ndcg_score,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "ground_truth": ground_truth,
            "normalized_predictions": dict(
                zip([job["company"] for job in valid_jobs], predicted_scores)
            ),
        }


def print_evaluation_results(metrics: Dict[str, float], recommended_jobs: List[Dict]):
    """Print detailed evaluation results"""
    print("\n=== Recommendation Evaluation Results ===")
    print(f"NDCG Score: {metrics['ndcg']:.3f}")
    print(f"Precision@k: {metrics['precision@k']:.3f}")
    print(f"Recall@k: {metrics['recall@k']:.3f}")

    print("\nDetailed Company Scores:")
    print("=" * 80)
    print(f"{'Company':<30} {'Original':<10} {'Normalized':<10} {'Ground Truth':<10}")
    print("-" * 80)

    for job in recommended_jobs:
        company = job["company"]
        if company == "Unknown":
            continue

        original = job["similarity"]
        normalized = metrics["normalized_predictions"].get(company, 0.0)
        ground_truth = metrics["ground_truth"].get(company, 0.0)
        print(
            f"{company:<30} {original:<10.3f} {normalized:<10.3f} {ground_truth:<10.3f}"
        )
```

Excute Evaluation

```python
# Use existing processed resume_texts
resume_processor = ResumeProcessor()
resume_text = resume_processor.process_resume(resume_texts)

evaluator = NDCGEvaluator()
metrics = evaluator.evaluate_recommendations(
    resume_text=resume_text,
    recommended_jobs=recommended_jobs,
    k=5,  # Evaluate top-5 recommendations
)

# Print evaluation results
print_evaluation_results(metrics, recommended_jobs)
```

<pre class="custom">
    === Recommendation Evaluation Results ===
    NDCG Score: 1.000
    Precision@k: 0.400
    Recall@k: 1.000
    
    Detailed Company Scores:
    ================================================================================
    Company                        Original   Normalized Ground Truth
    --------------------------------------------------------------------------------
    Symbolica AI                   0.774      1.000      0.600     
    Georgia Tech Research Institute 0.773      0.843      0.200     
    Georgia Tech Research Institute 0.773      0.843      0.200     
    Azure Sky Management Consulting 0.773      0.821      0.400     
    University of Tehran           0.766      0.000      0.900     
</pre>

## LLM-Based Resume Evaluation System

This section outlines the implementation of a system that uses a **Large Language Model (LLM)** to evaluate resumes by comparing them against job descriptions. The system provides actionable insights to improve resumes and assists in aligning candidates’ qualifications with job requirements.
---

What is `LLM-as-a-Judge`?

The `LLM-as-a-Judge` system leverages the advanced reasoning and natural language understanding capabilities of an LLM to serve as an impartial evaluator in the hiring process. By acting as a "judge," the LLM compares a candidate’s resume to job requirements, evaluates their alignment, and provides actionable feedback.

Key features of the `LLM-as-a-Judge` system include:
- `Contextual Understanding`: It comprehends detailed job descriptions and resumes beyond simple keyword matching, enabling nuanced evaluations.  
- `Feedback Generation`: Provides insights into the candidate's strengths and areas for improvement.  
- `Decision Support`: Assists hiring managers or applicants by generating a recommendation on the candidate's suitability for the role.

This system bridges the gap between human evaluation and automated analysis, ensuring more accurate and tailored results in the recruitment process.

---

Functionalities

The `LLM-as-a-Judge` system provides the following functionalities:

- `Detailed Analysis`: Analyzes resumes and job requirements in detail, identifying key qualifications and expectations.  
- `Alignment Evaluation`: Assesses how well the candidate's skills and experiences match the job requirements.  
- `Strengths and Improvement Areas`: Identifies the candidate's strengths and offers suggestions for improvement.  
- `Role Suitability Recommendation`: Provides a final recommendation on whether the candidate is a good fit for the role.  

---


## Key Components

### 1. **CriterionEvaluation**
The `CriterionEvaluation` class evaluates individual aspects of the resume based on predefined criteria:  
- **`score`** (`int`): A score from 1 to 5 representing the performance for a specific criterion.  
- **`reasoning`** (`str`): A detailed explanation of why the score was assigned.  
- **`evidence`** (`List[str]`): Specific elements from the resume that support the evaluation.  
- **`suggestions`** (`List[str]`): Targeted recommendations for improving the evaluated area.

---

### 2. **DetailedEvaluation**
The `DetailedEvaluation` class provides a comprehensive evaluation of the resume by aggregating results across multiple criteria:  
- **`technical_fit`**: Assessment of technical skills and their relevance to the job.  
- **`experience_relevance`**: Evaluation of how well the candidate’s work experience aligns with the role.  
- **`industry_knowledge`**: Examination of the candidate’s understanding of the target industry.  
- **`education_qualification`**: Review of academic background and certifications.  
- **`soft_skills`**: Analysis of interpersonal and communication skills.  
- **`overall_score`** (`int`): A total score (0-100) summarizing the resume's performance.  
- **`key_strengths`** (`List[str]`): Highlights of the resume's strongest areas.  
- **`improvement_areas`** (`List[str]`): Areas requiring enhancement for better alignment with the job.  
- **`final_recommendation`** (`str`): A conclusion on the candidate’s suitability for the position.

---

### 3. **LLMJudge**
The `LLMJudge` class uses an LLM to evaluate resumes against job descriptions by analyzing criteria such as technical fit, experience relevance, and soft skills.  
- **Responsibilities**:  
  - Processes resume text and job information.  
  - Uses a structured prompt to guide the LLM in scoring and providing feedback.  
  - Outputs a `DetailedEvaluation` object containing scores, evidence, and suggestions.  
- **Features**:  
  - Dynamic prompt generation for precise LLM instructions.  
  - Predefined evaluation criteria with customizable weights and descriptions.

---

### 4. **ResumeEvaluationSystem**
The `ResumeEvaluationSystem` orchestrates the entire resume evaluation process, from text extraction to generating improvement reports.  
- **Responsibilities**:  
  - Processes resumes to extract clean text for analysis.  
  - Selects the most relevant jobs for evaluation based on similarity scores.  
  - Generates detailed reports summarizing the evaluation and suggestions.  
- **Methods**:  
  - **`evaluate_with_recommendations`**: Evaluates a resume against the top `n` recommended jobs.  
  - **`format_evaluation_report`**: Converts the `DetailedEvaluation` object into a readable report.  

---

### Example: Resume Evaluation Results

---

💡 **Overall Score**: **85/100**

---

Evaluation Summary

- **🔧 Technical Fit (30%)**: **4/5**  
  - **Reasoning**: Strong Python and SQL skills; lacks cloud experience.  
  - **Suggestions**: Add cloud certifications like AWS.  

- **👔 Experience Relevance (25%)**: **4/5**  
  - **Reasoning**: Relevant projects but no measurable outcomes.  
  - **Suggestions**: Quantify achievements (e.g., "Increased sales by 15%").  

- **🎯 Industry Knowledge (15%)**: **3/5**  
  - **Reasoning**: Limited mention of industry expertise.  
  - **Suggestions**: Include domain-specific certifications or research.  

- **📚 Education Qualification (15%)**: **5/5**  
  - **Reasoning**: Relevant degree and certifications.  

- **🤝 Soft Skills (15%)**: **4/5**  
  - **Reasoning**: Leadership demonstrated; teamwork examples sparse.  
  - **Suggestions**: Add examples of collaboration.

---

Recommendations
- **Key Strengths**: Strong technical skills, leadership, relevant education.  
- **Improvements**: Add measurable outcomes, industry expertise, teamwork examples.  
- **Final Recommendation**: Highly suitable; minor revisions suggested.

LLM-Based Resume Evaluation System

This system leverages a **Large Language Model (LLM)** to evaluate resumes against job descriptions systematically. It provides detailed feedback based on predefined evaluation criteria, helping candidates understand their strengths, areas for improvement, and overall suitability for specific roles.

```python
from pydantic import BaseModel, Field


# Define Pydantic Models
class CriterionEvaluation(BaseModel):
    """Evaluation result for individual criteria"""

    score: int = Field(description="Evaluation score (1-5)")
    reasoning: str = Field(description="Reasoning behind the score")
    evidence: List[str] = Field(description="Evidence found in the resume")
    suggestions: List[str] = Field(description="Suggestions for improvement")


class DetailedEvaluation(BaseModel):
    """Detailed evaluation results"""

    technical_fit: CriterionEvaluation
    experience_relevance: CriterionEvaluation
    industry_knowledge: CriterionEvaluation
    education_qualification: CriterionEvaluation
    soft_skills: CriterionEvaluation
    overall_score: int = Field(description="Overall score (0-100)")
    key_strengths: List[str] = Field(description="Key strengths")
    improvement_areas: List[str] = Field(description="Areas for improvement")
    final_recommendation: str = Field(description="Final recommendation")


class LLMJudge:
    def __init__(self, model_name="gpt-4o", temperature=0.1):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=DetailedEvaluation)

        # Define evaluation criteria
        self.evaluation_criteria = {
            "technical_fit": {
                "weight": 30,
                "description": "Evaluation of technical fit",
                "subcriteria": [
                    "required_skills_match",
                    "tech_stack_relevance",
                    "skill_proficiency",
                ],
            },
            "experience_relevance": {
                "weight": 25,
                "description": "Evaluation of experience relevance",
                "subcriteria": ["role_similarity", "impact_scale", "problem_solving"],
            },
            "industry_knowledge": {
                "weight": 15,
                "description": "Evaluation of industry knowledge",
                "subcriteria": [
                    "domain_expertise",
                    "trend_awareness",
                    "industry_exposure",
                ],
            },
            "education_qualification": {
                "weight": 15,
                "description": "Evaluation of education and qualifications",
                "subcriteria": [
                    "degree_relevance",
                    "certifications",
                    "continuous_learning",
                ],
            },
            "soft_skills": {
                "weight": 15,
                "description": "Evaluation of soft skills",
                "subcriteria": [
                    "leadership_teamwork",
                    "communication",
                    "problem_approach",
                ],
            },
        }

        # Evaluation prompt template
        self.prompt_template = """You are a professional hiring evaluator.
        Evaluate the provided resume objectively and fairly based on the following criteria.

        Job Information:
        Company: {company_name}
        Position: {position}
        Job Description: {job_description}

        Resume Content:
        {resume_text}

        Evaluation Criteria:
        {evaluation_criteria}

        Guidelines for Evaluation:
        1. Assign a score from 1-5 for each evaluation area and provide detailed reasoning.
        2. Scoring criteria:
           5: Outstanding - Exceeds expectations significantly
           4: Excellent - Meets and slightly exceeds expectations
           3: Adequate - Meets expectations
           2: Needs Improvement - Falls slightly short of expectations
           1: Poor - Falls significantly short of expectations
        3. Provide specific evidence found in the resume for each area.
        4. Offer concrete suggestions for improvement.

        Provide the evaluation results in the following format:
        {format_instructions}
        """

        self.prompt = ChatPromptTemplate.from_template(
            template=self.prompt_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "evaluation_criteria": json.dumps(
                    self.evaluation_criteria, indent=2, ensure_ascii=False
                ),
            },
        )

    def evaluate(self, resume_text: str, job_info: dict) -> DetailedEvaluation:
        """Perform resume evaluation using LLM.

        Args:
            resume_text (str): Processed resume content
            job_info (dict): Dictionary containing job details including:
                - company: Company name
                - position: Job title
                - description: Job description

        Returns:
            DetailedEvaluation: Structured evaluation results including scores and feedback

        Raises:
            Exception: If evaluation process fails
        """

        try:
            # Format the evaluation prompt with job and resume information
            messages = self.prompt.format_messages(
                company_name=job_info.get("company", "Unknown"),
                position=job_info.get("position", "Unknown"),
                job_description=job_info.get("description", ""),
                resume_text=resume_text,
            )

            # Get LLM response for evaluation
            response = self.llm.invoke(messages)
            evaluation = self.parser.parse(response.content)
            return evaluation

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise


class ResumeEvaluationSystem:
    def __init__(self):
        self.resume_processor = ResumeProcessor()
        self.judge = LLMJudge()

    def evaluate_with_recommendations(
        self, resume_path: str, recommended_jobs: List[dict], top_n: int = 3
    ) -> List[Dict]:
        """Evaluate the resume for the recommended jobs"""
        # Extract resume text
        resume_text = self.resume_processor.process_resume(resume_path)

        # Select top N jobs
        sorted_jobs = sorted(
            recommended_jobs, key=lambda x: x["similarity"], reverse=True
        )[:top_n]
        evaluations = []

        for job in sorted_jobs:
            job_info = {
                "company": job["company"],
                "position": job["title"],
                "description": job["description"],
                "similarity_score": job["similarity"],
            }

            try:
                # Perform evaluation
                evaluation = self.judge.evaluate(resume_text, job_info)

                # Generate evaluation report
                report = format_evaluation_report(evaluation)

                evaluations.append(
                    {"job_info": job_info, "evaluation": evaluation, "report": report}
                )

            except Exception as e:
                print(f"Error evaluating for {job_info['company']}: {str(e)}")
                continue

        return evaluations


def format_evaluation_report(evaluation: DetailedEvaluation) -> str:
    """Format evaluation results into a report"""
    output = []
    output.append("\n📊 Resume Evaluation Report")
    output.append("=" * 50)

    output.append(f"\n💡 Overall Score: {evaluation.overall_score}/100\n")

    # Evaluation by criteria
    criteria_items = [
        ("🔧 Technical Fit (30%)", evaluation.technical_fit),
        ("👔 Experience Relevance (25%)", evaluation.experience_relevance),
        ("🎯 Industry Knowledge (15%)", evaluation.industry_knowledge),
        ("📚 Education Qualification (15%)", evaluation.education_qualification),
        ("🤝 Soft Skills (15%)", evaluation.soft_skills),
    ]

    for title, criterion in criteria_items:
        output.append(f"\n{title}")
        output.append(f"Score: {criterion.score}/5")
        output.append(f"Reasoning: {criterion.reasoning}")
        output.append("Evidence Found:")
        for evidence in criterion.evidence:
            output.append(f"  • {evidence}")
        output.append("Suggestions:")
        for suggestion in criterion.suggestions:
            output.append(f"  • {suggestion}")

    # Overall evaluation
    output.append("\n📋 Overall Evaluation")
    output.append("-" * 30)

    output.append("\n💪 Key Strengths:")
    for strength in evaluation.key_strengths:
        output.append(f"  • {strength}")

    output.append("\n📈 Areas for Improvement:")
    for area in evaluation.improvement_areas:
        output.append(f"  • {area}")

    output.append("\n🎯 Final Recommendation:")
    output.append(f"{evaluation.final_recommendation}")

    return "\n".join(output)


def print_comprehensive_report(evaluations: List[Dict]):
    """Display the complete evaluation results"""
    print("\n" + "=" * 80)
    print("📋 Comprehensive Resume Evaluation Report")
    print("=" * 80)

    for idx, eval_result in enumerate(evaluations, 1):
        job_info = eval_result["job_info"]
        evaluation = eval_result["evaluation"]

        print(f"\n{idx}. {job_info['company']} - {job_info['position']}")
        print(f"Recommendation Similarity Score: {job_info['similarity_score']:.2f}")
        print("-" * 50)
        print(eval_result["report"])
        print("\n" + "=" * 80)
```

Excute Evaluation

```python
# Resume file path
resume_path = "../data/joannadrummond-cv.pdf"

# Initialize evaluation system
evaluation_system = ResumeEvaluationSystem()

# First, get the resume text
resume_chunks = process_resume(resume_path)
resume_text = " ".join([chunk[0] for chunk in resume_chunks])

# Perform resume evaluation
print("Evaluating resume...")
evaluations = evaluation_system.evaluate_with_recommendations(
    resume_text,  # Pass the actual resume text instead of the path
    recommended_jobs=recommended_jobs,
    top_n=1,
)

# Print comprehensive report
print_comprehensive_report(evaluations)
```

<pre class="custom">Resume analysis completed.
    Number of extracted chunks: 7
    
    Career Analysis Summary:
    ------------------------
    Interests: Joanna Drummond's main areas of interest and research focus are in computer science, particularly in algorithms, artificial intelligence, and stable matching problems. Her research includes exploring strategy-proofness and preference elicitation in stable matching, as well as investigating multi-agent systems and decision-making under uncertainty. She has also worked on dialogue systems and educational applications, applying machine learning techniques to analyze student interactions and engagement.
    
    Recommended Roles:
    Evaluating resume...
    
    ================================================================================
    📋 Comprehensive Resume Evaluation Report
    ================================================================================
    
    1. Symbolica AI - Senior Machine Learning Research Engineer
    Recommendation Similarity Score: 0.77
    --------------------------------------------------
    
    📊 Resume Evaluation Report
    ==================================================
    
    💡 Overall Score: 70/100
    
    
    🔧 Technical Fit (30%)
    Score: 3/5
    Reasoning: The candidate has a strong background in computer science and machine learning, with proficiency in Python, which is relevant for the role. However, there is no explicit mention of experience with PyTorch or JAX, which are required skills for the position.
    Evidence Found:
      • Proficiency in Python
      • Research experience in machine learning and algorithms
    Suggestions:
      • Gain experience with PyTorch and JAX to better align with the job requirements.
      • Highlight any relevant projects or experiences involving distributed training or GPU optimization.
    
    👔 Experience Relevance (25%)
    Score: 3/5
    Reasoning: The candidate has extensive research experience, but most of it appears to be academic. There is limited evidence of non-academic machine learning engineering roles, which are crucial for this position.
    Evidence Found:
      • Research Assistant at University of Toronto since 2011
      • Research Intern at Microsoft Research in 2016
    Suggestions:
      • Include more details about any industry projects or collaborations.
      • Emphasize any practical applications of research in real-world scenarios.
    
    🎯 Industry Knowledge (15%)
    Score: 3/5
    Reasoning: The candidate has a solid understanding of algorithms and AI, but there is no specific mention of knowledge in large-scale AI systems or current trends in the industry.
    Evidence Found:
      • Research interests in algorithms and AI
      • PhD in Computer Science
    Suggestions:
      • Stay updated with the latest trends in large-scale AI and theorem proving.
      • Participate in industry conferences or workshops to gain more exposure.
    
    📚 Education Qualification (15%)
    Score: 5/5
    Reasoning: The candidate has a strong educational background with a PhD in Computer Science, which is highly relevant for the position.
    Evidence Found:
      • PhD in Computer Science from University of Toronto
      • M.S. in Computer Science from University of Toronto
    Suggestions:
      • Consider obtaining certifications in specific machine learning frameworks if applicable.
    
    🤝 Soft Skills (15%)
    Score: 3/5
    Reasoning: The resume does not provide much information on soft skills such as leadership, teamwork, or communication, which are important for collaboration in a research environment.
    Evidence Found:
      • Participation in program committees and as a reviewer
    Suggestions:
      • Include examples of teamwork or leadership roles in projects.
      • Highlight any presentations or publications to demonstrate communication skills.
    
    📋 Overall Evaluation
    ------------------------------
    
    💪 Key Strengths:
      • Strong educational background in computer science
      • Proficiency in Python and machine learning research
    
    📈 Areas for Improvement:
      • Experience with PyTorch and JAX
      • Non-academic machine learning engineering experience
      • Demonstration of soft skills
    
    🎯 Final Recommendation:
    The candidate has a strong academic background and relevant research experience, but needs to gain more practical experience with the specific tools and frameworks required for the role. Consideration for the position is recommended if the candidate can demonstrate proficiency in the required skills and provide evidence of industry experience.
    
    ================================================================================
</pre>

## LLM-Based Resume Revise System

This tutorial demonstrates how to create a system that evaluates and improves resumes using a **Large Language Model (LLM)**. 

The system provides actionable suggestions to optimize resumes for specific job descriptions, enhancing the candidate’s chances of securing a role.

---

Key Components

1. **EnhancementSuggestion Model**
The `EnhancementSuggestion` model defines the structure for improvement suggestions:
- **`section`**: The specific resume section being improved (e.g., "Skills" or "Work Experience").
- **`current_content`**: The original content of the section.
- **`improved_content`**: The suggested improvement for the section.
- **`explanation`**: A detailed explanation of why the improvement is recommended.

---

2. **ResumeEnhancement Model**
The `ResumeEnhancement` model provides a holistic improvement report:
- **`improvements`**: A list of section-specific suggestions.
- **`keyword_optimization`**: Suggested keywords to include in the resume for optimization.
- **`general_suggestions`**: Overall suggestions for structure and presentation.
- **`action_items`**: Practical, actionable items for the candidate to implement.

---

3. **ResumeEnhancementSystem**
The `ResumeEnhancementSystem` class uses an LLM to analyze resumes and generate detailed, job-specific improvement suggestions. This system:
- Accepts the resume text, job information, and evaluation results as inputs.
- Produces a structured output aligning with the `ResumeEnhancement` model.
- Focuses on realistic, actionable improvements tailored to the target job.

4. **IntegratedResumeSystem**
The `IntegratedResumeSystem` combines evaluation and enhancement processes into a seamless workflow:
- **Step 1**: The `ResumeEvaluationSystem` evaluates the resume against job requirements, providing initial scoring and feedback.
- **Step 2**: The `ResumeEnhancementSystem` builds upon the evaluation results to generate actionable suggestions for improvement.
- **Step 3**: A comprehensive improvement report is created, highlighting section-specific improvements, keyword optimizations, and general suggestions.

---
## Example Output Format

### **📋 Section-Specific Improvements**
- **Section**: Work Experience  
  **Current**: Managed team projects in retail operations.  
  **Improved**: Led cross-functional teams to increase sales by 15% within six months.  
  **Reason**: Emphasizes measurable outcomes and aligns with the leadership skills required for the target role.

### **🔍 Recommended Keywords**
- "Cross-functional leadership"
- "Revenue growth"
- "Data-driven decision-making"

### **💡 General Suggestions**
- Use consistent formatting for job titles and dates.  
- Highlight certifications relevant to the target job.

### **✅ Actionable Steps**
1. Update work experience details to emphasize achievements.  
2. Include certifications and training relevant to the role.  
3. Incorporate suggested keywords into the skills and summary sections.


```python
class EnhancementSuggestion(BaseModel):
    """Suggestions for improvement for each resume section"""

    section: str = Field(description="Resume section")
    current_content: str = Field(description="Current content")
    improved_content: str = Field(description="Suggested improvement")
    explanation: str = Field(description="Reason for the improvement and explanation")


class ResumeEnhancement(BaseModel):
    """Overall suggestions for resume improvement"""

    improvements: List[EnhancementSuggestion] = Field(
        description="Suggestions for each section"
    )
    keyword_optimization: List[str] = Field(description="Keywords to optimize")
    general_suggestions: List[str] = Field(description="General suggestions")
    action_items: List[str] = Field(description="Actionable items")


class ResumeEnhancementSystem:
    def __init__(self, model_name="gpt-4o", temperature=0.1):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=ResumeEnhancement)

        # Prompt template for generating improvement suggestions
        self.prompt_template = """You are a professional resume consultant.
        Based on the provided evaluation results, offer detailed and actionable suggestions for improving the resume.

        Current Resume:
        {resume_text}

        Evaluation Results:
        {evaluation_results}

        Job Information:
        {job_info}

        Please include the following considerations when making your suggestions:
        1. Specific improvement suggestions for each section
        2. Key job-related keywords
        3. General structural and expression improvements
        4. Short-term and long-term actionable items

        Pay particular attention to the following:
        - Emphasize areas with high scores
        - Provide concrete solutions for areas with low scores
        - Tailor suggestions to the characteristics of the job
        - Ensure realistic and actionable recommendations

        Provide the improvement suggestions in the following format:
        {format_instructions}
        """

        self.prompt = ChatPromptTemplate.from_template(
            template=self.prompt_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def generate_improvements(
        self, resume_text: str, evaluation_results: List[Dict], job_info: Dict
    ) -> ResumeEnhancement:
        """Generate improvement suggestions based on evaluation results.

        Args:
            resume_text (str): Original resume content
            evaluation_results (List[Dict]): Previous evaluation results
            job_info (Dict): Target job information

        Returns:
            ResumeEnhancement: Structured improvement suggestions including:
                - Section-specific improvements
                - Keyword optimizations
                - General suggestions
                - Actionable items

        Raises:
            Exception: If suggestion generation fails
        """
        try:
            # Serialize evaluation_results (if DetailedEvaluation objects are included)
            evaluation_data = [
                (
                    eval_result.model_dump()
                    if hasattr(eval_result, "model_dump")
                    else eval_result
                )
                for eval_result in evaluation_results
            ]

            messages = self.prompt.format_messages(
                resume_text=resume_text,
                evaluation_results=json.dumps(
                    evaluation_data, ensure_ascii=False, indent=2
                ),
                job_info=json.dumps(job_info, ensure_ascii=False, indent=2),
            )

            response = self.llm.invoke(messages)
            suggestions = self.parser.parse(response.content)
            return suggestions

        except Exception as e:
            print(f"Error while generating improvement suggestions: {str(e)}")
            raise


def format_enhancement_report(enhancement: ResumeEnhancement) -> str:
    """Format the improvement suggestions into a report"""
    output = []
    output.append("\n📝 Resume Improvement Report")
    output.append("=" * 50)

    # Section-specific suggestions
    output.append("\n📋 Section-Specific Improvements")
    output.append("-" * 30)
    for improvement in enhancement.improvements:
        output.append(f"\n[{improvement.section}]")
        output.append("Current:")
        output.append(f"  {improvement.current_content}")
        output.append("Improved:")
        output.append(f"  {improvement.improved_content}")
        output.append("Reason:")
        output.append(f"  {improvement.explanation}")

    # Keyword optimization
    output.append("\n🔍 Recommended Keywords")
    output.append("-" * 30)
    for keyword in enhancement.keyword_optimization:
        output.append(f"• {keyword}")

    # General suggestions
    output.append("\n💡 General Suggestions")
    output.append("-" * 30)
    for suggestion in enhancement.general_suggestions:
        output.append(f"• {suggestion}")

    # Action items
    output.append("\n✅ Actionable Steps")
    output.append("-" * 30)
    for item in enhancement.action_items:
        output.append(f"• {item}")

    return "\n".join(output)


class IntegratedResumeSystem:
    """A system combining evaluation and improvement"""

    def __init__(self):
        self.evaluation_system = ResumeEvaluationSystem()
        self.enhancement_system = ResumeEnhancementSystem()

    def analyze_and_improve(
        self, resume_path: str, recommended_jobs: List[dict], top_n: int = 3
    ):
        """Perform integrated resume evaluation and improvement analysis.

        Workflow:
        1. Process resume text from PDF
        2. Evaluate resume against top-n recommended jobs
        3. Generate improvement suggestions for each evaluation

        Args:
            resume_path (str): Path to resume PDF file
            recommended_jobs (List[dict]): List of potential job matches
            top_n (int): Number of top jobs to analyze

        Returns:
            List[Dict]: List of dictionaries containing:
                - job_info: Target job details
                - evaluation: Detailed evaluation results
                - enhancement: Improvement suggestions

        Raises:
            Exception: If analysis process fails
        """
        try:
            # First, process the resume to get the text content
            resume_chunks = process_resume(resume_path)
            resume_text = " ".join([chunk[0] for chunk in resume_chunks])

            # 1. Perform resume evaluation
            print("Evaluating the resume...")
            evaluations = self.evaluation_system.evaluate_with_recommendations(
                resume_text,  # Pass the processed text instead of path
                recommended_jobs=recommended_jobs,
                top_n=top_n,
            )

            # 2. Generate improvement suggestions for each recommended job
            print("Generating improvement suggestions...")
            improvements = []

            for eval_result in evaluations:
                job_info = eval_result["job_info"]
                evaluation = eval_result["evaluation"]

                # Generate improvement suggestions using the already processed resume text
                enhancement = self.enhancement_system.generate_improvements(
                    resume_text=resume_text,  # Use the processed text
                    evaluation_results=[evaluation.model_dump()],
                    job_info=job_info,
                )

                improvements.append(
                    {
                        "job_info": job_info,
                        "evaluation": evaluation,
                        "enhancement": enhancement,
                    }
                )

            return improvements

        except Exception as e:
            print(f"Error during analysis and improvement: {str(e)}")
            raise
```

Excute Evaluation

you can choose how many jobs you want to evaluate by changing the `top_n` value.

```python
# Resume file path
resume_path = "../data/joannadrummond-cv.pdf"

# Initialize the integrated system
system = IntegratedResumeSystem()

# Perform analysis and improvements
results = system.analyze_and_improve(
    resume_path=resume_path, recommended_jobs=recommended_jobs, top_n=3
)

# Display the results
for result in results:
    print(f"\nJob: {result['job_info']['position']} @ {result['job_info']['company']}")
    print("=" * 80)
    print("\n[Evaluation Results]")
    print(result["evaluation"])
    print("\n[Improvement Suggestions]")
    print(format_enhancement_report(result["enhancement"]))
    print("=" * 80)
```

<pre class="custom">Resume analysis completed.
    Number of extracted chunks: 7
    
    Career Analysis Summary:
    ------------------------
    Interests: Joanna Drummond's primary academic interests and research focus lie in the fields of computer science, particularly in algorithms, artificial intelligence, and game theory. Her research has extensively explored stable matching problems, preference elicitation, and decision-making under uncertainty, with applications in multi-agent systems and educational technologies. She has also investigated student engagement and dialogue systems, applying machine learning techniques to educational data.
    
    Recommended Roles:
    Evaluating the resume...
    Generating improvement suggestions...
    
    Job: Senior Machine Learning Research Engineer @ Symbolica AI
    ================================================================================
    
    [Evaluation Results]
    technical_fit=CriterionEvaluation(score=3, reasoning='The candidate has a strong background in computer science and AI, with experience in Python and other programming languages. However, there is no explicit mention of experience with PyTorch, JAX, or distributed training techniques, which are crucial for the role.', evidence=['Proficiency in Python, Java, Julia, R, MATLAB', 'Research experience in AI and algorithms'], suggestions=['Gain experience with PyTorch and JAX', 'Familiarize with distributed training techniques']) experience_relevance=CriterionEvaluation(score=3, reasoning='The candidate has extensive research experience, but most of it appears to be academic. There is limited evidence of non-academic machine learning engineering roles, which are important for this position.', evidence=['Research Assistant roles at University of Toronto and University of Pittsburgh', 'Research Intern at Microsoft Research'], suggestions=['Seek opportunities in industry roles related to machine learning engineering', 'Highlight any non-academic projects or collaborations']) industry_knowledge=CriterionEvaluation(score=3, reasoning="The candidate has a solid understanding of AI and algorithms, but there is limited evidence of specific domain expertise in large-scale AI or structured reasoning, which are key to Symbolica AI's mission.", evidence=['Research interests in algorithms, AI, and game theory', 'PhD in Computer Science with a focus on AI'], suggestions=['Engage with current trends in large-scale AI and structured reasoning', 'Participate in industry conferences or workshops']) education_qualification=CriterionEvaluation(score=5, reasoning='The candidate has a strong educational background with a PhD in Computer Science, which is highly relevant to the position.', evidence=['PhD in Computer Science from University of Toronto', 'MS in Computer Science from University of Toronto', 'BS in Computer Science and Mathematics from University of Pittsburgh'], suggestions=['Continue engaging in continuous learning through courses or certifications in relevant areas']) soft_skills=CriterionEvaluation(score=3, reasoning='The resume does not provide explicit evidence of soft skills such as leadership, teamwork, or communication, which are important for collaboration in a research team.', evidence=['Participation in program committees and as a reviewer'], suggestions=['Highlight experiences that demonstrate leadership and teamwork', 'Include examples of effective communication in research or project settings']) overall_score=70 key_strengths=['Strong educational background in computer science and AI', 'Extensive research experience in algorithms and AI'] improvement_areas=['Experience with PyTorch, JAX, and distributed training', 'Non-academic machine learning engineering experience', 'Demonstration of soft skills in leadership and teamwork'] final_recommendation='The candidate shows potential with a strong educational background and research experience. However, to be a better fit for the role, they should gain more industry experience and develop specific technical skills required for the position.'
    
    [Improvement Suggestions]
    
    📝 Resume Improvement Report
    ==================================================
    
    📋 Section-Specific Improvements
    ------------------------------
    
    [Technical Skills]
    Current:
      Python Java Julia R Matlab Unix Shell Scripting (bash) Linux Mac OSX Windows LATEX Weka
    Improved:
      Python, Java, Julia, R, MATLAB, Unix Shell Scripting (bash), Linux, Mac OSX, Windows, LATEX, Weka, PyTorch, JAX, Distributed Training Techniques
    Reason:
      Adding PyTorch, JAX, and distributed training techniques aligns with the job requirements and demonstrates the candidate's readiness for the role.
    
    [Experience]
    Current:
      Research Intern: Microsoft Research, with Ian Kash and Peter Key, May 2016 to August 2016. Investigated simple pricing for cloud computing.
    Improved:
      Research Intern: Microsoft Research, with Ian Kash and Peter Key, May 2016 to August 2016. Investigated simple pricing for cloud computing. Collaborated on a project that involved distributed systems and cloud-based solutions.
    Reason:
      Highlighting collaboration on distributed systems projects can demonstrate relevant experience for the role.
    
    [Experience]
    Current:
      Research Assistant: University of Toronto, Department of Computer Science, Dr. Craig Boutilier, August 2011 to December 2014; Dr. Allan Borodin and Dr. Kate Larson, January 2015 to Present.
    Improved:
      Research Assistant: University of Toronto, Department of Computer Science, Dr. Craig Boutilier, August 2011 to December 2014; Dr. Allan Borodin and Dr. Kate Larson, January 2015 to Present. Engaged in projects that required teamwork and leadership in multi-disciplinary research teams.
    Reason:
      Adding details about teamwork and leadership addresses the soft skills gap identified in the evaluation.
    
    [Education]
    Current:
      PhD Computer Science: University of Toronto, (expected) Spring 2017. Co-advisors: Allan Borodin, Kate Larson. Achieved Candidacy: Spring 2015. GPA: 3.83
    Improved:
      PhD Computer Science: University of Toronto, (expected) Spring 2017. Co-advisors: Allan Borodin, Kate Larson. Achieved Candidacy: Spring 2015. GPA: 3.83. Engaged in continuous learning through workshops and courses on large-scale AI and structured reasoning.
    Reason:
      Emphasizing continuous learning in relevant areas shows commitment to staying updated with industry trends.
    
    🔍 Recommended Keywords
    ------------------------------
    • PyTorch
    • JAX
    • Distributed Training
    • Large-scale AI
    • Structured Reasoning
    • Machine Learning Engineering
    
    💡 General Suggestions
    ------------------------------
    • Reorganize the resume to clearly separate academic and industry experiences.
    • Use bullet points for each role to highlight key achievements and responsibilities.
    • Include a summary section at the top to quickly convey key strengths and career objectives.
    
    ✅ Actionable Steps
    ------------------------------
    • Gain hands-on experience with PyTorch and JAX through online courses or projects.
    • Seek out industry collaborations or internships to gain non-academic machine learning engineering experience.
    • Attend industry conferences and workshops to enhance knowledge in large-scale AI and structured reasoning.
    • Develop a portfolio of projects that demonstrate technical skills and industry knowledge.
    ================================================================================
    
    Job: Field Office ISSM - Open Rank-RS-Albuquerque, NM @ Georgia Tech Research Institute
    ================================================================================
    
    [Evaluation Results]
    technical_fit=CriterionEvaluation(score=2, reasoning='The candidate has a strong background in computer science and programming languages, but lacks specific experience with the required security frameworks and certifications.', evidence=['Proficient in Python, Java, Julia, R, MATLAB, Unix Shell Scripting, Linux, Mac OSX, Windows.', 'Research experience in computer science and algorithms.'], suggestions=['Gain experience with security frameworks such as JSIG, RMF, ICD 503, NIST 800, NISPOM, and DAAPM.', 'Obtain relevant security certifications like CISSP or Security+.']) experience_relevance=CriterionEvaluation(score=2, reasoning="The candidate's experience is primarily academic and research-focused, with limited direct relevance to the ISSM role's responsibilities.", evidence=['Research Intern at Microsoft Research.', 'Research Assistant at University of Toronto and University of Pittsburgh.'], suggestions=['Seek practical experience in information security management or related roles.', 'Engage in projects that involve system security and incident response.']) industry_knowledge=CriterionEvaluation(score=2, reasoning='The candidate has a strong academic background but lacks exposure to the specific industry requirements and trends in cybersecurity.', evidence=['Research interests in algorithms, AI, and game theory.', 'Limited mention of cybersecurity industry exposure.'], suggestions=['Stay updated with cybersecurity trends and best practices.', 'Participate in industry conferences or workshops related to cybersecurity.']) education_qualification=CriterionEvaluation(score=4, reasoning='The candidate has a strong educational background in computer science, which is relevant to the position.', evidence=['PhD in Computer Science from University of Toronto.', 'MS in Computer Science from University of Toronto.', 'BS in Computer Science and Mathematics from University of Pittsburgh.'], suggestions=['Consider obtaining certifications relevant to cybersecurity to enhance qualifications.']) soft_skills=CriterionEvaluation(score=3, reasoning='The candidate has demonstrated teamwork and communication skills through research collaborations, but lacks specific evidence of leadership or problem-solving in a security context.', evidence=['Collaborated with multiple researchers and advisors.', 'Participated in program committees and as a reviewer.'], suggestions=['Develop leadership skills through leading projects or teams.', 'Enhance communication skills by presenting research findings at conferences.']) overall_score=55 key_strengths=['Strong educational background in computer science.', 'Proficiency in multiple programming languages.'] improvement_areas=['Lack of specific cybersecurity experience and certifications.', 'Limited exposure to industry trends and practices in cybersecurity.'] final_recommendation='The candidate shows potential due to their strong educational background but needs to gain relevant experience and certifications in cybersecurity to be a strong fit for the ISSM role.'
    
    [Improvement Suggestions]
    
    📝 Resume Improvement Report
    ==================================================
    
    📋 Section-Specific Improvements
    ------------------------------
    
    [Technical Skills]
    Current:
      Python Java Julia R Matlab Unix Shell Scripting (bash) Linux Mac OSX Windows LATEX Weka
    Improved:
      Python, Java, Julia, R, MATLAB, Unix Shell Scripting (bash), Linux, Mac OSX, Windows, LATEX, Weka, JSIG, RMF, ICD 503, NIST 800, NISPOM, DAAPM
    Reason:
      Adding relevant security frameworks and standards will align the resume with the job requirements and demonstrate a commitment to gaining necessary cybersecurity knowledge.
    
    [Experience]
    Current:
      Research Intern: Microsoft Research, with Ian Kash and Peter Key, May 2016 to August 2016. Investigated simple pricing for cloud computing.
    Improved:
      Research Intern: Microsoft Research, with Ian Kash and Peter Key, May 2016 to August 2016. Investigated simple pricing for cloud computing. Participated in security-related projects to understand cloud security protocols.
    Reason:
      Highlighting any involvement in security-related projects, even if minor, can help bridge the gap between current experience and the job requirements.
    
    [Education]
    Current:
      PhD Computer Science: University of Toronto, (expected) Spring 2017. Co-advisors: Allan Borodin, Kate Larson. Achieved Candidacy: Spring 2015. GPA: 3.83
    Improved:
      PhD Computer Science: University of Toronto, (expected) Spring 2017. Co-advisors: Allan Borodin, Kate Larson. Achieved Candidacy: Spring 2015. GPA: 3.83. Relevant coursework: Information Security, Network Security.
    Reason:
      Including relevant coursework can demonstrate foundational knowledge in cybersecurity, which is crucial for the ISSM role.
    
    [Awards and Honors]
    Current:
      Microsoft Research PhD Fellowship Program Finalist, 2016 Reviewer, Algorithmica, 2015 Reviewer, SAGT 2015 Reviewer, AAAI-15 Ontario Graduate Scholarship, 2014 Reviewer, COMSOC-2014 Microsoft Research Graduate Women’s Scholarship Recipient, 2012 Google Anita Borg Memorial Scholarship Finalist, 2012 Ontario Graduate Scholarship, 2012 Awardee of 2011 NSF Graduate Research Fellowship Program DREU Recipient, Chosen for Distributed Research Experience for Undergraduates Program Best Undergraduate Poster, University of Pittsburgh Department of Computer Science 10th Annual Computer Science Day
    Improved:
      Microsoft Research PhD Fellowship Program Finalist, 2016. Reviewer, Algorithmica, 2015. Reviewer, SAGT 2015. Reviewer, AAAI-15. Ontario Graduate Scholarship, 2014. Reviewer, COMSOC-2014. Microsoft Research Graduate Women’s Scholarship Recipient, 2012. Google Anita Borg Memorial Scholarship Finalist, 2012. Ontario Graduate Scholarship, 2012. Awardee of 2011 NSF Graduate Research Fellowship Program. DREU Recipient, Chosen for Distributed Research Experience for Undergraduates Program. Best Undergraduate Poster, University of Pittsburgh Department of Computer Science 10th Annual Computer Science Day. Consider pursuing cybersecurity-related awards or recognitions.
    Reason:
      While the current awards demonstrate academic excellence, pursuing cybersecurity-related awards can enhance the resume's relevance to the ISSM role.
    
    🔍 Recommended Keywords
    ------------------------------
    • Information Security
    • Cybersecurity
    • Security Frameworks
    • Incident Response
    • System Vulnerabilities
    • Risk Mitigation
    • Security Certifications
    
    💡 General Suggestions
    ------------------------------
    • Reformat the resume to include clear sections with headings such as 'Technical Skills', 'Experience', 'Education', 'Awards', and 'Research Interests'.
    • Use bullet points for listing skills and experiences to improve readability.
    • Include a summary or objective statement at the beginning of the resume to highlight key strengths and career goals related to cybersecurity.
    
    ✅ Actionable Steps
    ------------------------------
    • Short-term: Enroll in courses or workshops related to cybersecurity frameworks and certifications.
    • Short-term: Attend cybersecurity conferences or webinars to gain industry knowledge.
    • Long-term: Obtain certifications such as CISSP or Security+ to enhance qualifications.
    • Long-term: Seek opportunities to gain practical experience in cybersecurity roles or projects.
    ================================================================================
    
    Job: Field Office ISSM - Open Rank-RS-Albuquerque, NM @ Georgia Tech Research Institute
    ================================================================================
    
    [Evaluation Results]
    technical_fit=CriterionEvaluation(score=2, reasoning='The candidate has a strong background in computer science and programming languages, but lacks specific experience with the required security frameworks and certifications such as CISSP, Security+, JSIG, RMF, ICD 503, NIST 800, NISPOM, and DAAPM.', evidence=['Proficient in Python, Java, Julia, R, Matlab, Unix Shell Scripting, Linux, Mac OSX, Windows.', 'Research experience in algorithms, AI, and game theory.'], suggestions=['Gain certifications like CISSP or Security+.', 'Familiarize with security frameworks such as NIST 800 and RMF.']) experience_relevance=CriterionEvaluation(score=2, reasoning='The candidate has extensive research experience but lacks direct experience in roles similar to the ISSM position, particularly in managing classified information systems and security compliance.', evidence=['Research Intern at Microsoft Research.', 'Research Assistant at University of Toronto and University of Pittsburgh.'], suggestions=['Seek roles or projects that involve information security management.', 'Gain experience in system accreditation and authorization processes.']) industry_knowledge=CriterionEvaluation(score=2, reasoning="The candidate's industry knowledge is primarily academic and research-focused, with limited exposure to cybersecurity industry trends and practices.", evidence=['Research interests in algorithms, AI, and game theory.', 'No mention of cybersecurity industry exposure.'], suggestions=['Engage with cybersecurity industry events and publications.', 'Network with professionals in the cybersecurity field.']) education_qualification=CriterionEvaluation(score=4, reasoning='The candidate has strong educational qualifications with a PhD in Computer Science and a high GPA, which aligns well with the technical requirements of the position.', evidence=['PhD in Computer Science from University of Toronto.', 'MS in Computer Science with a GPA of 3.93.'], suggestions=['Consider additional certifications in information security.']) soft_skills=CriterionEvaluation(score=3, reasoning='The candidate has demonstrated communication and teamwork skills through research collaborations and committee participation, but lacks evidence of leadership in security-focused environments.', evidence=['Program Committee member for CoopMAS 2017.', 'Reviewer for various academic conferences.'], suggestions=['Develop leadership skills in security-focused projects.', 'Engage in roles that require security policy communication.']) overall_score=55 key_strengths=['Strong educational background in computer science.', 'Proficiency in multiple programming languages.'] improvement_areas=['Lack of specific cybersecurity certifications and experience.', 'Limited exposure to industry-specific security practices.'] final_recommendation='The candidate shows potential due to their strong educational background and technical skills. However, they need to gain relevant cybersecurity experience and certifications to be a strong fit for the ISSM position.'
    
    [Improvement Suggestions]
    
    📝 Resume Improvement Report
    ==================================================
    
    📋 Section-Specific Improvements
    ------------------------------
    
    [Technical Skills]
    Current:
      Python Java Julia R Matlab Unix Shell Scripting (bash) Linux Mac OSX Windows LATEX Weka
    Improved:
      Python, Java, Julia, R, Matlab, Unix Shell Scripting (bash), Linux, Mac OSX, Windows, LATEX, Weka, NIST 800, RMF, CISSP, Security+
    Reason:
      Adding relevant cybersecurity frameworks and certifications will align the technical skills with the job requirements for the ISSM position.
    
    [Experience]
    Current:
      Research Intern: Microsoft Research, with Ian Kash and Peter Key, May 2016 to August 2016. Investigated simple pricing for cloud computing.
    Improved:
      Research Intern: Microsoft Research, with Ian Kash and Peter Key, May 2016 to August 2016. Investigated simple pricing for cloud computing. Gained exposure to cloud security protocols and compliance measures.
    Reason:
      Highlighting any exposure to security protocols during past roles can help bridge the gap between current experience and the job requirements.
    
    [Research Interests and Focus Areas]
    Current:
      Joanna Drummond's primary academic interests and research focus lie in the fields of computer science, particularly in algorithms, artificial intelligence, and game theory.
    Improved:
      Joanna Drummond's primary academic interests and research focus lie in the fields of computer science, particularly in algorithms, artificial intelligence, game theory, and cybersecurity frameworks.
    Reason:
      Including cybersecurity frameworks in the research interests can demonstrate a broader interest in the field relevant to the ISSM position.
    
    [Education]
    Current:
      PhD Computer Science: University of Toronto, (expected) Spring 2017. Co-advisors: Allan Borodin, Kate Larson. Achieved Candidacy: Spring 2015. GPA: 3.83
    Improved:
      PhD Computer Science: University of Toronto, (expected) Spring 2017. Co-advisors: Allan Borodin, Kate Larson. Achieved Candidacy: Spring 2015. GPA: 3.83. Consider pursuing certifications like CISSP or Security+ to complement the academic background.
    Reason:
      Suggesting additional certifications can enhance the educational qualifications to better fit the job requirements.
    
    🔍 Recommended Keywords
    ------------------------------
    • CISSP
    • Security+
    • NIST 800
    • RMF
    • cybersecurity
    • information security
    • classified systems
    • system accreditation
    • security compliance
    
    💡 General Suggestions
    ------------------------------
    • Reorganize the resume to clearly separate technical skills, experience, education, and certifications.
    • Use bullet points for each role to highlight key achievements and responsibilities.
    • Include a summary section at the top to quickly convey the candidate's strengths and career goals.
    
    ✅ Actionable Steps
    ------------------------------
    • Enroll in CISSP or Security+ certification courses to gain relevant credentials.
    • Attend cybersecurity industry events and webinars to build industry knowledge.
    • Network with professionals in the cybersecurity field to gain insights and potential mentorship.
    • Seek volunteer or part-time roles in information security to gain practical experience.
    ================================================================================
</pre>

```python

```
