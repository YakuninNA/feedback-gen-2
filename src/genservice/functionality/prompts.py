qa_extraction_prompt_template = """
You task is to process a technical interview transcript.

**Rules:**
1. Extract all questions posed by Interviewer, consolidate questions that are logically connected into one;;
2. Pair each question with its corresponding answers by Candidate, consolidate answers that are logically connected and pertain to the same question;
3. Explicitly note if the Candidate hesitated, was unsure of the answer, or responded only after receiving a hint;
4. **Do not fabricate any information**, be strict;
5. **Do not exclude any information from the technical interview transcript;**
6. **Do not include any additional information beyond what is provided;**
7. **Ensure the Interviewer entity only asks questions;**
8. **Ensure the Candidate entity only provides answers;**
9. **Do not mix information from questions with answers.**

**Output Format:**
Output must be valid JSON format with double quotes around keys and string values, ensure proper escaping:

[
  {{
    "Interviewer": [Interviewer's question here]",
    "Candidate": [Candidate's answer here]
  }}
]   
...

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_extraction}

**Subject for processing:**
Here is the technical interview transcript itself:
{processed_data}
"""

qa_polish_prompt_template = """
You task is to process a technical interview transcript.

**Rules:**
1. **Extract all relevant information from the Candidate's answers to the corresponding Interviewer's questions;**
2. Ensure the essence and specifics of the Candidate's answers are included, even if the answer is inaccurate;
3. Provide the Candidate's answers as direct citations with no AI interference;
4. **Explicitly mention if the Candidate hesitated, was unsure of the answer, or responded only after receiving a hint;**
5. Ensure each Candidate's answer is comprehensive yet no longer than 150 words;
6. **Do not include any additional information beyond what is provided;**
7. Avoid using any advanced vocabulary; speak in simple, human words;
8. **Anonymize all content; avoid using proper names.**

**Output Format:**
Output must be valid JSON format with double quotes around keys and string values, ensure proper escaping:

[
  {{
    "Interviewer": [Interviewer's question here]",
    "Candidate": [Candidate's answer here]
  }}
]   
...

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_polish}

**Subject for processing:**
Here is the technical interview transcript itself:
{qa_extracted_data}
"""

qa_categorization_prompt_template = """
You task is to map and categorize a technical interview transcript against a set of categories.

**Rules:**
1. Each Interviewer's question and Candidate's answer pair can be allocated to only one category;
2. **Use the provided category names exactly as given; do not alter their names in any way;**
3. **Do not create new categories; use only those provided.**

**Output Format:**
Output must be valid JSON format with double quotes around keys and string values, ensure proper escaping:

[
  {{
    "Interviewer": [Interviewer's question here]",
    "Candidate": [Candidate's answer here]
    "Category": [Category's name here]
  }}
]   
...

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_categorization}

**Subjects for processing:**
Here are the categories:
{categories}
Here is the technical interview transcript itself:
{qa_polished_data}
"""

requirements_extraction_prompt_template = """
You task is to process input data into strictly formatted set of requirements.
 
**Rules:**
1. Process each requirement sequentially, line by line;
2. **Include only the names of the requirements;**
3. **Do not include any additional names or symbols;**
4. If a requirement specifies a particular framework, generalize it instead of enumerating;
5. **Avoid adding any AI-generated remarks or additional information;**
6. Consolidate requirements with similar topics;
7. **Do not use numbering or bullet points of any kind;**
8. Ensure each requirement is on a separate line.

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_requirements}

**Subject for processing:**
Here is the input data:
{requirements}
"""

engineering_basics_extraction_prompt_template = """
You task is to process a technical interview transcript.

**Rules:**
1. Ensure each section is no longer than 200 words;
2. **Provide a comprehensive summary of the answer using ';' as a separator;**
3. **Avoid adding any AI-generated remarks or additional information;**
4. **Be strict yet fair; do not sugarcoat;**
5. Avoid using any advanced vocabulary; speak in simple words;
6. **Ensure all names (company, personal names, etc.) are anonymized.**

**Output Format:**
---
### Team Size:
 - [Extracted Overview of the candidate's experience working in teams of different sizes, 3-5 sentences]
### Git Experience, Git Flow Organization, CI/CD:
 - [Extracted Bullet 1] - candidate's Git experience, 2-3 sentences
 - [Extracted Bullet 2] - candidate's Git Flow experience (how it was organized in their team)
 - [Extracted Bullet 3] - candidate's experience with CI/CD
### Experience with Agile/Scrum/Kanban:
 - [Extracted Overview of the candidate's experience working with different methodologies, 1-3 sentences]
---

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_engineering_basics}

**Subject for processing:**
Here is the technical interview transcript itself:
{soft_categorized_answers}
"""

technical_skills_extraction_prompt_template = """
You task is to process a technical interview transcript.

**Rules:**
1. Make a comprehensive summary of the Candidate's expertise for each category in {tech_requirements};
2. The summary should be factologial, meaning it should lack any self-imposed evaluation;
3. **If a category in {tech_requirements} does not appear examined data, return [No questions asked];**
4. **Do not use any additional information beyond what is provided in the examined data;**
5. **Do not add any additional sections besides the categories provided in {tech_requirements};**
6. Avoid using advanced vocabulary; speak in simple terms.

**Output Format:**
---
### Category 1:
 - [Insert Summary Here]
### Category 2:
 - [Insert Summary Here]
### Category 3:
 - [Insert Summary Here]   
---

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_technical_skills}

**Subject for processing:**
Here is the technical interview transcript itself:
{tech_categorized_answers}
"""

experience_extraction_prompt_template = """
You task is to process a technical interview transcript.

**Rules:**
1. Ensure each section is no longer than 200 words;
2. **Provide a comprehensive summary of the answer using ';' as a separator;**
3. **Avoid adding any AI-generated remarks or additional information;**
4. **Be strict yet fair; do not sugarcoat;**
5. Avoid using any advanced vocabulary; speak in simple words;
6. **Ensure all names (company, personal names, etc.) are anonymized.**

**Output Format:**
---
1. Experience:
### Introduction:
 - [Extracted Answer] - 3-5 sentences, must be tailored towards **{position_name}** position    

### Project Description:
 - [Extracted Answer] - 3-5 sentences, must be tailored towards **{position_name}** position
---
2. Product Area Impact:
### Main Responsibilities:
 - [Extracted Bullet 1] - must be tailored towards **{position_name}** position
 - [Extracted Bullet 2] - must be tailored towards **{position_name}** position
 - [Extracted Bullet 3] - must be tailored towards **{position_name}** position
 - [Extracted Bullet 4] - must be tailored towards **{position_name}** position
 - [Extracted Bullet 5] - must be tailored towards **{position_name}** position

### Candidate's Position and Main Stack of Technologies:
 - [Extracted Answer] - must be tailored towards **{position_name}** position    
---
3. Independency:
### Previous Experience/Main Responsibilities (Conclusion):
 - [Extracted Overview of 1 and 2, indicator of the candidate's independency, 2-3 sentences]
---
4. Proactivity:
### Previous Experience/Main Responsibilities (Conclusion):
 - [Extracted Overview of 1 and 2, indicator of the candidate's proactivity, 2-3 sentences]
---
5. Mentoring:
### Previous Mentoring Experience:
 - [Extracted Answer]
---

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_experience}

**Subject for processing:**
Here is the technical interview transcript itself:
{soft_categorized_answers}
"""

candidate_name_extraction_prompt_template = """
You task is to process input data into strictly formatted document name.

**Rules:**
1. Extract only candidate's name and surname;
2. If no such data is can be extracted, set default name - NO_NAME_SURNAME.

**Few-shot Example:**
Below is an mock example. Do not quote this example in your response. Instead, follow its structure:
{few_shot_candidate_name}

**Subject for processing:**
Here is the input data:
{transcript_name}
"""
