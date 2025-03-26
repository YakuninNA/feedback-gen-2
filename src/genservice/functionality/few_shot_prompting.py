few_shot_extraction = """
[
  {{
    "Interviewer": "What is your experience with cloud platforms?",
    "Candidate": "I have worked with AWS and Azure, mainly using EC2 for virtual machines and S3 for storage."
  }}
]
"""

few_shot_polish = """
[
  {{
    "Interviewer": "What is the general experience with AWS services?",
    "Candidate": "Experience includes setting up EC2 instances, RDS, and ECS containers with Fargate; familiar with Terraform and EKS Kubernetes clusters; organized applications into different subnets across multiple accounts for development, production, and staging."
  }}
]
"""

few_shot_categorization = """
[
  {{
    "Interviewer": "What is your introduction and experience in technology?",
    "Candidate": "I started working in IT from university, where I was promoted to computer classes administrator. I loved console and Linux, which led to an Internet administrator role. I transitioned into development, working with various technologies like front-end frameworks and back-end languages. I also have experience in mobile development and DevOps, configuring servers and CI/CD pipelines. Currently, I work in a small outsourced company, handling Kubernetes and cloud configurations, but I lack experience with large-scale applications.",
    "Category": "Introduction"
  }}
]
"""

few_shot_requirements = """
Example Input:
1. Python, Pytest;
2. Asyncio Experience;
3. AWS Cloud Computing;
4. Experience with Terraform.

Example Output:
Python, PyTest, Asyncio
AWS
Terraform
"""

few_shot_experience = """
Example Output:
---
1. Experience:
### Introduction:
 - Software development engineer with more than three years of experience; specialize in software development and distributed systems; worked on projects with different approaches and sizes of teams.

### Project Description:
 - The first project involved medical reports at Microsoft, utilizing the win form.NET framework;
the second project was a Warehouse application using WPF and MVVM patterns, along with Entity Framework and MSSQL;
another project was for an American government company, involving registration, marriages, debts, and businesses.
---
2. Product Area Impact:
### Main Responsibilities:
 - Translating business requirements into technical tasks;
 - Designing, planning, and prioritizing;
 - Code implementation;
 - Quality assurance;
 - Presenting the final product for evaluation.

### Candidate's Position and Main Stack of Technologies:
 - Works as a Software Development Engineer now. Stack of technologies includes **[Names of Technologies]**.
---
3. Independency:
### Previous Experience/Main Responsibilities (Conclusion):
 - While describing their previous experience, the candidate mentioned at least two projects where they had to take full release cycle responsibilities: from discussing business requirements with stakeholders, to translating them into technical tasks, planning, implementing, and presenting the final product for evaluation.
---
4. Proactivity:
### Previous Experience/Main Responsibilities (Conclusion):
 - The candidate explained that their professional experience is tightly connected with projects of flexible and dynamic requirements. As a result, their designing workflow evolved to predicting (or at least presupposing) extensibility of their codebase, keeping the number of non-standard solutions to the bare minimum to avoid unnecessary customization.
---
5. Mentoring:
### Previous Mentoring Experience:
 - The candidate has no significant mentoring experience.
---
"""


few_shot_engineering_basics = """
Example Output:
---
### Team Size:
 - Experience working with teams of different sizes, from 6 to 25 people; now works in a cross-functional
team of 10 people, 5 backend and 5 frontend engineers.

### Git Experience, Git Flow Organization, CI/CD:
 - Current Git flow depends on project size; main branch is protected, with feature branches for development; developers create branches, work on features, and submit pull requests for merging.
 - Worked extensively with Git; primarily use Visual Studio for .NET development and Git Bash for specific commands; prefer to merge to maintain branch hierarchy; understand Azure DevOps and have used it to create pipelines; prefer classic UI editor for ease of use.
 - CI involves building artifacts like Docker images and running tests; CD is divided into delivery and deployment, with tools like Jenkins and GitHub Actions used for CI.

### Experience with Agile/Scrum/Kanban:
 - Used Scrum for about ten months with daily standups and two-week sprints; switched to a Kanban, due to a lot of 
cross-functional tasks between different teams; easier to track using Kanban Board.  
---         
"""


few_shot_technical_skills = """
Example Output:
---
### Python:
 - The candidate has a good practical experience both in implementing a modern asynchronous services (Flask, asyncio, 
noSQL DBs such as Elasticsearch and MongoDB, and Ansible) and in maintaining and improving some really old legacy 
codebase (Python 2 with PostgreSQL). The candidate lacks some experience in async theory.

### Linux:
 - Linux:
The candidate uses Linux as their main development platform, and he uses Linux inside their service containers as well. 
THey proved his familiarity with Linux CLI by naming a list of his usual Bash commands: docker, git, ssh, grep, and 
also a bunch of commands from the coreutils package.        
---
"""


few_shot_candidate_name = """
Example Input:
7ZjFhGaFkuPaTbj2_Andrew-Ermeneu-Interview-c6666aa6-7f38-ef11-8409-000d3ab454a8-2024-07-08-13-02-GMT-2-mp4-st

Example Output:
Andrew-Ermeneu_
"""