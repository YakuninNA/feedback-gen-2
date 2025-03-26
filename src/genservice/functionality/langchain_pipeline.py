import json

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnableLambda,
    RunnableMap,
    RunnableSequence,
    RunnableParallel
)

from genservice.functionality.prompts import (
    qa_extraction_prompt_template,
    qa_polish_prompt_template,
    qa_categorization_prompt_template,
    requirements_extraction_prompt_template,
    engineering_basics_extraction_prompt_template,
    technical_skills_extraction_prompt_template,
    experience_extraction_prompt_template,
    candidate_name_extraction_prompt_template
)
from genservice.functionality.system_messages import (
    QA_EXTRACTION_SYSTEM_MESSAGE,
    QA_POLISH_SYSTEM_MESSAGE,
    QA_CATEGORIZATION_SYSTEM_MESSAGE,
    REQUIREMENTS_SYSTEM_MESSAGE,
    EXPERIENCE_SYSTEM_MESSAGE,
    ENGINEERING_BASICS_SYSTEM_MESSAGE,
    TECHNICAL_SKILLS_SYSTEM_MESSAGE,
    CANDIDATE_NAME_EXTRACTION_SYSTEM_MESSAGE
)
from genservice.functionality.utility import (
    add_timestamp,
    extract_requirements,
    validate_json,
    process_categorized_answers,
    debug_step
)

from config import API_KEY

llm_mini = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=1,
    openai_api_key=API_KEY
)

llm_maxi = ChatOpenAI(
    model_name="gpt-4o-2024-08-06",
    temperature=0.85,
    top_p=0.25,
    openai_api_key=API_KEY
)

# initialize system message
interviewer_name_extraction_system_message = SystemMessagePromptTemplate.from_template(
    CANDIDATE_NAME_EXTRACTION_SYSTEM_MESSAGE
)
requirements_extraction_system_message = SystemMessagePromptTemplate.from_template(
    REQUIREMENTS_SYSTEM_MESSAGE
)
qa_extraction_system_message = SystemMessagePromptTemplate.from_template(
    QA_EXTRACTION_SYSTEM_MESSAGE
)
qa_polish_system_message = SystemMessagePromptTemplate.from_template(
    QA_POLISH_SYSTEM_MESSAGE
)
qa_categorization_system_message = SystemMessagePromptTemplate.from_template(
    QA_CATEGORIZATION_SYSTEM_MESSAGE
)
experience_extraction_system_message = SystemMessagePromptTemplate.from_template(
    EXPERIENCE_SYSTEM_MESSAGE
)
engineering_basics_extraction_system_message = SystemMessagePromptTemplate.from_template(
    ENGINEERING_BASICS_SYSTEM_MESSAGE
)
technical_skills_extraction_system_message = SystemMessagePromptTemplate.from_template(
    TECHNICAL_SKILLS_SYSTEM_MESSAGE
)


# initialize prompts
interviewer_name_extraction_prompt = HumanMessagePromptTemplate.from_template(
    candidate_name_extraction_prompt_template
)
requirements_extraction_prompt = HumanMessagePromptTemplate.from_template(
    requirements_extraction_prompt_template
)
qa_extraction_user_prompt = HumanMessagePromptTemplate.from_template(
    qa_extraction_prompt_template
)
qa_polish_user_prompt = HumanMessagePromptTemplate.from_template(
    qa_polish_prompt_template
)
qa_categorization_user_prompt = HumanMessagePromptTemplate.from_template(
    qa_categorization_prompt_template
)
experience_extraction_prompt = HumanMessagePromptTemplate.from_template(
    experience_extraction_prompt_template
)
engineering_basics_extraction_prompt = HumanMessagePromptTemplate.from_template(
    engineering_basics_extraction_prompt_template
)
technical_skills_extraction_prompt = HumanMessagePromptTemplate.from_template(
    technical_skills_extraction_prompt_template
)


# initialize chat prompts
interviewer_name_extraction_extraction_chat_prompt = ChatPromptTemplate.from_messages([
    interviewer_name_extraction_system_message,
    interviewer_name_extraction_prompt
])

requirements_extraction_chat_prompt = ChatPromptTemplate.from_messages([
    requirements_extraction_system_message,
    requirements_extraction_prompt
])

qa_extraction_chat_prompt = ChatPromptTemplate.from_messages([
    qa_extraction_system_message,
    qa_extraction_user_prompt
])

qa_polish_chat_prompt = ChatPromptTemplate.from_messages([
    qa_polish_system_message,
    qa_polish_user_prompt
])

qa_categorization_chat_prompt = ChatPromptTemplate.from_messages([
    qa_categorization_system_message,
    qa_categorization_user_prompt
])

experience_extraction_chat_prompt = ChatPromptTemplate.from_messages([
    experience_extraction_system_message,
    experience_extraction_prompt
])

engineering_basics_chat_prompt = ChatPromptTemplate.from_messages([
    engineering_basics_extraction_system_message,
    engineering_basics_extraction_prompt
])

technical_skills_chat_prompt = ChatPromptTemplate.from_messages([
    technical_skills_extraction_system_message,
    technical_skills_extraction_prompt
])


# initialize llm chains
interviewer_name_extraction_chain = interviewer_name_extraction_extraction_chat_prompt | llm_mini
requirements_extraction_chain = requirements_extraction_chat_prompt | llm_mini
qa_extraction_chain = qa_extraction_chat_prompt | llm_maxi
qa_polish_chain = qa_polish_chat_prompt | llm_maxi
qa_categorization_chain = qa_categorization_chat_prompt | llm_maxi
experience_extraction_chain = experience_extraction_chat_prompt | llm_mini
engineering_basics_chain = engineering_basics_chat_prompt | llm_mini
technical_skills_chain = technical_skills_chat_prompt | llm_maxi


# function for parsing interviewer name into a coherent format
async def parse_candidate_name(
    few_shot_candidate_name: str,
    transcript_name: str,
    timestamp: str
):
    name_parsing_chain = RunnableSequence(
        interviewer_name_extraction_chain,
        RunnableLambda(lambda output: add_timestamp(
            output.content,
            timestamp))
    )

    result = await name_parsing_chain.ainvoke({
        'transcript_name': transcript_name,
        'few_shot_candidate_name': few_shot_candidate_name
    })

    return result


# function for parsing requirements into a coherent format
async def parse_technical_requirements(
    requirements_text: str,
    general_requirements: list,
    few_shot_requirements: str
):
    reqs_parsing_chain = RunnableSequence(
        requirements_extraction_chain,
        RunnableLambda(lambda output: extract_requirements(
            output.content,
            general_requirements))
    )

    result = await reqs_parsing_chain.ainvoke({
        'requirements': requirements_text,
        'few_shot_requirements': few_shot_requirements
    })

    return result


# function for parsing input JSON into a categorized list of questions and answers
async def run_qa_extraction_pipeline(
    processed_data: dict,
    tech_requirements: list,
    general_requirements: list,
    categories: list,
    few_shot_extraction: str,
    few_shot_polish: str,
    few_shot_categorization: str
):
    parse_extracted_answers_chain = RunnableMap({
        "qa_extracted_data": lambda output: validate_json(output.content),
        "few_shot_polish": lambda _: few_shot_polish
    })

    parse_polished_answers_chain = RunnableMap({
        "qa_polished_data": lambda output: validate_json(output.content),
        "few_shot_categorization": lambda _: few_shot_categorization,
        "categories": lambda _: categories,
    })

    parse_categorized_answers_chain = RunnableLambda(
        lambda output: process_categorized_answers({
            'categorized_answers': output.content,
            'tech_requirements': tech_requirements,
            'general_requirements': general_requirements
        })
    )

    sequential_chain = RunnableSequence(
        qa_extraction_chain,
        debug_step("After qa_extraction_chain"),
        parse_extracted_answers_chain,
        debug_step("After parse_extracted_answers_chain"),
        qa_polish_chain,
        debug_step("After qa_polish_chain"),
        parse_polished_answers_chain,
        debug_step("After parse_polished_answers_chain"),
        qa_categorization_chain,
        debug_step("After qa_categorization_chain"),
        parse_categorized_answers_chain
    )

    inputs = {
        "processed_data": json.dumps(processed_data),
        "few_shot_extraction": few_shot_extraction
    }

    result = await sequential_chain.ainvoke(inputs)
    return result


# function for concurrently parsing categorized data and distribute it into feedback sections
async def run_ti_sections_extraction_pipeline(
    soft_categorized_answers: list,
    tech_categorized_answers: list,
    position_name: str,
    tech_requirements: list,
    few_shot_experience: str,
    few_shot_engineering_basics: str,
    few_shot_technical_skills: str
):
    prepare_parallel_inputs = RunnableLambda(
        lambda output: {
            'soft_categorized_answers': soft_categorized_answers,
            'tech_categorized_answers': tech_categorized_answers,
            'position_name': position_name,
            'tech_requirements': tech_requirements,
            'few_shot_experience': few_shot_experience,
            'few_shot_engineering_basics': few_shot_engineering_basics,
            'few_shot_technical_skills': few_shot_technical_skills
        }
    )

    parallel_chain = RunnableParallel({
        "experience_extraction_result": RunnableSequence(
            RunnableMap({
                "soft_categorized_answers": lambda x: x['soft_categorized_answers'],
                "position_name": lambda x: x['position_name'],
                "few_shot_experience": lambda x: x['few_shot_experience']
            }),
            experience_extraction_chain,
        ),
        "engineering_basics_result": RunnableSequence(
            RunnableMap({
                "soft_categorized_answers": lambda x: x['soft_categorized_answers'],
                "few_shot_engineering_basics": lambda x: x['few_shot_engineering_basics']
            }),
            engineering_basics_chain,
        ),
        "technical_skills_result": RunnableSequence(
            RunnableMap({
                "tech_categorized_answers": lambda x: x['tech_categorized_answers'],
                "tech_requirements": lambda x: ', '.join(x['tech_requirements']),
                "few_shot_technical_skills": lambda x: x["few_shot_technical_skills"]
            }),
            technical_skills_chain,
        )
    })

    parallel_sequence = RunnableSequence(
        prepare_parallel_inputs,
        parallel_chain
    )

    result = await parallel_sequence.ainvoke({})

    experience_result = result['experience_extraction_result'].content
    engineering_result = result['engineering_basics_result'].content
    technical_skills_result = result['technical_skills_result'].content

    return {
        'experience_result': experience_result,
        'engineering_result': engineering_result,
        'technical_skills_result': technical_skills_result
    }
