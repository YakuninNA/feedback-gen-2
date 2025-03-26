import json
import markdown
import re

from json_repair import repair_json
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


# JSON to dialogue conversion
def convert_json_to_dialogue(json_string):
    # Check if the input is valid JSON, if not, throw an error
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError("Input is not valid JSON") from e

    if not isinstance(data, list):
        raise ValueError("JSON data is not a list of dialogue entries")

    # Get rid of 'speaker_id' and 'timestamps'
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item in the JSON array must be a dictionary")
        item.pop('speaker_id', None)
        item.pop('timestamp', None)

    # Rename speakers into 'Interviewer' and 'Candidate'
    # Identify speakers
    speakers = set(item.get('speaker_name', 'Unknown') for item in data)

    # Assign 'Interviewer' to the speaker who asks the most questions
    question_counts = {speaker: 0 for speaker in speakers}

    for item in data:
        sentence = item.get('sentence', '')
        speaker = item.get('speaker_name', 'Unknown')
        if sentence.strip().endswith('?'):
            question_counts[speaker] += 1

    # Determine the speaker with the most questions
    interviewer_speaker = max(question_counts, key=question_counts.get)

    # Map speakers to roles
    speaker_mapping = {speaker: 'Candidate' for speaker in speakers}
    speaker_mapping[interviewer_speaker] = 'Interviewer'

    # Generate the dialogue string
    dialogue_lines = []
    for item in data:
        sentence = item.get('sentence', '')
        speaker_name = item.get('speaker_name', 'Unknown')
        speaker_role = speaker_mapping.get(speaker_name, 'Unknown')

        # Include only non-empty sentences
        if sentence.strip():
            # **New Condition**: Include only sentences with 30 or more characters
            if len(sentence.strip()) >= 30:
                dialogue_lines.append(f"{speaker_role}: {sentence.strip()}")

    dialogue = '\n'.join(dialogue_lines)
    return dialogue


# function for adding timestamp to an interviewer name
def add_timestamp(
    interviewer_name: str,
    timestamp: str
) -> str:
    file_name = interviewer_name + timestamp

    return file_name


# function for requirements processing into a needed format
def extract_requirements(
    input_requirements: str,
    gen_requirements: list
) -> dict:
    tech_requirements = [term.strip() for term in re.split(r'[,\n]+', input_requirements)]
    combined_requirements = tech_requirements + gen_requirements

    return {
        'tech_requirements': tech_requirements,
        'general_requirements': gen_requirements,
        'all_requirements': combined_requirements
    }


# function to ensure the json is valid
def validate_json(input_text: str) -> dict:
    repaired_json = repair_json(input_text)
    return json.loads(repaired_json)


# function to create 2 dicts available for future parsing
def process_categorized_answers(inputs: dict) -> dict:
    categorized_answers = inputs['categorized_answers']
    categorized_answers_json = validate_json(categorized_answers)

    tech_requirements = inputs['tech_requirements']
    gen_requirements = inputs['general_requirements']

    soft_categorized_answers = [item for item in categorized_answers_json if item['Category'] in gen_requirements]
    tech_categorized_answers = [item for item in categorized_answers_json if item['Category'] in tech_requirements]

    return {
        'soft_categorized_answers': soft_categorized_answers,
        'tech_categorized_answers': tech_categorized_answers
    }


# debug function for logs
def print_ai_message(ai_message):
    print(f"Content:\n{ai_message.content}\n")


def print_dict(data):
    print(json.dumps(data, indent=2))


def debug_step(name):
    def debug_lambda(x):
        print(f"\n{name} Output:")
        print(f"Type of x: {type(x)}")
        if isinstance(x, AIMessage):
            print_ai_message(x)
        else:
            print_dict(x)
        return x
    return RunnableLambda(debug_lambda)


# Hardcoded variables for sections
general_requirements = [
    "Introduction",
    "Main responsibilities",
    "Git/CI/CD",
    "Scrum/Kanban/Waterfall",
    "SOLID/Object Oriented Programming/Design Patterns"
]


# Markdown formatting func
def preprocess_data(data):
    data = re.sub(r'(:)\s*$', '', data, flags=re.MULTILINE)
    data = re.sub(r'""(.*?)""', r'\1', data, flags=re.DOTALL)
    data = re.sub(r'\n*---\n*', r'\n---\n', data)
    lines = data.strip().split('\n')

    processed_lines = []

    section_heading_pattern = re.compile(r'^\d+\.\s')
    subheading_pattern = re.compile(r'^###')
    separator_pattern = re.compile(r'^---$')

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            processed_lines.append('<br>')
            continue

        if separator_pattern.match(stripped_line):
            processed_lines.append('<hr class="separator">')
            continue

        if section_heading_pattern.match(stripped_line):
            processed_lines.append(f'<div class="level-0 section-heading">{stripped_line}</div>')
            continue

        if subheading_pattern.match(stripped_line):
            processed_lines.append(f'<div class="level-1 subheading">{stripped_line}</div>')
            continue

        processed_lines.append(f'<div class="level-3 content-line">{stripped_line}</div>')
    processed_data = '\n'.join(processed_lines)

    return processed_data


# Markdown converting func
def convert_to_html(data):
    preprocessed_data = preprocess_data(data)
    html = markdown.markdown(
        preprocessed_data,
        extensions=['extra', 'sane_lists', 'codehilite'],
        extension_configs={
            'codehilite': {
                'guess_lang': False
            }
        }
    )
    return html


