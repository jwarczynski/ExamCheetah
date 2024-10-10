import json

from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate

from tqdm import tqdm

from rag_pipeline import query_rag
from core import get_embedding_function

DATA_PATH: str = 'data'
CHROMA_PATH: str = 'chroma'
EMBEDDING_MODEL_NAME: str = 'nomic-embed-text'
CHAT_MODEL_NAME: str = 'phi3'
MP_DATASET_PATH: str = '../datasets/mp.json'
QUIZZ_QUESTION_TEMPLATE_PATH: str = 'single_answer_question_template.txt'

quizz_question_template: str = open(QUIZZ_QUESTION_TEMPLATE_PATH, 'r').read()
validation_template: str = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def main():
    global language_model
    language_model = Ollama(model=CHAT_MODEL_NAME)
    db: Chroma = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(EMBEDDING_MODEL_NAME),
    )
    with open(MP_DATASET_PATH, 'r') as f:
        mp_data = json.load(f)
    num_questions = len(mp_data['data'])
    print(f'Number of questions: {num_questions}')

    accuracy = 0
    for question in tqdm(mp_data['data']):
        query = question['question']
        correct_answer = question['correct']
        answers = [f'{chr(97 + i)}) {q}' for i, q in enumerate(question['answers'])]
        question = f'{query}\n' + '\n'.join(answers)

        response, prompt_string, sources = query_rag(
            query=question,
            prompt_template=quizz_question_template,
            language_model=language_model,
            db=db,
        )

        try:
            accuracy += query_and_validate(validation_template, response, correct_answer)
        except ValueError as e:
            print(f'Error: {e}')

    accuracy = accuracy / num_questions
    print(f'Accuracy: {accuracy}')


def query_and_validate(template: str, actual_response: str, expected_response: str) -> bool:
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(template)
    prompt: PromptValue = prompt_template.format_prompt(actual_response=actual_response, expected_response=expected_response)

    judgement: str = language_model.invoke(prompt)
    judgement: str = judgement.strip().lower()
    if "true" in judgement:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {judgement}" + "\033[0m")
        return True
    elif "false" in judgement:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {judgement}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )


if __name__ == '__main__':
    main()
