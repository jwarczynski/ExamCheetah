from langchain_community.vectorstores import Chroma
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate


def query_rag(
        query: str,
        prompt_template: str,
        language_model: BaseLLM,
        db: Chroma
) -> tuple[str, str | list[str | dict], list[None]]:
    retrieved_chunks = db.similarity_search_with_score(query, k=5)
    context_text = '\n\n---\n\n'.join([chunk.page_content for chunk, _ in retrieved_chunks])
    prompt_template_ = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template_.format_prompt(context=context_text, question=query)

    prompt_string = prompt.to_messages()[0].content
    sources = [chunk.metadata.get('id', None) for chunk, _ in retrieved_chunks]

    response = language_model.invoke(prompt)
    formatted_response = f"Response: {response}\nSources: {sources}"

    return formatted_response, prompt_string, sources
