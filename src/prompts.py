import string
import logging
import os
import typing

os.makedirs('../logs', exist_ok=True)
logging.basicConfig(filename='../logs/llm_app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def fill_rag_prompt(string_template: string.Template, context_pieces: typing.List[str], user_query: str) -> str:
    """
    Substitutes the supplied values in a RAG prompt.
    :param string_template: The supplied RAG template.
    :param context_pieces: The retrieved context pieces.
    :param user_query: The initial user query.
    :return: The filled query to pass to the LLM.
    """

    assert context_pieces and any([i.strip() for i in context_pieces]), 'Context pieces are missing'
    assert user_query, 'User query is missing'

    return string_template.substitute(context_pieces='\n'.join(context_pieces), user_query=user_query)


RAG_CONTEXT_INFERENCE = string.Template('$context_pieces\n\nGiven the context pieces above, reply to the following user'
                                        ' query:\n$user_query')
