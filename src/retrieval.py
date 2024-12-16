import typing
import os
import logging

os.makedirs('../logs', exist_ok=True)
logging.basicConfig(filename='../logs/llm_app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def retrieval_func(input_text: str) -> typing.List[str]:
    """
    !Mock function.

    Based on an input user query, it retrieves semantically similar context chunks and returns them.

    :param input_text: The user's input text.
    :return: A list that of semantically similar to the user query, text chunks.
    """

    return [f'Placeholder relevant context piece #{i}, ignore.' for i in range(5)]
