import typing
import logging
import logging.config as lcfg
import yaml


with open('log_config.yml', 'rt') as f:
    config = yaml.safe_load(f.read())
    lcfg.dictConfig(config)
logger = logging.getLogger('mainlogger')


def retrieval_func(input_text: str) -> typing.List[str]:
    """
    !Mock function.

    Based on an input user query, it retrieves semantically similar context chunks and returns them.

    :param input_text: The user's input text.
    :return: A list that of semantically similar to the user query, text chunks.
    """

    return [f'Placeholder relevant context piece #{i}, ignore.' for i in range(5)]
