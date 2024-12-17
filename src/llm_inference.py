from openai import AzureOpenAI
from openai._exceptions import RateLimitError
import typing
import logging
import logging.config as lcfg
import yaml
from retrying import retry


with open('log_config.yml', 'rt') as f:
    config = yaml.safe_load(f.read())
    lcfg.dictConfig(config)
logger = logging.getLogger('mainlogger')


@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000,
       retry_on_exception=lambda exept: isinstance(exept, RateLimitError))
def openai_inference(conversation_history: typing.List[typing.Dict[str, str]], user_query: str,
                     api_key: str, api_version: str, endpoint: str,
                     deployment: str) -> typing.List[typing.Dict[str, str]]:
    """
    Performs inference with an OpenAI model.
    :param user_query: The string of the user query.
    :param conversation_history: The conversational history in the form specified by the OpenAI library. Specifically:
    conversation_history = [{"role": "user", "content": "Input"},
                            {"role": "assistant", "content": "Output"}, ...]
    :param api_key: model api_key configuration
    :param api_version: model api_version configuration
    :param endpoint: model endpoint configuration
    :param deployment: model deployment configuration
    :return: The conversational history with the addition of the last user-assistant appended on the end, in the same
    format.
    """

    # Initialize client
    try:
        client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
    except Exception as e:
        logging.error(f'Client initialization failed: {e}')
        raise e

    # assert that the last dialogue was done by the model
    if conversation_history:  # if not new conversation
        assert conversation_history[-1]['role'] == 'assistant', 'Convesation must end with the assistant writing.'

    new_conversation_history = conversation_history + [{"role": "user", "content": user_query}]
    completion = client.chat.completions.create(model=deployment, messages=new_conversation_history)
    new_conversation_history = (new_conversation_history +
                                [{"role": "assistant", "content": completion.choices[0].message.content}])

    return new_conversation_history
