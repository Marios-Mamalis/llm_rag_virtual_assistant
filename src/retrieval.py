import logging
import logging.config as lcfg
import yaml
import numpy as np
from openai import AzureOpenAI
import typing


with open('log_config.yml', 'rt') as f:
    config = yaml.safe_load(f.read())
    lcfg.dictConfig(config)
logger = logging.getLogger('mainlogger')


def openai_azure_get_embeddings(input_string: str, api_key: str, api_version: str, endpoint: str,
                                deployment_name: str) -> np.array:
    """
    Calculates embeddings for a given string with AzureOpenAI embedding models.
    :param input_string: The given string.
    :param api_key: The api_key configuration
    :param api_version: The api_version configuration
    :param endpoint: The endpoint configuration
    :param deployment_name: The deployment_name configuration
    :return: A numpy array of length equal to the size of the output of the embedding model.
    """

    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
    response = client.embeddings.create(input=input_string, model=deployment_name)

    return np.array(response.data[0].embedding)


class Vectorstore:
    def __init__(self, api_key: str, api_version: str, endpoint: str, deployment_name: str):
        self.api_key = api_key
        self.api_version = api_version
        self.endpoint = endpoint
        self.deployment_name = deployment_name

        self.embeddings = None
        self.texts = []

    def add_document(self, input_string: str):
        embedding = openai_azure_get_embeddings(input_string, self.api_key, self.api_version, self.endpoint,
                                                self.deployment_name)

        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.empty((0, len(embedding)), dtype=np.float32)

        self.embeddings = np.vstack([self.embeddings, embedding])
        self.texts.append(input_string)

    def retrieve_similar(self, query: str, top_k: int = 3) -> typing.List[str]:
        """Based on cosine similarity matching"""
        query_embedding = openai_azure_get_embeddings(query, self.api_key, self.api_version, self.endpoint,
                                                      self.deployment_name)
        query_part = query_embedding / np.linalg.norm(query_embedding)
        stored_embeddings_part = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(stored_embeddings_part, query_part)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.texts[i] for i in top_k_indices]
