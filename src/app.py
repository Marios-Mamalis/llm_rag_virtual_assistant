import typing
from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import dotenv
import logging
import logging.config as lcfg
import os
import sys
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.prompts import RAG_CONTEXT_INFERENCE, fill_rag_prompt
from src.retrieval import retrieval_func
from src.llm_inference import openai_inference


with open('log_config.yml', 'rt') as f:
    config = yaml.safe_load(f.read())
    lcfg.dictConfig(config)
logger = logging.getLogger('mainlogger')


app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
config = {}


@app.get("/", response_class=HTMLResponse)
async def userpage():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())


@app.get("/health")
def check_status() -> typing.Dict[str, str]:
    return {"status message": "Server is up."}


@app.on_event("startup")
def read_env_variables():
    """
    Attaches configuration variables to the current session.
    Requires a YAML configuration with the necessary keys and values, and environment variables or a .env file
    containing the required keys.
    """

    # read non-sensitive data
    with open('config.yml', 'r') as ff:
        model_config = yaml.safe_load(ff)
    # read sensitive data
    dotenv.load_dotenv()  # if local .env can be used

    try:
        config['openai_llm_api_key'] = os.environ['AZURE_OPENAI_API_KEY']

        config['openai_llm_endpoint'] = model_config['OPENAI']['LLM']['ENDPOINT']
        config['openai_llm_deployment'] = model_config['OPENAI']['LLM']['DEPLOYMENT_NAME']
        config['openai_llm_api_version'] = model_config['OPENAI']['LLM']['API_VERSION']

        config['openai_embed_deployment'] = model_config['OPENAI']['EMBED']['DEPLOYMENT_NAME']
    except KeyError as e:
        logging.error(f'Missing configuration: {e}. Exiting.')
        sys.exit(1)


class UserQuery(BaseModel):
    query_text: str


@app.post("/rag_inference")
def rag_inference(user_query: UserQuery) -> typing.Dict[str, str]:
    """
    Given a user query, performs RAG inference to answer it.
    :param user_query: The user's query as a string.
    :return: The system's response in the format of {'response': <assistant response>}
    """

    user_query = user_query.query_text
    context_pieces = retrieval_func(user_query)
    input_text = fill_rag_prompt(context_pieces=context_pieces, user_query=user_query,
                                 string_template=RAG_CONTEXT_INFERENCE)
    assistant_response = openai_inference(
        conversation_history=[], user_query=input_text,
        api_key=config['openai_llm_api_key'], api_version=config['openai_llm_api_version'],
        endpoint=config['openai_llm_endpoint'], deployment=config['openai_llm_deployment']
    )

    return {'response': assistant_response[-1]['content']}
