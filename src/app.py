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
from src.prompts import RAG_SYSTEM_PROMPT, fill_rag_system_prompt
from src.retrieval import Vectorstore
from src.llm_inference import openai_inference


with open('log_config.yml', 'rt') as f:
    config = yaml.safe_load(f.read())
    lcfg.dictConfig(config)
logger = logging.getLogger('mainlogger')


app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')


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
        app.openai_api_key = os.environ['AZURE_OPENAI_API_KEY']

        app.openai_endpoint = model_config['OPENAI']['ENDPOINT']
        app.openai_api_version = model_config['OPENAI']['API_VERSION']

        app.openai_llm_deployment = model_config['OPENAI']['LLM']['DEPLOYMENT_NAME']
        app.openai_embed_deployment = model_config['OPENAI']['EMBED']['DEPLOYMENT_NAME']
    except KeyError as e:
        logging.error(f'Missing configuration: {e}. Exiting.')
        sys.exit(1)

    # Instantiate the vectorstore
    app.rag_store = Vectorstore(app.openai_api_key, app.openai_api_version, app.openai_endpoint,
                                app.openai_embed_deployment)

    # Seed it
    # research article
    app.rag_store.add_document("Recent studies show that global temperatures have risen by approximately 1.2°C since"
                               " the pre-industrial era, resulting in a decrease in crop yields. In tropical regions,"
                               " such as Sub-Saharan Africa, maize production has dropped by 20% due to prolonged "
                               "droughts, while in temperate zones, wheat yields are declining due to increased heat "
                               "stress during the growing season.")
    # process to return an item
    app.rag_store.add_document("To return an item, please ensure it is in its original condition and packaging. You can"
                               " initiate the return process by logging into your account, selecting the item, and "
                               "choosing the 'Return' option. Returns must be requested within 30 days of receiving the"
                               " item, and a return shipping label will be provided.")
    # legal case
    app.rag_store.add_document("In the case of XYZ Corporation v. ABC Ltd., the court ruled that the defendant's use of"
                               " the plaintiff's patented technology without authorization constituted a clear "
                               "violation of intellectual property rights under section 101 of the Patent Act. The "
                               "court further emphasized that the defendant's actions caused significant financial "
                               "harm to the plaintiff, justifying both compensatory damages and injunctive relief.")
    # medical
    app.rag_store.add_document("Patient presents with a 5-day history of persistent fever, dry cough, and mild "
                               "shortness of breath. Past medical history is significant for asthma. On examination, "
                               "the patient has a temperature of 101°F and slight wheezing. A rapid antigen test for "
                               "influenza was positive, and a chest X-ray shows mild infiltration in the right lower "
                               "lobe.")
    # news
    app.rag_store.add_document("In a recent announcement, the government unveiled a comprehensive tax reform plan aimed"
                               " at reducing corporate tax rates and increasing personal tax brackets for higher-income"
                               " earners. Key political figures, including Senator Jane Doe, have expressed mixed "
                               "reactions, with some supporting the measure as a boost to economic growth, while others"
                               " argue it disproportionately benefits the wealthy.")


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
    context_pieces = app.rag_store.retrieve_similar(user_query, top_k=1)

    # inject context into system prompt
    system_context_prompt = fill_rag_system_prompt(context_pieces=context_pieces, string_template=RAG_SYSTEM_PROMPT)

    assistant_response = openai_inference(
        conversation_history=[
            {'role': 'system', 'content': system_context_prompt}
        ], user_query=user_query, api_key=app.openai_api_key,
        api_version=app.openai_api_version, endpoint=app.openai_endpoint, deployment=app.openai_llm_deployment
    )

    logger.info(f'Context piece used:\n\n{" ".join(context_pieces)}')

    return {'response': assistant_response[-1]['content']}
