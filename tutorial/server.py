from typing import Any
import pinecone
import os

from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")

embeddings = OpenAIEmbeddings()
vector_store = Pinecone.from_existing_index("recipes", embeddings)

# 13.9 Chef API
app = FastAPI(
    title="CheftGPT. The best provider of Indian Recipes in the world.",
    description="Give ChefGPT the name of an ingredient and it will give you multiple recipes to use that ingredient on in return.",
    servers=[
        {
            "url": "https://rage-adapter-gtk-wooden.trycloudflare.com",
        },
    ],
)


# 13.5 API Key Auth
class Quote(BaseModel):
    quote: str = Field(description="The quote that Nicolacus Maximus said.")
    year: int = Field(description="The year when Nicolacus Maximus said the quote.")


@app.get(
    "/quote",
    summary="Returns a random quote by Nicolacus Maximus.",
    description="Upon receiving a GET request this endpoint will return a real quote said by Nicolacus Maximus himself.",
    response_description="A quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said.",
    response_model=Quote,
    openapi_extra={"x-openai-isConsequential": False},
)
def get_quote(request: Request):
    print(request.headers)
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950,
    }


# 13.6 OAuth
class Document(BaseModel):
    page_content: str


@app.get(
    "/recipes",
    summary="Returns a list of recipes.",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient.",
    response_description="A Document object that contains the recipe and preparation instructions",
    response_model=list[Document],
    openapi_extra={"x-openai-isConsequential": True},  # allow or decline
    openapi_extra={"x-openai-isConsequential": False},  # allow, always allow or decline
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    return docs.page_content


user_token_db = {"ABCDEF": "nico"}


@app.get(
    "/authorize",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(
    response_type: str, client_id: str, redirect_uri: str, scope: str, state: str
):
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Log Into Nicolacus Maximus</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolacus Maximus GPT</a>
        </body>
    </html>
    """


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(payload: Any = Body(None), code=Form(...)):
    print(payload)
    return {"access_token": user_token_db[code]}


"""
# 13.10 Code Challenge

- [ ] OAuth인증을 통해 레시피를 저장하는 유저가 누군지 알아낸다.
- [ ] Make Database
- [ ] 유저들의 즐겨찾기 목록 보여주기
- [ ] 특정 유저의 레시피를 나열해줄 새로운 url(endpoint)을 만들어야 한다.
"""
