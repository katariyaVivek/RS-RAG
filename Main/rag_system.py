import sys,os,httpx
sys.dont_write_bytecode = True

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS,DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from llm_utils import genrate_response,genrate_subquestions

DATA_PATH ="../data/main_data/synthetic_data.csv"
FAISS_PATH = "../vectorstore"
RAG_K_THRESOLD = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"
OPENAI_ENDPOINT = "https://aalto-openai-apigw.azure-api.net"
OPENAI_KEY = "sk-proj-5ov3zmpOJP7poiGsVHVdT3BlbkFJSOIk0igQhuuWESKO0Ble"

def update_base_url(request : httpx.Request):
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path='/v1/chat')

def reciprocal_rank_fusion(document_rank_list: list[dict], k=50):
    fused_score = {}
    for doc_list in document_rank_list:
        for rank,(doc,score) in enumerate(doc_list.items()):
            if doc not in fused_score:
                fused_score[doc] = 0
            fused_score[doc] += 1 / (rank + k)
    reranked_rersults = {doc: score for doc , score in sorted(fused_score.items(), key=lambda x: x[1], reverse=True)}
    return reranked_rersults

def retrive_docs_id(question : str,k=3):
    docs_score = vectorstore_db.similarity_search_with_score(question, k=k)
    docs_score = {str(doc.metadata["ID"]):score for doc,score in docs_score}
    return docs_score

def retrive_id_and_rerank(subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
        document_rank_list.append(retrive_docs_id(subquestion,RAG_K_THRESOLD))
    reranked_documents = reciprocal_rank_fusion(document_rank_list)
    return reranked_documents


def retrive_documents_with_id(documents: pd.DataFrame, doc_id_with_score:dict):
    id_list = list(doc_id_with_score.keys())
    retrived_documents = list(documents[documents["ID"].isin(id_list)]["Resume"])
    return retrived_documents

if __name__ == "__main__":
    documents = pd.read_csv(DATA_PATH)
    documents['ID'] = documents["ID"].astype(str)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs = {"device" : "cpu"}
    )
llm = ChatOpenAI(
    default_headers={"Ocp-Apim-Subscription-Key": OPENAI_KEY},
    base_url=OPENAI_ENDPOINT,
    api_key=False,
    http_async_client=httpx.Client(
        event_hooks={
            "request" : [update_base_url]
        }),
)

question = ""

vectorstore_db = FAISS.load_local(FAISS_PATH,embedding_model,distance_strategy=DistanceStrategy.COSINE)
subquestion_list = genrate_subquestions(llm,question)
print(subquestion_list)
print("===========================")
id_list = retrive_id_and_rerank(subquestion_list)
document_list = retrive_documents_with_id(documents, id_list)
for document in document_list:
    print(document)
    print("===================================")
response = genrate_response(llm, question, document_list)


