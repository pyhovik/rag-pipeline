from config import RagConfig
COLLECTION_NAME = RagConfig.COLLECTION_NAME
EMBEDDING_MODEL = RagConfig.EMBEDDING_MODEL
SOURCE_URLS_FILE = RagConfig.SOURCE_URLS_FILE
CHUNK_SIZE = RagConfig.CHUNK_SIZE
CHUNK_OVERLAP = RagConfig.CHUNK_OVERLAP

print("Qdrant client init...")
from qdrant_client import QdrantClient
client = QdrantClient(
    #location = ":memory:",
    path="/tmp/qrant.db"
)

print("Embeddings model init...")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("VectoreStore init...")
from langchain_qdrant import QdrantVectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

print("Prompt init...")
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

print("LLM init...")
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0,
    # base_url="http://100.110.2.78:11434"
)

question = input("What is your question?\n")
retrieved_docs = vector_store.similarity_search(question)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
messages = prompt.invoke({"question": question, "context": context})
response = llm.invoke(messages)
print("Answer:\n", response.content)




## Код ниже использует LangGraph для создания rag-приложения
## Будет полезно использовать его для "тюнинга" приложения
##
# print("Define state for application...")
# from langchain_core.documents import Document
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict


# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str

# print("Define application steps")
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}

# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}


# print("Compile application and test")
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

# response = graph.invoke({"question": "What is ECCM?"})
# print(response["answer"])
