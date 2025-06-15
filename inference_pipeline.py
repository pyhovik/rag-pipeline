from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain import hub
from langchain_ollama import ChatOllama
from config import RagConfig

print("Qdrant client init...")
client = QdrantClient(
    #location = ":memory:",
    path="/tmp/qdrant.db"
)

print("Embeddings model init...")
embeddings = HuggingFaceEmbeddings(model_name=RagConfig.EMBEDDING_MODEL)

print("VectoreStore init...")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=RagConfig.COLLECTION_NAME,
    embedding=embeddings
)

print("Prompt init...")
prompt = hub.pull("rlm/rag-prompt")

print("LLM init...")
llm = ChatOllama(
    model=RagConfig.CHAT_MODEL,
    temperature=0,
    # base_url="http://100.110.2.78:11434"
)

def get_answer(question: str) -> str:
    retrieved_docs = vector_store.similarity_search(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    messages = prompt.invoke({"question": question, "context": context})
    response = llm.invoke(messages)
    return response.content


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


if __name__ == "__main__":
    question = "Что такое ECCM?"
    retrieved_docs = vector_store.similarity_search(question)
    for doc in retrieved_docs:
        print(doc.metadata)
    # context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # messages = prompt.invoke({"question": question, "context": context})
    # response = llm.invoke(messages)
    # print("Answer:\n", response.content)

    from qdrant_client.models import Document, VectorParams, Distance
    MODEL_NAME = "BAAI/bge-small-en"
    COLLECTION_NAME = "demo-collection"

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=Document(
            text=question, 
            model=MODEL_NAME
        )
    )
    for point in search_result.points:
        print(point.payload["title"], point.payload["start_index"])
