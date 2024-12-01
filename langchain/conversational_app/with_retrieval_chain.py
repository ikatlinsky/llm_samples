from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
raw_documents = PyPDFLoader("italy.pdf").load()
documents = text_splitter.split_documents(raw_documents)
italy_db = FAISS.from_documents(documents=documents, embedding=embeddings)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

convesation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=italy_db.as_retriever(),
    memory=memory,
    verbose=True,
)

resp = convesation.invoke("Hi there.")
print(resp)
resp = convesation.invoke("What is the most iconic place in Rome?")
print(resp)
