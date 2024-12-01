from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool

from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent

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

italy_tool = create_retriever_tool(
  italy_db.as_retriever(),
  "italy_travel",
  "Searches for travel information about Italy.",
)
generic_search_tool = Tool.from_function(
  func=SerpAPIWrapper().run,
  name="Search",
  description="Searches the web for information when information is needed about current events.",
)

tools = [italy_tool, generic_search_tool]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

conversation = create_conversational_retrieval_agent(
  llm=llm,
  tools=tools,
  memory_key="chat_history",
  verbose=True, 
)
resp = conversation("Tell me something about Pantheon.")
print(resp["output"])
resp = conversation("What is the weather in Rome currently?")
print(resp["output"])