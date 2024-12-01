from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain

load_dotenv(override=True)

llm = ChatOpenAI()

memory = ConversationBufferMemory()
convesation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

resp = convesation.run("Hi there.")
print(resp)
resp = convesation.run("What is the most iconic place in Rome?")
print(resp)
