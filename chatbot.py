import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

os.environ["USER_AGENT"] = "my-rag-chatbot/1.0"

# loader = WebBaseLoader("https://www.linkedin.com/in/adityaavarshney/")
loader = PyPDFLoader("resume.pdf")
docs = loader.load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
# ☝️ api_key mat likho — .env se auto-read hoga

prompt = ChatPromptTemplate.from_template("""
You are JARVIS, a helpful assistant.
Use context if relevant, otherwise use general knowledge.

<context>
{context}
</context>

Question: {input}
Answer:""")

chain = prompt | llm | StrOutputParser()

def get_response(question: str) -> str:
    context = "\n\n".join(doc.page_content for doc in retriever.invoke(question))
    return chain.invoke({"context": context, "input": question})