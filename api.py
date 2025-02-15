import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import tempfile
#
# Import your LangChain components
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import uvicorn
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Fetch the Gemini API key from the environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
print(GOOGLE_API_KEY)
app = FastAPI()

# Global variables to store the conversational chain and session state.
CONVERSATIONAL_CHAIN = None
CHAT_MESSAGES = []  # Global chat history for the current session
SESSION_STORE = {}  # Session history store if needed

# Dummy implementations (replace with your actual implementations)


# Load and split the PDF document and return the documents and text chunks
def load_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

# Initialize the Hugging Face embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def create_vector_store(chunks):
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]

def initialize_chain(resume_pdf_path: str, job_description: str, analysis: str):
    chunks = load_split_pdf(resume_pdf_path)
    vector_store = create_vector_store(chunks)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=500,
        google_api_key=GOOGLE_API_KEY  # Use the env var here
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed, otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = f"""
    You are an AI assistant helping to match a candidate's resume with a job description.
    Use the provided resume information, job description, and candidate analysis to answer the question.
    If relevant, highlight **matching** and **missing** skills.
    Be concise and limit responses to three sentences.

    🔹 **Job Description:**
    {job_description}

    🔹 **Candidate Analysis:**
    {analysis}

    🔹 **Resume Context:**
    {{context}}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

@app.post("/upload")
async def upload_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    analysis: str = Form(...)
):
    global CONVERSATIONAL_CHAIN, CHAT_MESSAGES

    try:
        suffix = os.path.splitext(resume.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await resume.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    CONVERSATIONAL_CHAIN = initialize_chain(tmp_path, job_description, analysis)
    CHAT_MESSAGES = []

    os.remove(tmp_path)

    return JSONResponse(content={"message": "Upload successful and chain initialized."})

@app.post("/chat")
async def chat(request: ChatRequest):
    global CONVERSATIONAL_CHAIN, CHAT_MESSAGES

    if CONVERSATIONAL_CHAIN is None:
        raise HTTPException(status_code=400, detail="Please upload")

    user_query = request.query.strip()
    session_id = request.session_id

    CHAT_MESSAGES.append({"role": "user", "content": user_query})

    input_data = {
        "input": user_query,
        "chat_history": CHAT_MESSAGES,
    }
    try:
        response = CONVERSATIONAL_CHAIN.invoke(
            input_data,
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain invocation failed: {e}")

    answer_text = response.get("answer", "No answer generated.")
    CHAT_MESSAGES.append({"role": "assistant", "content": answer_text})

    return JSONResponse(content={"answer": answer_text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
