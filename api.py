import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import tempfile
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
app = FastAPI()

# Global variables to store the conversational chain and session state.
CONVERSATIONAL_CHAIN = None
CHAT_MESSAGES = []  # Global chat history for the current session
SESSION_STORE = {}  # Session history store if needed

# Dummy implementations (replace with your actual implementations)


# Load and split the PDF document and return the documents and text chunks
def load_split_pdf(file_path):
    # Load the PDF document and split it into chunks
    loader = PyPDFLoader(file_path)  # Initialize the PDF loader with the file path
    documents = loader.load()  # Load the PDF document

    # Initialize the recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # Set the maximum chunk size
        chunk_overlap=20,  # Set the number of overlapping characters between chunks
        separators=["\n\n", "\n", " ", ""],  # Define resume-specific separators for splitting
    )

    # Split the loaded documents into chunks
    chunks = text_splitter.split_documents(documents)
    return chunks

# Initialize the Hugging Face embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyAs_-UoJozNSf1UcvPUlTr4yfMvDcsX0Qc')

def create_vector_store(chunks):
    # Store embeddings into the vector store
    vector_store = FAISS.from_documents(
        documents=chunks,  # Input chunks to the vector store
        embedding=embeddings  # Use the initialized embeddings model
    )
    return vector_store

# Function to get or create a session chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]

def initialize_chain(resume_pdf_path: str, job_description: str, analysis: str):
    """
    Initialize the conversational chain from the resume, job description, and analysis.
    """
    # 🔹 Load and process resume PDF
    chunks = load_split_pdf(resume_pdf_path)
    vector_store = create_vector_store(chunks)

    # 🔹 Setting up retriever with resume as primary knowledge source
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
    )

    # 🔹 Initialize language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=500,
        google_api_key='AIzaSyAs_-UoJozNSf1UcvPUlTr4yfMvDcsX0Qc'  # ensure this env var is set
    )

    # 🔹 Contextualizing user question
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

    # 🔹 Build a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 🔹 System Prompt to Include Job Context and Analysis
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

    # 🔹 Create a prompt for answering questions
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 🔹 Set up the question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 🔹 Wrap the chain with message history support
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
    session_id: str = "default_session"  # Optionally allow client-specified session IDs

@app.post("/upload")
async def upload_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    analysis: str = Form(...)
):
    """
    Upload endpoint to process the resume PDF, job description, and analysis.
    This initializes the conversational chain.
    """
    global CONVERSATIONAL_CHAIN, CHAT_MESSAGES

    # Save the uploaded PDF to a temporary file.
    try:
        suffix = os.path.splitext(resume.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await resume.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    # Initialize the chain with the uploaded resume and provided texts.
    CONVERSATIONAL_CHAIN = initialize_chain(tmp_path, job_description, analysis)

    # Reset chat messages
    CHAT_MESSAGES = []

    # Optionally, remove the temporary file after processing.
    os.remove(tmp_path)

    return JSONResponse(content={"message": "Upload successful and chain initialized."})

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint where the user submits a query and gets an answer.
    """
    global CONVERSATIONAL_CHAIN, CHAT_MESSAGES

    if CONVERSATIONAL_CHAIN is None:
        raise HTTPException(status_code=400, detail="Please do something")

    user_query = request.query.strip()
    session_id = request.session_id

    # Append the user query to the global chat history.
    CHAT_MESSAGES.append({"role": "user", "content": user_query})

    # Prepare input for the conversational chain.
    input_data = {
        "input": user_query,
        "chat_history": CHAT_MESSAGES,
    }
    try:
        # Invoke the chain.
        response = CONVERSATIONAL_CHAIN.invoke(
            input_data,
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain invocation failed: {e}")

    answer_text = response.get("answer", "No answer generated.")

    # Append the assistant's answer to the chat history.
    CHAT_MESSAGES.append({"role": "assistant", "content": answer_text})

    return JSONResponse(content={"answer": answer_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)