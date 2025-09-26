import os
import base64
from dotenv import load_dotenv, find_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Groq Python SDK
from groq import Groq

# -------------------------
# ENV setup
# -------------------------
load_dotenv(find_dotenv())
GROQ_API = os.environ.get("GROQ_API") or os.environ.get("GROQ_API_KEY")


# -------------------------
# Conversation Memory Setup
# -------------------------
def get_conversation_memory():
    """Create and return conversation memory"""
    return ConversationBufferWindowMemory(
        k=5,  # Keep last 5 exchanges in memory
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


# -------------------------
# Utility: Encode Image
# -------------------------
def encode_image(image_path: str):
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------------
# Vision: Analyze Image with Groq
# -------------------------
def analyze_image_with_groq(query: str, image_path: str, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """Send image + query to Groq multimodal model and get findings."""
    client = Groq(api_key=GROQ_API)
    encoded_image = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe medical findings relevant to this query: {query}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content


# -------------------------
# Text LLM (for RAG)
# -------------------------
def load_llm():
    return ChatGroq(
        temperature=0.4,
        groq_api_key=GROQ_API,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )


# -------------------------
# Custom Prompt for Medical RAG with Memory
# -------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are a professional medical consultant chatbot trained on verified medical literature, textbooks, and clinical guidelines provided in PDF format. 
Your primary role is to assist users with accurate, evidence-based information about symptoms, diagnoses, and treatments.

Always follow these principles:
- Use only the information extracted from the provided documents. Do not invent or speculate.
- If a question cannot be answered based on the documents, clearly state that and suggest consulting a licensed medical professional.
- Prioritize clarity, empathy, and professionalism in your tone.
- When referencing treatments or diagnoses, cite the source document or section if available.
- Avoid giving definitive medical advice or prescriptions. Instead, offer general guidance based on the literature.

Chat History:
{chat_history}

Context from documents: 
{context}

Question: {question}

Answer directly and professionally:
"""


def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["chat_history", "context", "question"]
    )


# -------------------------
# Load FAISS DB
# -------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# Create the QA chain with memory
def create_qa_chain():
    memory = get_conversation_memory()

    return ConversationalRetrievalChain.from_llm(
        llm=load_llm(),
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            'prompt': set_custom_prompt()
        }
    )


# Initialize the QA chain
qa_chain = create_qa_chain()


# -------------------------
# Unified Query Handler with Memory
# -------------------------
def medical_chatbot(query: str, image_path: str = None):
    """
    If image is provided, analyze it first with Groq Vision,
    then pass findings + query into RAG.
    """
    if image_path:
        print("\n[INFO] Image provided. Extracting findings with Groq Vision...\n")
        image_findings = analyze_image_with_groq(query, image_path)
        full_query = f"Image findings: {image_findings}\n\nUser question: {query}"
    else:
        full_query = query

    response = qa_chain.invoke({"question": full_query})
    return response


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("Medical Chatbot with Memory - Type 'exit' to end the conversation")

    while True:
        mode = input("\nDo you want to upload an image? (y/n/exit): ").strip().lower()
        if mode == 'exit':
            break
        elif mode == "y":
            query = input("Write your medical query here: ")
            if query.lower() == 'exit':
                break
            image_path = input("Enter image path: ").strip()
            result = medical_chatbot(query, image_path=image_path)
        else:
            query = input("Write your medical query here: ")
            if query.lower() == 'exit':
                break
            result = medical_chatbot(query)

        print("\nRESULT:\n", result["answer"])
        print("\nSOURCES:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\nSOURCE {i}: {doc.metadata}")
            print(doc.page_content[:400], "...")