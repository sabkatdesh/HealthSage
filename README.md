# HealthSage
An AI-powered **medical chatbot** that combines **Retrieval-Augmented Generation (RAG)**, **conversation memory**, and **multimodal image analysis**.  
Built with **LangChain, FAISS, HuggingFace Embeddings, Groq LLM**, and **Gradio UI**.

---

## 🚀 Features
- ✅ **Medical Q&A** grounded in verified textbooks & clinical PDFs  
- ✅ **Retrieval-Augmented Generation (RAG)** with FAISS Vector Store  
- ✅ **Conversation memory** (remembers last 5 interactions)  
- ✅ **Image support** (upload X-rays, scans, reports for analysis)  
- ✅ **Source transparency** – every answer shows supporting documents  
- ✅ **Modern ChatGPT-style UI** with Gradio  

---

<img width="1897" height="842" alt="image" src="https://github.com/user-attachments/assets/450cb4fc-6bb3-4f95-9599-af1024b8da60" />

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sabkatdesh/HealthSage.git
cd HealthSage
```

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

### 3. Install Dependencies
pip install -r requirements.txt

### 4.Set Up Environment Variables

Create a .env file in the project root:
# .env
GROQ_API_KEY=your_groq_api_key_here

python embed_pdfs.py


📚 Prepare Knowledge Base
Place your medical PDF documents in the data/ folder.

Run the embedding script to process and index them:

python embed_pdfs.py


▶️ Run the Application

Launch the Gradio UI: app.py

Project Structure:
HealthSage/
│── app.py                # Gradio frontend
│── doctor_brain.py       # Core chatbot logic (RAG + memory + image support)
│── embed_pdfs.py         # Preprocess & embed PDF knowledge base
│── data/                 # Your medical PDFs
│── vectorstore/          # FAISS vector database
│── assets/               # Screenshots for README
│── requirements.txt      # Dependencies
│── .env                  # Environment variables
│── README.md             # Project documentation


⚠️ Disclaimer

This project is for educational and research purposes only.
It does not provide professional medical advice.
Always consult a licensed medical professional for real health concerns.


📄 License
MIT License – feel free to use and modify, but please credit the original author.

👨‍💻 Author: Sabkat Desh

🌟 If you like this project, don’t forget to star the repo!


