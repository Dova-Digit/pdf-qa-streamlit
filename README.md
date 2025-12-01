🇬🇧 Assistant PDF QA — Local RAG (Streamlit + FAISS + Ollama)

Streamlit application that allows asking questions about one or multiple PDFs using:

Local embeddings (HuggingFace: sentence-transformers/all-MiniLM-L6-v2)

Local LLM via Ollama (e.g., mistral)

Optional OpenAI support

FAISS vector store for offline indexing

UI features: chat history, sources, import/export conversation, i18n (IT/EN)

Runs fully offline, no external API required.

🔧 Features

Load multiple PDFs

Extract text + split into chunks

Vector indexing using FAISS

Retrieve most relevant chunks

Generate answers with local LLM

Display sources (file + page)

Optional persistent index

Import/export chat

Clean Streamlit interface

📌 Requirements

Windows / Linux / macOS

Python 3.10+

(Optional) Ollama installed + mistral model

Install Ollama:

winget install Ollama.Ollama
ollama pull mistral

📦 Install Python Dependencies

Using the included requirements.txt:

pip install -r requirements.txt


Or manually:

pip install streamlit langchain langchain-community langchain-openai langchain-huggingface langchain-ollama langchain-text-splitters faiss-cpu python-dotenv pypdf sentence-transformers

▶️ Run the Streamlit App
streamlit run app.py

▶️ Run from Windows Shortcut (.bat)

UI mode
run_ui.bat launches the Streamlit interface.

CLI mode
run_pdf_qa.bat launches the assistant:

python pdf_qa_assistant.py --llm ollama --ollama-model mistral --embedder hf --no-input

🧠 How the RAG Pipeline Works

Load PDFs → PyPDFLoader

Split text → RecursiveCharacterTextSplitter

Encode using HuggingFace embeddings

Index vectors with FAISS

Retrieve top-k relevant chunks

Generate the final answer via Ollama LLM

🛠 Troubleshooting

Ollama model not found
→ ollama pull mistral

Keras / TensorFlow conflict
→ uninstall TF/Keras or set:

set TRANSFORMERS_NO_TF=1
set TRANSFORMERS_NO_JAX=1


PDF not found
→ place a PDF in the folder or specify: --pdf "file.pdf"

Low-quality answers
→ increase --top-k, try model llama3

📄 License

MIT License.


---

# 🎉 **ADESSO È PRONTO PER GITHUB.**

Vuoi che:

✅ ti aggiungo anche uno **screenshot UI** nel README?  
(se sì: fai uno screenshot → mandamelo → lo preparo e ti dico dove metterlo)

Vuoi anche un badge professionale (Python, Streamlit, FAISS, Ollama)?