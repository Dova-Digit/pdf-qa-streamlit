# 📚 Assistant PDF QA — Local RAG (Streamlit + FAISS + Ollama)

> IT 🇮🇹 + EN 🇬🇧

# 🇮🇹 Versione Italiana

Applicazione **Streamlit** per fare domande su uno o più **PDF** in locale, con:

- **Embeddings locali** (HuggingFace: `sentence-transformers/all-MiniLM-L6-v2`)
- **LLM locale** via **Ollama** (es. `mistral`)
- **FAISS** come motore vettoriale
- UI tipo chat (cronologia, fonti, import/export conversazioni, i18n IT/EN)

Funziona **100% offline**, senza API esterne.

---

## 🧭 Indice (IT)
- Funzionalità
- Requisiti
- Installazione
- Esecuzione
- Come funziona (RAG)
- Troubleshooting
- Licenza
- English Version

---

## 🚀 Funzionalità
- Caricamento PDF multipli
- Estrazione testo + suddivisione in chunk
- Indicizzazione vettoriale con FAISS
- Recupero dei chunk rilevanti
- Generazione risposta via LLM (Ollama)
- Visualizzazione fonti (file + pagina)
- Persistenza indice opzionale
- Import/Export chat (JSON)

---

## 🔧 Requisiti
- Windows / Linux / macOS
- Python 3.10+
- (Opzionale) Ollama con modello mistral

### Installa Ollama (Windows)
```bash
winget install Ollama.Ollama
ollama pull mistral
```

---

# 🇬🇧 English Version

**Streamlit** app to ask questions about one or multiple **PDFs** locally, featuring:

- Local embeddings (HuggingFace: `sentence-transformers/all-MiniLM-L6-v2`)
- Local **LLM** via **Ollama** (e.g., `mistral`)
- **FAISS** vector store
- Chat-style **UI** (history, sources, import/export, i18n IT/EN)

Runs **100% offline**, no external APIs.

---

## 🧭 Table of Contents (EN)
- Features
- Requirements
- Install
- Run
- RAG Pipeline
- Troubleshooting
- License

---

## Features
- Load multiple PDFs
- Extract text + split into chunks
- Vector indexing with FAISS
- Retrieve top-k relevant chunks
- Generate answers via local LLM (Ollama)
- Show sources (file + page)
- Optional persistent index
- Import/Export chat (JSON)

---

## Requirements
- Windows / Linux / macOS
- Python 3.10+
- (Optional) Ollama installed with model mistral

### Install Ollama (Windows)
```bash
winget install Ollama.Ollama
ollama pull mistral
```

