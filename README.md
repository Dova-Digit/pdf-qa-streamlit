# 📚 Assistant PDF QA — Local RAG (Streamlit + FAISS + Ollama)

Applicazione **Streamlit** che permette di fare domande su uno o più PDF usando:

- **Embeddings locali** (HuggingFace: `sentence-transformers/all-MiniLM-L6-v2`)
- **LLM locale** tramite **Ollama** (es. `mistral`)
- **FAISS** come motore vettoriale
- UI in stile chat con: cronologia, fonti, import/export conversazione

Funziona **100% offline**, senza API esterne.

---

## 🚀 Funzionalità

- Caricamento PDF multipli  
- Estrazione testo + suddivisione in chunk  
- Indicizzazione vettoriale con FAISS  
- Recupero dei chunk rilevanti  
- Generazione risposta via LLM (Ollama)  
- Mostra fonti (file + pagina)  
- Persistenza indice opzionale  
- Import/export chat  

---

## 🔧 Requisiti

- Windows / Linux / macOS  
- Python **3.10+**  
- (Opzionale) **Ollama** installato con modello `mistral`

Installazione Ollama:

```bash
winget install Ollama.Ollama
ollama pull mistral
