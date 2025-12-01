# --- Disable TF/JAX to avoid Keras 3 issues before any other import ---
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import os
import tempfile
from pathlib import Path
import shutil
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import requests
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ---------------------------- Utils ----------------------------
load_dotenv()

# ---------------------------- i18n (IT/EN) ----------------------------
LANGS: Dict[str, Dict[str, Any]] = {
    "it": {
        "title": "📚 PDF Question Answering — Local RAG",
        "caption": "Embeddings locali (HuggingFace) + LLM locale (Ollama) o OpenAI opzionale",
        "config": "⚙️ Config",
        "llm": "LLM",
        "ollama_model": "Modello Ollama",
        "openai_model": "Modello OpenAI",
        "embeddings": "Embeddings",
        "hf_model": "Modello HF",
        "topk": "Top-K",
        "chunksize": "Chunk size",
        "overlap": "Chunk overlap",
        "persist": "Cartella indice (opzionale)",
        "upload": "Carica uno o più PDF",
        "build": "📦 Build / Load Index",
        "rebuild": "♻️ Rebuild index",
        "rebuild_help": "Cancella l'indice persistito (se impostato) e ricostruisce da zero",
        "clear": "🧹 Clear chat",
        "chatio": "💬 Chat I/O",
        "export": "💾 Export chat (.json)",
        "import": "📥 Import chat (.json)",
        "import_help": "File JSON esportato dall'app",
        "import_mode": "Modalità import",
        "replace": "Replace",
        "append": "Append",
        "import_btn": "Importa chat",
        "no_pdf": "Nessun PDF selezionato o trovato.",
        "building": "Costruzione/Caricamento indice FAISS...",
        "ready": "Indice pronto ✅ — {n} file",
        "removed": "Cartella indice rimossa: {p}",
        "rebuilding": "Ricostruzione indice FAISS da zero...",
        "rebuilt": "Indice ricostruito ✅",
        "pages_header": "📑 Pagine per file",
        "chat_input": "Fai una domanda sui PDF...",
        "must_build": "Prima clicca su **Build / Load Index** nella sidebar.",
        "show_sources": "Mostra fonti",
        "footer": "Made with LangChain + FAISS + Streamlit. Embeddings locali (HF), LLM locale (Ollama) o OpenAI opzionale.",
        "pages_unknown": "Non ho i dati sul numero di pagine. Clicca su **Build / Load Index** o fai **Rebuild**.",
        "err_import_format": "Formato non valido: atteso un array JSON di oggetti con campi 'role' e 'content'.",
        "err_import": "Errore durante l'import: {e}",
        "ok_import": "Chat importata con successo ✅",
        "p_abbr": "pag.",
        "howmany_patterns": ["quante pagine", "numero di pagine", "totale pagine", "quanti fogli", "numero pagine"],
        "sys_prompt": "Usa SOLO il seguente contesto per rispondere. Se il contesto non contiene informazioni sufficienti, rispondi che non hai informazioni.\nContesto:\n{context}",
        "page_not_found": "Pagina {page_num} non trovata nei documenti. Il documento ha massimo {max_page} pagine.",
        "bio_prompt": "Basandoti SOLO sul seguente contesto, descri chi è {name} secondo i documenti. Se il nome non viene menzionato, rispondi che non hai informazioni specifiche.\n\nContesto:\n{context}",
        "no_info_person": "Non ho informazioni specifiche su {name} nei documenti caricati.",
        "strict_prompt": "Rispondi SOLO basandoti sul contesto fornito. Non inventare informazioni. Se il contesto non contiene la risposta, di' che non lo sai.\n\nContesto:\n{context}\n\nDomanda: {question}",
    },
    "en": {
        "title": "📚 PDF Question Answering — Local RAG",
        "caption": "Local embeddings (HuggingFace) + Local LLM (Ollama) or optional OpenAI",
        "config": "⚙️ Settings",
        "llm": "LLM",
        "ollama_model": "Ollama model",
        "openai_model": "OpenAI model",
        "embeddings": "Embeddings",
        "hf_model": "HF model",
        "topk": "Top-K",
        "chunksize": "Chunk size",
        "overlap": "Chunk overlap",
        "persist": "Persist index folder (optional)",
        "upload": "Upload one or more PDFs",
        "build": "📦 Build / Load Index",
        "rebuild": "♻️ Rebuild index",
        "rebuild_help": "Delete persisted index (if any) and rebuild from scratch",
        "clear": "🧹 Clear chat",
        "chatio": "💬 Chat I/O",
        "export": "💾 Export chat (.json)",
        "import": "📥 Import chat (.json)",
        "import_help": "JSON file exported from this app",
        "import_mode": "Import mode",
        "replace": "Replace",
        "append": "Append",
        "import_btn": "Import chat",
        "no_pdf": "No PDF selected or found.",
        "building": "Building/Loading FAISS index...",
        "ready": "Index ready ✅ — {n} file(s)",
        "removed": "Index folder removed: {p}",
        "rebuilding": "Rebuilding FAISS index from scratch...",
        "rebuilt": "Index rebuilt ✅",
        "pages_header": "📑 Pages per file",
        "chat_input": "Ask a question about the PDFs...",
        "must_build": "Please click **Build / Load Index** in the sidebar first.",
        "show_sources": "Show sources",
        "footer": "Made with LangChain + FAISS + Streamlit. Local HF embeddings, local Ollama LLM or optional OpenAI.",
        "pages_unknown": "I don't have page counts yet. Click **Build / Load Index** or **Rebuild**.",
        "err_import_format": "Invalid format: expected a JSON array of objects with 'role' and 'content'.",
        "err_import": "Import error: {e}",
        "ok_import": "Chat imported successfully ✅",
        "p_abbr": "p.",
        "howmany_patterns": ["how many pages", "number of pages", "total pages"],
        "sys_prompt": "Use ONLY the following context to answer. If the context doesn't contain sufficient information, respond that you don't have information.\nContext:\n{context}",
        "page_not_found": "Page {page_num} not found in documents. Maximum page is {max_page}.",
        "bio_prompt": "Based ONLY on the following context, describe who {name} is according to the documents. If the name is not mentioned, respond that you don't have specific information.\n\nContext:\n{context}",
        "no_info_person": "I don't have specific information about {name} in the loaded documents.",
        "strict_prompt": "Answer ONLY based on the provided context. Do not invent information. If the context doesn't contain the answer, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}",
    },
}

def t(lang: str, key: str, **kwargs) -> str:
    s = LANGS[lang][key]
    return s.format(**kwargs) if kwargs else s


# ---------------------------- Sidebar / Layout ----------------------------
st.set_page_config(page_title="PDF QA (Local)", page_icon="📚", layout="wide")

# Language switch
lang_label = "Language / Lingua"
lang_sel = st.sidebar.selectbox(lang_label, ["Italiano", "English"], index=0)
LANG = "it" if lang_sel == "Italiano" else "en"

st.title(t(LANG, "title"))
st.caption(t(LANG, "caption"))

# ---------------------------- Ollama helpers ----------------------------
def _ollama_available(base: str = "http://localhost:11434") -> bool:
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def _ollama_list_models(base: str = "http://localhost:11434") -> list[str]:
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


with st.sidebar:
    st.header(t(LANG, "config"))
    
    # Debug option
    debug_mode = st.checkbox("🔧 Debug Mode", value=False)
    
    if st.button("🔄 Reset UI state"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

    mode_llm = st.selectbox(t(LANG, "llm"), ["ollama", "openai"], index=0, key="mode_llm")

    if mode_llm == "ollama":
        ok = _ollama_available()
        st.markdown("✅ Ollama connesso" if ok else "❌ Ollama non raggiungibile su http://localhost:11434")
        models = _ollama_list_models() if ok else []
        if models:
            ollama_model = st.selectbox(t(LANG, "ollama_model"), models, index=(models.index("llama3:8b") if "llama3:8b" in models else 0), key="ollama_model")
        else:
            ollama_model = st.text_input(t(LANG, "ollama_model"), st.session_state.get("ollama_model", "llama3:8b"), key="ollama_model")
    else:
        ollama_model = st.text_input(t(LANG, "ollama_model"), st.session_state.get("ollama_model", "llama3:8b"), key="ollama_model")

    openai_model = st.text_input(t(LANG, "openai_model"), "gpt-4o-mini")

    embedder = st.selectbox(t(LANG, "embeddings"), ["hf", "openai"], index=0)
    hf_model = st.text_input(t(LANG, "hf_model"), "sentence-transformers/all-MiniLM-L6-v2")

    top_k = st.slider(t(LANG, "topk"), 2, 12, 6)
    chunk_size = st.slider(t(LANG, "chunksize"), 300, 2000, 1000, step=50)
    overlap = st.slider(t(LANG, "overlap"), 0, 400, 200, step=20)

    persist_dir = st.text_input(t(LANG, "persist"), "")

    st.markdown("---")
    uploaded = st.file_uploader(t(LANG, "upload"), type=["pdf"], accept_multiple_files=True)
    build_btn = st.button(t(LANG, "build"))
    rebuild_btn = st.button(t(LANG, "rebuild"), help=t(LANG, "rebuild_help"))
    clear_chat = st.button(t(LANG, "clear"))

    st.markdown("---")
    st.subheader(t(LANG, "chatio"))
    export_json = json.dumps(st.session_state.get("messages", []), ensure_ascii=False, indent=2)
    fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    st.download_button(t(LANG, "export"), data=export_json, file_name=fname, mime="application/json")

    imported = st.file_uploader(t(LANG, "import"), type=["json"], help=t(LANG, "import_help"))
    import_mode = st.radio(t(LANG, "import_mode"), [t(LANG, "replace"), t(LANG, "append")], horizontal=True)
    import_btn = st.button(t(LANG, "import_btn"))


# ---------------------------- Session state (chat + index) ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def load_docs_from_pdf(path: Path, chunk_size: int, overlap: int):
    """Load a PDF and return (chunks, n_pages)."""
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(pages)
    
    # Ensure proper metadata
    for chunk in chunks:
        chunk.metadata["source_file"] = Path(path).name
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = str(path)
    
    return chunks, len(pages)


# ---------------------------- Build / Load index ----------------------------
@st.cache_resource(show_spinner=False)
def _get_embeddings(embedder: str, hf_model: str):
    if embedder == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        return HuggingFaceEmbeddings(model_name=hf_model)


@st.cache_resource(show_spinner=False)
def _load_or_build_index(
    file_paths: Tuple[str, ...],
    embedder: str,
    hf_model: str,
    chunk_size: int,
    overlap: int,
    persist_dir: str | None,
):
    embeddings = _get_embeddings(embedder, hf_model)
    pages_by_file: Dict[str, int] = {}

    # Clear cache if rebuild requested
    if st.session_state.get("force_rebuild"):
        if persist_dir and persist_dir.strip():
            pd = Path(persist_dir)
            if pd.exists():
                shutil.rmtree(pd)
        st.session_state["force_rebuild"] = False

    # Try loading persisted index
    if persist_dir and persist_dir.strip():
        pd = Path(persist_dir)
        if (pd / "index.faiss").exists() and (pd / "index.pkl").exists():
            try:
                vs = FAISS.load_local(str(pd), embeddings, allow_dangerous_deserialization=True)
                all_docs = []
                for fp in file_paths:
                    try:
                        docs, n_pages = load_docs_from_pdf(Path(fp), chunk_size, overlap)
                        all_docs.extend(docs)
                        pages_by_file[Path(fp).name] = n_pages
                    except Exception:
                        pass
                return vs, pages_by_file, all_docs
            except Exception as e:
                if debug_mode:
                    st.warning(f"Error loading index: {e}")

    # Build from scratch
    all_docs = []
    for fp in file_paths:
        docs, n_pages = load_docs_from_pdf(Path(fp), chunk_size, overlap)
        all_docs.extend(docs)
        pages_by_file[Path(fp).name] = n_pages

    vs = FAISS.from_documents(all_docs, embeddings)

    if persist_dir and persist_dir.strip():
        pd = Path(persist_dir)
        pd.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(pd))

    return vs, pages_by_file, all_docs


# ---------------------------- Improved Search Functions ----------------------------
def extract_person_name(query: str, lang: str) -> str | None:
    """Extract person name from biographical queries."""
    normalized_query = re.sub(r"[\'`´]", "'", query.lower())
    normalized_query = re.sub(r"\s+", " ", normalized_query)
    
    patterns = {
        "it": [r"chi\s+(?:è|e'|era)\s+(.+)", r"chi\s+é\s+(.+)", r"parlami\s+di\s+(.+)", r"informazioni\s+su\s+(.+)", r"chi\s+sarebbe\s+(.+)", r"cos[']?è\s+(.+)", r"cosa\s+è\s+(.+)"],
        "en": [r"who\s+(?:is|was)\s+(.+)", r"tell\s+me\s+about\s+(.+)", r"information\s+about\s+(.+)", r"what\s+do\s+you\s+know\s+about\s+(.+)", r"what\s+is\s+(.+)"]
    }
    
    for pattern in patterns[lang]:
        match = re.search(pattern, normalized_query)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'[?.,!]', '', name)
            name = ' '.join(word.capitalize() for word in name.split())
            return name
    return None


def filter_docs_by_source(docs: List, target_source: str) -> List:
    """Filter documents by source file name."""
    return [doc for doc in docs if doc.metadata.get("source_file") == target_source]


def hybrid_search_for_person(person_name: str, all_docs: List, retriever, query: str, top_k: int = 8):
    """Perform hybrid search with source filtering."""
    
    # Filter by current PDF source
    current_files = list(st.session_state.get("pages_by_file", {}).keys())
    if current_files:
        target_source = current_files[0]  # Use first PDF
        all_docs = filter_docs_by_source(all_docs, target_source)
    
    # Exact match search
    exact_match_docs = []
    for doc in all_docs:
        if person_name.lower() in doc.page_content.lower():
            exact_match_docs.append(doc)
    
    # Semantic search
    try:
        semantic_docs = retriever.invoke(query)
        # Filter semantic results by source
        if current_files:
            semantic_docs = filter_docs_by_source(semantic_docs, target_source)
    except Exception:
        semantic_docs = []
    
    # Combine results
    combined_docs = exact_match_docs.copy()
    seen_sources = set((doc.metadata.get("source", ""), doc.metadata.get("page", -1)) for doc in exact_match_docs)
    
    for doc in semantic_docs:
        doc_key = (doc.metadata.get("source", ""), doc.metadata.get("page", -1))
        if doc_key not in seen_sources:
            combined_docs.append(doc)
            seen_sources.add(doc_key)
    
    return combined_docs[:top_k]


# Prepare files
tmpdir = Path(tempfile.gettempdir()) / "pdf_qa_streamlit"
tmpdir.mkdir(parents=True, exist_ok=True)

selected_files: List[Path] = []
if uploaded:
    for up in uploaded:
        out = tmpdir / up.name
        with open(out, "wb") as f:
            f.write(up.read())
        selected_files.append(out)
else:
    cwd = Path.cwd()
    selected_files = sorted(cwd.glob("*.pdf"))

if build_btn or rebuild_btn:
    if not selected_files:
        st.error(t(LANG, "no_pdf"))
    else:
        if rebuild_btn:
            st.session_state["force_rebuild"] = True
            try:
                _load_or_build_index.clear()
            except:
                pass
        
        with st.spinner(t(LANG, "building" if build_btn else "rebuilding")):
            vs, pages_by_file, all_docs = _load_or_build_index(
                tuple(str(p) for p in selected_files),
                embedder,
                hf_model,
                chunk_size,
                overlap,
                persist_dir or None,
            )
        st.session_state["vectorstore_ready"] = True
        st.session_state["retriever"] = vs.as_retriever(search_kwargs={"k": top_k})
        st.session_state["pages_by_file"] = pages_by_file
        st.session_state["all_docs"] = all_docs
        
        if debug_mode:
            st.info(f"Loaded {len(all_docs)} chunks from {len(pages_by_file)} files")
            for name, count in pages_by_file.items():
                st.info(f"📄 {name}: {count} pages")
        
        st.success(t(LANG, "ready", n=len(selected_files)))


if clear_chat:
    st.session_state["messages"] = []
    st.experimental_rerun()

if import_btn and imported is not None:
    try:
        data = json.loads(imported.read().decode("utf-8"))
        if isinstance(data, list) and all(isinstance(x, dict) and "role" in x and "content" in x for x in data):
            if import_mode == t(LANG, "replace"):
                st.session_state["messages"] = data
            else:
                st.session_state["messages"].extend(data)
            st.success(t(LANG, "ok_import"))
            st.experimental_rerun()
        else:
            st.error(t(LANG, "err_import_format"))
    except Exception as e:
        st.error(t(LANG, "err_import", e=e))


# Chat history display
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            with st.expander(t(LANG, "show_sources")):
                for s in m["sources"]:
                    if s.get("page") is not None:
                        st.caption(f"📄 {s['name']} — {t(LANG, 'p_abbr')} {s['page']}: {s['snippet']}...")
                    else:
                        st.caption(f"📄 {s['name']}: {s['snippet']}...")


# Main QA logic
query = st.chat_input(t(LANG, "chat_input"))

if st.session_state.get("pages_by_file"):
    with st.sidebar:
        st.markdown("---")
        st.subheader(t(LANG, "pages_header"))
        for name, n in st.session_state["pages_by_file"].items():
            st.caption(f"**{name}** — {n} {'pagine' if LANG=='it' else 'pages'}")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Clean query for pattern matching
    clean_query = re.sub(r'[?.,!]', '', query.lower().strip())

    # 1. Page count questions (FIRST - highest priority)
    if any(k in clean_query for k in LANGS[LANG]["howmany_patterns"]):
        pmap = st.session_state.get("pages_by_file", {})
        if not pmap:
            answer = t(LANG, "pages_unknown")
        else:
            unit = "pagine" if LANG == "it" else "pages"
            lines = [f"**{name}**: {n} {unit}" for name, n in pmap.items()]
            answer = "\n".join(lines)
        
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
        st.stop()

    # 2. Biographical queries (SECOND)
    person_name = extract_person_name(clean_query, LANG)
    if person_name and st.session_state.get("vectorstore_ready"):
        all_docs = st.session_state.get("all_docs", [])
        retriever = st.session_state["retriever"]
        
        retrieved_docs = hybrid_search_for_person(person_name, all_docs, retriever, query, top_k + 4)
        context = "\n\n".join(d.page_content for d in retrieved_docs)
        
        if context.strip():
            full_prompt = t(LANG, "bio_prompt").format(name=person_name, context=context)
        else:
            answer = t(LANG, "no_info_person", name=person_name)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
            st.stop()
        
        # LLM call
        try:
            if mode_llm == "ollama":
                llm = OllamaLLM(model=ollama_model, temperature=0)
                answer = llm.invoke(full_prompt)
            else:
                llm = ChatOpenAI(model=openai_model, temperature=0)
                answer = llm.invoke(full_prompt).content
        except Exception as e:
            if os.environ.get("OPENAI_API_KEY"):
                llm = ChatOpenAI(model=openai_model, temperature=0)
                answer = llm.invoke(full_prompt).content
            else:
                st.error("LLM not available")
                st.stop()

        # Sources
        src_list = []
        seen = set()
        for d in retrieved_docs:
            src = d.metadata.get("source_file", "?")
            page = d.metadata.get("page")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                snippet = d.page_content.strip().replace("\n", " ")[:200]
                src_list.append({"name": src, "page": (page + 1) if page is not None else None, "snippet": snippet})
                if len(src_list) >= 8: break

        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": src_list})
        with st.chat_message("assistant"):
            st.write(answer)
            if debug_mode and src_list:
                with st.expander("🔍 Debug Sources"):
                    for s in src_list:
                        st.write(f"Source: {s}")
            if src_list:
                with st.expander(t(LANG, "show_sources")):
                    for s in src_list:
                        if s.get("page"):
                            st.caption(f"📄 {s['name']} — {t(LANG, 'p_abbr')} {s['page']}: {s['snippet']}...")
                        else:
                            st.caption(f"📄 {s['name']}: {s['snippet']}...")
        st.stop()

    # 3. Page-specific queries (THIRD)
    m = re.search(r"(?:pag(?:ina)?|page)\s*(\d+)", clean_query)
    if m:
        page_num = int(m.group(1))
        target_page_idx = page_num - 1
        all_docs = st.session_state.get("all_docs", [])

        if not all_docs:
            answer = t(LANG, "pages_unknown")
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
            st.stop()

        # Filter by current PDF source
        current_files = list(st.session_state.get("pages_by_file", {}).keys())
        if current_files:
            target_source = current_files[0]
            all_docs = filter_docs_by_source(all_docs, target_source)

        page_docs = [d for d in all_docs if d.metadata.get("page") == target_page_idx]

        if not page_docs:
            max_page = max([d.metadata.get("page", 0) for d in all_docs]) if all_docs else 0
            max_page_num = max_page + 1
            if page_num > max_page_num:
                answer = t(LANG, "page_not_found", page_num=page_num, max_page=max_page_num)
            else:
                answer = f"{'La pagina' if LANG == 'it' else 'Page'} {page_num} {'esiste ma non contiene testo estraibile.' if LANG == 'it' else 'exists but contains no extractable text.'}"
            
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
            st.stop()

        context = "\n\n".join(d.page_content for d in page_docs)
        full_prompt = t(LANG, "strict_prompt").format(context=context, question=query)

        # LLM call
        try:
            if mode_llm == "ollama":
                llm = OllamaLLM(model=ollama_model, temperature=0)
                answer = llm.invoke(full_prompt)
            else:
                llm = ChatOpenAI(model=openai_model, temperature=0)
                answer = llm.invoke(full_prompt).content
        except Exception as e:
            if os.environ.get("OPENAI_API_KEY"):
                llm = ChatOpenAI(model=openai_model, temperature=0)
                answer = llm.invoke(full_prompt).content
            else:
                st.error("LLM not available")
                st.stop()

        src_list = []
        seen = set()
        for d in page_docs:
            src = d.metadata.get("source_file", "?")
            page = d.metadata.get("page")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                snippet = d.page_content.strip().replace("\n", " ")[:200]
                src_list.append({"name": src, "page": (page + 1) if page is not None else None, "snippet": snippet})

        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": src_list})
        with st.chat_message("assistant"):
            st.write(answer)
            if src_list:
                with st.expander(t(LANG, "show_sources")):
                    for s in src_list:
                        if s.get("page"):
                            st.caption(f"📄 {s['name']} — {t(LANG, 'p_abbr')} {s['page']}: {s['snippet']}...")
                        else:
                            st.caption(f"📄 {s['name']}: {s['snippet']}...")
        st.stop()

    # 4. Normal RAG flow (LAST)
    if not st.session_state.get("vectorstore_ready"):
        answer = t(LANG, "must_build")
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
    else:
        retriever = st.session_state["retriever"]
        
        # Filter by current PDF source for normal queries too
        current_files = list(st.session_state.get("pages_by_file", {}).keys())
        if current_files:
            target_source = current_files[0]
            # Create a filtered retriever
            all_docs = st.session_state.get("all_docs", [])
            filtered_docs = filter_docs_by_source(all_docs, target_source)
            if filtered_docs:
                # Create temporary vectorstore with filtered docs
                embeddings = _get_embeddings(embedder, hf_model)
                temp_vs = FAISS.from_documents(filtered_docs, embeddings)
                retriever = temp_vs.as_retriever(search_kwargs={"k": top_k})
        
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in retrieved_docs)
        
        full_prompt = t(LANG, "strict_prompt").format(context=context, question=query)

        if mode_llm == "openai":
            llm = ChatOpenAI(model=openai_model, temperature=0)
            answer = llm.invoke(full_prompt).content
        else:
            try:
                llm = OllamaLLM(model=ollama_model, temperature=0)
                answer = llm.invoke(full_prompt)
            except Exception:
                if os.environ.get("OPENAI_API_KEY"):
                    llm = ChatOpenAI(model=openai_model, temperature=0)
                    answer = llm.invoke(full_prompt).content
                else:
                    st.error("LLM not available")
                    st.stop()

        src_list = []
        seen = set()
        for d in retrieved_docs:
            src = d.metadata.get("source_file", "?")
            page = d.metadata.get("page")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                snippet = d.page_content.strip().replace("\n", " ")[:200]
                src_list.append({"name": src, "page": (page + 1) if page is not None else None, "snippet": snippet})
                if len(src_list) >= 6: break

        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": src_list})

        with st.chat_message("assistant"):
            st.write(answer)
            if src_list:
                with st.expander(t(LANG, "show_sources")):
                    for s in src_list:
                        if s.get("page"):
                            st.caption(f"📄 {s['name']} — {t(LANG, 'p_abbr')} {s['page']}: {s['snippet']}...")
                        else:
                            st.caption(f"📄 {s['name']}: {s['snippet']}...")

st.markdown("---")
st.caption(t(LANG, "footer"))
