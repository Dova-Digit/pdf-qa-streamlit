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

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        "howmany_patterns": ["quante pagine", "numero di pagine", "totale pagine", "quanti fogli"],
        "sys_prompt": "Usa il seguente contesto per rispondere. Se non sai, dillo.\\nContesto:\\n{context}",
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
        "sys_prompt": "Use the given context to answer. If you don't know, say you don't know.\\nContext:\\n{context}",
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

with st.sidebar:
    st.header(t(LANG, "config"))
    mode_llm = st.selectbox(t(LANG, "llm"), ["ollama", "openai"], index=0)
    ollama_model = st.text_input(t(LANG, "ollama_model"), "mistral")
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


# ---------------------------- Build / Load index ----------------------------
@st.cache_resource(show_spinner=False)
def _get_embeddings(embedder: str, hf_model: str):
    if embedder == "openai":
        st.info("Using OpenAI embeddings: text-embedding-3-small")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        st.info(f"Using HF embeddings: {hf_model}")
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

    # 1) Try loading a persisted index
    if persist_dir:
        pd = Path(persist_dir)
        if (pd / "index.faiss").exists() and (pd / "index.pkl").exists():
            vs = FAISS.load_local(str(pd), embeddings, allow_dangerous_deserialization=True)
            # Recompute page counts quickly from original PDFs
            for fp in file_paths:
                try:
                    _docs, _np = load_docs_from_pdf(Path(fp), chunk_size, overlap)
                    pages_by_file[Path(fp).name] = _np
                except Exception:
                    pass
            st.session_state["pages_by_file"] = pages_by_file
            return vs

    # 2) Otherwise build from scratch
    all_docs = []
    for fp in file_paths:
        docs, n_pages = load_docs_from_pdf(Path(fp), chunk_size, overlap)
        all_docs.extend(docs)
        pages_by_file[Path(fp).name] = n_pages

    st.session_state["pages_by_file"] = pages_by_file

    vs = FAISS.from_documents(all_docs, embeddings)

    if persist_dir:
        pd = Path(persist_dir)
        pd.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(pd))

    return vs


# prepare temp files for uploaded PDFs
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
    # auto-detect PDFs in current working dir
    cwd = Path.cwd()
    selected_files = sorted(cwd.glob("*.pdf"))

if build_btn:
    if not selected_files:
        st.error(t(LANG, "no_pdf"))
    else:
        with st.spinner(t(LANG, "building")):
            vs = _load_or_build_index(
                tuple(str(p) for p in selected_files),
                embedder,
                hf_model,
                chunk_size,
                overlap,
                persist_dir or None,
            )
        st.session_state["vectorstore_ready"] = True
        st.session_state["retriever"] = vs.as_retriever(search_kwargs={"k": top_k})
        st.success(t(LANG, "ready", n=len(selected_files)))


# ---------------------------- Rebuild index ----------------------------
if rebuild_btn:
    files_for_build: List[Path] = selected_files
    if not files_for_build:
        st.error(t(LANG, "no_pdf"))
    else:
        if persist_dir:
            pd = Path(persist_dir)
            if pd.exists():
                try:
                    shutil.rmtree(pd)
                    st.info(t(LANG, "removed", p=pd))
                except Exception as e:
                    st.error(f"{e}")
        try:
            _load_or_build_index.clear()
            _get_embeddings.clear()
        except Exception:
            pass
        st.session_state.pop("vectorstore_ready", None)
        st.session_state.pop("retriever", None)

        with st.spinner(t(LANG, "rebuilding")):
            vs = _load_or_build_index(
                tuple(str(p) for p in files_for_build),
                embedder,
                hf_model,
                chunk_size,
                overlap,
                persist_dir or None,
            )
        st.session_state["vectorstore_ready"] = True
        st.session_state["retriever"] = vs.as_retriever(search_kwargs={"k": top_k})
        st.success(t(LANG, "rebuilt"))


# ---------------------------- Clear chat ----------------------------
if clear_chat:
    st.session_state["messages"] = []
    st.experimental_rerun()

# ---------------------------- Import chat ----------------------------
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


# ---------------------------- Chat history (render) ----------------------------
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


# ---------------------------- QA UI ----------------------------
query = st.chat_input(t(LANG, "chat_input"))

# Sidebar stats: pages per file (if available)
if st.session_state.get("pages_by_file"):
    with st.sidebar:
        st.markdown("---")
        st.subheader(t(LANG, "pages_header"))
        for name, n in st.session_state["pages_by_file"].items():
            st.caption(f"**{name}** — {n} {'pagine' if LANG=='it' else 'pages'}")

if query:
    # show user msg and store
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Quick answer path: page count questions
    lowq = query.lower().strip()
    if any(k in lowq for k in LANGS[LANG]["howmany_patterns"]):
        pmap = st.session_state.get("pages_by_file", {})
        if not pmap:
            answer = t(LANG, "pages_unknown")
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
        else:
            unit = "pagine" if LANG == "it" else "pages"
            lines = [f"**{name}**: {n} {unit}" for name, n in pmap.items()]
            answer = "\n".join(lines)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
        st.stop()

    # Normal RAG flow
    if not st.session_state.get("vectorstore_ready"):
        answer = t(LANG, "must_build")
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
    else:
        retriever = st.session_state["retriever"]
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in retrieved_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", t(LANG, "sys_prompt")),
            ("human", "{question}"),
        ])
        messages = prompt.invoke({"context": context, "question": query})

        llm = ChatOpenAI(model=openai_model, temperature=0) if mode_llm == "openai" else OllamaLLM(model=ollama_model, temperature=0)
        answer = StrOutputParser().invoke(llm.invoke(messages))

        # store + render assistant message with sources
        src_list: List[Dict[str, Any]] = []
        seen = set()
        for d in retrieved_docs:
            src = Path(d.metadata.get("source", "?")).name
            page = d.metadata.get("page")
            key = (src, page)
            if key in seen:
                continue
            seen.add(key)
            snippet = d.page_content.strip().replace("\n", " ")[:200]
            src_list.append({"name": src, "page": (page + 1) if page is not None else None, "snippet": snippet})
            if len(src_list) >= 6:
                break

        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": src_list})

        with st.chat_message("assistant"):
            st.write(answer)
            if src_list:
                with st.expander(t(LANG, "show_sources")):
                    for s in src_list:
                        if s.get("page") is not None:
                            st.caption(f"📄 {s['name']} — {t(LANG, 'p_abbr')} {s['page']}: {s['snippet']}...")
                        else:
                            st.caption(f"📄 {s['name']}: {s['snippet']}...")

# Footer
st.markdown("---")
st.caption(t(LANG, "footer"))