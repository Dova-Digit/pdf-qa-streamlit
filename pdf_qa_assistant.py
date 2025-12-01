# --- Disabilita TensorFlow/JAX per evitare problemi con Keras 3 ---
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import os
from pathlib import Path
import argparse
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Carica .env (OPENAI_API_KEY se usi --embedder openai o --llm openai)
load_dotenv()

# ---------------- Helpers ----------------
def list_pdfs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.pdf"))

def load_docs_from_pdf(pdf_file: Path, chunk_size: int = 1000, overlap: int = 200) -> Tuple[List, int]:
    """Carica un PDF, fa chunk e aggiunge metadati 'source'. Ritorna (docs, num_pagine)."""
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.split_documents(pages)
    for d in docs:
        d.metadata.setdefault("source", str(pdf_file))
    return docs, len(pages)

def choose_pdf(interactive: bool = True, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"PDF not found: {p}")
        return p

    pdfs = list_pdfs(Path.cwd())
    if not pdfs:
        if not interactive:
            raise FileNotFoundError("No .pdf found in current folder and no --pdf provided.")
        print("[!] No .pdf found in current folder.\n")
        while True:
            user_in = input("Enter path to a PDF file (or drag & drop here): ").strip().strip('"')
            if not user_in:
                continue
            p = Path(user_in)
            if p.is_file():
                return p
            print(f"[!] '{p}' is not a valid file. Try again.\n")

    if len(pdfs) == 1 or not interactive:
        return pdfs[0]

    print("Found multiple PDFs in the folder:\n")
    for i, p in enumerate(pdfs, 1):
        print(f"  [{i}] {p.name}")
    print("")
    while True:
        pick = input(f"Select 1-{len(pdfs)} (Enter for 1): ").strip()
        if pick == "":
            return pdfs[0]
        if pick.isdigit() and 1 <= int(pick) <= len(pdfs):
            return pdfs[int(pick) - 1]
        print("Invalid selection. Try again.\n")

def join_docs(docs: List) -> str:
    return "\n\n".join(d.page_content for d in docs)

def show_sources(docs: List, max_sources: int = 6, snippet_chars: int = 160):
    print("\nSources:")
    seen = set()
    count = 0
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", None)
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        count += 1
        snippet = d.page_content.strip().replace("\n", " ")[:snippet_chars]
        if page is not None:
            print(f" - {Path(src).name} | page {page + 1}: {snippet}...")
        else:
            print(f" - {Path(src).name}: {snippet}...")
        if count >= max_sources:
            break

def retrieve(retriever, q):
    """Compat: usa get_relevant_documents() (vecchie versioni) oppure invoke() (nuove versioni)."""
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(q)
    return retriever.invoke(q)

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="PDF QA Assistant (LangChain + FAISS)")
parser.add_argument("--pdf", type=str, default=None, help="Path a un PDF specifico. Se omesso, auto-detect nella cartella.")
parser.add_argument("--all-pdfs", action="store_true", help="Indicizza TUTTI i PDF della cartella in un unico indice")
parser.add_argument("--persist", type=str, default=None, help="Cartella dove salvare/ricaricare l'indice FAISS")
parser.add_argument("--no-input", action="store_true", help="Nessun prompt interattivo; usa il primo PDF trovato.")
parser.add_argument("--embedder", type=str, default="hf", choices=["openai", "hf"], help="Embeddings: 'openai' (a pagamento) o 'hf' (HuggingFace locale)")
parser.add_argument("--hf-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modello HF per embeddings")
parser.add_argument("--llm", type=str, default="ollama", choices=["openai", "ollama"], help="LLM per la generazione")
parser.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="Modello OpenAI (se --llm openai)")
parser.add_argument("--ollama-model", type=str, default="mistral", help="Modello Ollama (se --llm ollama)")
parser.add_argument("--top-k", type=int, default=4, help="Quanti chunk recuperare")
parser.add_argument("--show-sources", action="store_true", help="Mostra fonti (file + pagina) con snippet")
args = parser.parse_args()

# ---------------- Caricamento PDF(s) ----------------
all_docs: List = []
pages_by_file: dict[str, int] = {}

if args.all_pdfs:
    pdfs = list_pdfs(Path.cwd()) if args.pdf is None else [Path(args.pdf)]
    if not pdfs:
        raise SystemExit("[ERROR] No PDFs found in the current folder.")
    print(f"[OK] Indexing {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(" -", p.name)
        _docs, _npages = load_docs_from_pdf(p)
        all_docs.extend(_docs)
        pages_by_file[str(p)] = _npages
else:
    pdf_path = choose_pdf(interactive=(not args.no_input), explicit=args.pdf)
    print(f"[OK] Using PDF: {pdf_path.resolve()}\n")
    _docs, _npages = load_docs_from_pdf(pdf_path)
    all_docs.extend(_docs)
    pages_by_file[str(pdf_path)] = _npages

# ---------------- Embeddings ----------------
if args.embedder == "openai":
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("[OK] Using OpenAI embeddings: text-embedding-3-small")
else:
    embeddings = HuggingFaceEmbeddings(model_name=args.hf_model)
    print(f"[OK] Using HuggingFace embeddings: {args.hf_model}")

# ---------------- FAISS: load o build + persist ----------------
vectorstore = None
if args.persist:
    persist_path = Path(args.persist)
    faiss_index_file = persist_path / "index.faiss"
    faiss_store_file = persist_path / "index.pkl"
    if faiss_index_file.exists() and faiss_store_file.exists():
        print(f"[OK] Loading FAISS from: {persist_path}")
        vectorstore = FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"[OK] Building FAISS and saving to: {persist_path}")
        persist_path.mkdir(parents=True, exist_ok=True)
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(str(persist_path))
else:
    print("[OK] Building FAISS in-memory (not persisted)")
    vectorstore = FAISS.from_documents(all_docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": args.top_k})

# ---------------- LLM ----------------
if args.llm == "openai":
    llm = ChatOpenAI(model=args.openai_model, temperature=0)
    print(f"[OK] Using OpenAI LLM: {args.openai_model}")
else:
    llm = OllamaLLM(model=args.ollama_model, temperature=0)
    print(f"[OK] Using Ollama LLM: {args.ollama_model}")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Usa il seguente contesto per rispondere. Se non sai, dillo.\nContesto:\n{context}"),
    ("human", "{question}")
])

parser_out = StrOutputParser()

# Risposta rapida: numero pagine (se richiesto)
def _maybe_answer_total_pages(query: str) -> bool:
    lowq = query.lower().strip()
    keys = ["quante pagine", "numero di pagine", "totale pagine", "quanti fogli"]
    if any(k in lowq for k in keys):
        print("\nAnswer:")
        for src, n in pages_by_file.items():
            print(f" - {Path(src).name}: {n} pagine")
        return True
    return False

# ---------------- Loop Q&A ----------------
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # quick path per domande sul numero di pagine
    if _maybe_answer_total_pages(query):
        continue

    retrieved_docs = retrieve(retriever, query)
    context = join_docs(retrieved_docs)

    messages = prompt.invoke({"context": context, "question": query})
    answer = parser_out.invoke(llm.invoke(messages))

    print("\nAnswer:", answer)
    if args.show_sources:
        show_sources(retrieved_docs)
