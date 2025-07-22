import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tree_sitter import Language, Parser
import tree_sitter_cpp
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from pathlib import Path
from langchain_core.documents import Document
import pdfplumber
from dotenv import load_dotenv
from config import DOCUMENTS_PATH, SOURCES_PATH

load_dotenv()

DOCUMENTS_FULL_PATH = os.path.expanduser(DOCUMENTS_PATH)
SOURCES_FULL_PATH = os.path.expanduser(SOURCES_PATH)


# Step 2: Load Files (Markdown, C/C++, PDF)
def load_text_files(directory, extensions):
    documents = []
    directory_path = Path(directory)
    for ext in extensions:
        for file_path in directory_path.rglob(f"*.{ext}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": str(file_path), "type": f"{ext}"}
                ))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return documents

def load_pdf_files(directory):
    documents = []
    directory_path = Path(directory)
    for file_path in directory_path.rglob("*.pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                tot_pages = len(pdf.pages)
                num_page=1
                print(f"Reading {file_path}")
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    print(f"Page {num_page}/{tot_pages}")
                    num_page+=1
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": str(file_path), "type": "pdf"}
                    ))
                print(f"{file_path} loaded OK.")                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return documents

# Load Markdown, C/C++, and PDF files
markdown_files = load_text_files(DOCUMENTS_FULL_PATH, ["md"])
cpp_files = load_text_files(SOURCES_FULL_PATH, ["c", "cpp", "h", "hpp"])
pdf_files = load_pdf_files(DOCUMENTS_FULL_PATH)

# Combine documents
documents = markdown_files + cpp_files + pdf_files
print(f"Loaded {len(documents)} files (Markdown: {len(markdown_files)}, C/C++: {len(cpp_files)}, PDF: {len(pdf_files)})")

# Step 3: Chunk the Documents
# Chunk Markdown files
def chunk_markdown(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "#", "##", "###"]
    )
    markdown_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            markdown_chunks.append({
                "content": chunk,
                "metadata": {
                    **doc.metadata,
                    "chunk_id": f"{doc.metadata['source']}_{i}",
                    "type": "markdown"
                }
            })
    return markdown_chunks

# Chunk PDF files
def chunk_pdf(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ","]  # More granular for PDFs
    )
    pdf_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            pdf_chunks.append({
                "content": chunk,
                "metadata": {
                    **doc.metadata,
                    "chunk_id": f"{doc.metadata['source']}_{i}",
                    "type": "pdf"
                }
            })
    return pdf_chunks

# Chunk C/C++ files with Tree-Sitter
def chunk_cpp(documents):
    CPP_LANGUAGE = tree_sitter_cpp.language()
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "}"]
    )

    cpp_chunks = []

    for doc in documents:
        source = doc.page_content.encode('utf-8')
        tree = parser.parse(source)
        root_node = tree.root_node

        def get_node_text(node):
            return source[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

        def traverse_node(node, depth=0):
            chunks = []
            node_type = node.type
            node_text = get_node_text(node)

            if node_type in ('function_definition', 'method_definition'):
                name = "unknown"
                for child in node.children:
                    if child.type in ('declarator', 'function_declarator'):
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                name = get_node_text(subchild)
                                break
                chunks.append({
                    "content": node_text,
                    "metadata": {
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['source']}_{node.start_byte}_function",
                        "type": "code",
                        "section": "function",
                        "name": name
                    }
                })

            elif node_type in ('class_specifier', 'struct_specifier'):
                name = "unknown"
                for child in node.children:
                    if child.type == 'identifier':
                        name = get_node_text(child)
                        break
                chunks.append({
                    "content": node_text,
                    "metadata": {
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['source']}_{node.start_byte}_class",
                        "type": "code",
                        "section": node_type,
                        "name": name
                    }
                })

            for child in node.children:
                chunks.extend(traverse_node(child, depth + 1))

            return chunks

        ast_chunks = traverse_node(root_node)

        non_ast_content = []
        last_end = 0
        for chunk in ast_chunks:
            start_byte = source.index(chunk['content'].encode('utf-8'), last_end)
            if start_byte > last_end:
                non_ast_content.append(source[last_end:start_byte].decode('utf-8', errors='ignore'))
            last_end = start_byte + len(chunk['content'].encode('utf-8'))

        if last_end < len(source):
            non_ast_content.append(source[last_end:].decode('utf-8', errors='ignore'))

        for i, content in enumerate(non_ast_content):
            if content.strip():
                sub_chunks = text_splitter.split_text(content)
                for j, sub_chunk in enumerate(sub_chunks):
                    ast_chunks.append({
                        "content": sub_chunk,
                        "metadata": {
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata['source']}_non_ast_{i}_{j}",
                            "type": "code",
                            "section": "general"
                        }
                    })

        cpp_chunks.extend(ast_chunks)

    return cpp_chunks

# Chunk files
markdown_chunks = chunk_markdown(markdown_files)
cpp_chunks = chunk_cpp(cpp_files)
pdf_chunks = chunk_pdf(pdf_files)
all_chunks = markdown_chunks + cpp_chunks + pdf_chunks
print(f"Created {len(all_chunks)} chunks (Markdown: {len(markdown_chunks)}, C/C++: {len(cpp_chunks)}, PDF: {len(pdf_chunks)})")

# Step 4: Generate Embeddings with OpenAI API
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

texts = [chunk["content"] for chunk in all_chunks]
metadatas = [chunk["metadata"] for chunk in all_chunks]

vector_store = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)

vector_store.save_local("faiss_index")
print("Embeddings generated and stored in FAISS index")

# Step 5: Example Retrieval and Response Generation
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(query, vector_store):
    docs = vector_store.similarity_search(query, k=5)
    context = "\n".join([f"[{doc.metadata['source']} ({doc.metadata['type']})]: {doc.page_content}" for doc in docs])
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided code and documentation. Use the context to provide accurate answers."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

# Test query
query = "How do I create a rigid body in Bullet3?"
response = generate_response(query, vector_store)
print(f"Query: {query}")
print(f"Response: {response}")