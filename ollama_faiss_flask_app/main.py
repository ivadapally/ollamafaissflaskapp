
from flask import Flask, request, jsonify, render_template
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import pdfplumber
from scholarly import scholarly

app = Flask(__name__)

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
papers = []

def fetch_arxiv_papers(query, max_results=1):
    url = f"https://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
    response = requests.get(url)
    papers.clear()
    for entry in response.text.split('<entry>')[1:]:
        title = entry.split('<title>')[1].split('</title>')[0].strip()
        abstract = entry.split('<summary>')[1].split('</summary>')[0].strip()
        link = entry.split('<id>')[1].split('</id>')[0].strip()
        papers.append({'title': title, 'abstract': abstract, 'link': link})
        print(f"📄 Title: {title}\n🔗 Link: {link}\n📚 Abstract: {abstract[:100]}...\n")

def add_to_faiss():
    global faiss_index
    documents = [paper['abstract'] for paper in papers]
    embeddings = model.encode(documents)
    faiss_index.add(np.array(embeddings, dtype='float32'))

def search_faiss(query, top_k=3):
    global faiss_index
    query_embedding = model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding, dtype='float32'), k=top_k)
    return [papers[i] for i in I[0] if i != -1]

def summarize_with_llama(text):
    ollama_endpoint = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3",
        "messages": [{"role": "user", "content": f"Summarize this: {text}"}],
        "stream": True
    }

    try:
        response = requests.post(ollama_endpoint, json=payload, stream=True)
        response.raise_for_status()

        # Collect streamed chunks line-by-line
        summary = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    summary += chunk.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    continue  # skip bad chunks
        return summary or "⚠️ No content received."

    except requests.exceptions.ConnectionError:
        return "⚠️ Error: Unable to connect to Ollama."


def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "".join(page.extract_text() or "" for page in pdf.pages).strip()

def generate_citation(title):
    search_query = scholarly.search_pubs(title)
    try:
        paper_info = next(search_query)
        bib = paper_info.get("bib", {})
        author = bib.get("author", "Unknown Author")
        year = bib.get("year", "n.d.")
        paper_title = bib.get("title", title)
        journal = bib.get("journal", "Unknown Journal")
        return f"{author} ({year}). {paper_title}. {journal}."
    except StopIteration:
        return "No citation found."


@app.route('/')
def index():
    return render_template('index.html', papers=papers)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    results = search_faiss(query)
    for paper in results:
        paper['summary'] = summarize_with_llama(paper['abstract'])
    return render_template('results.html', results=results)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    extracted_text = extract_text_from_pdf(file)
    summary = summarize_with_llama(extracted_text)
    return jsonify({"summary": summary})

@app.route('/citation', methods=['POST'])
def citation():
    title = request.form.get('title')
    citation = generate_citation(title)
    return jsonify({"citation": citation})

@app.route('/papers')
def show_papers():
    return render_template('papers.html', papers=papers)

@app.route('/summarize', methods=['POST'])
def summarize():
    abstract = request.json.get('abstract')
    if not abstract:
        return jsonify({"summary": "❌ No abstract provided."}), 400
    summary = summarize_with_llama(abstract)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    fetch_arxiv_papers("Model Context Protocol")
    add_to_faiss()
    app.run(debug=True, port=5000, use_reloader=False)
