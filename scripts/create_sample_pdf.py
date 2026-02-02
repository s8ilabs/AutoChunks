
import fitz  # PyMuPDF
import os

def create_sample_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    
    # Add a Heading
    page.insert_text((50, 50), "Chunking Strategy Deep Dive", fontsize=20, color=(0, 0, 1))
    
    # Add some paragraphs
    text = """
    Retrieval-Augmented Generation (RAG) relies heavily on high-quality text chunking. 
    Standard chunking methods often ignore the structural elements of a document, such as headers, 
    tables, and lists. This can lead to chunks that are semantically incomplete or confusing 
    to the LLM during the synthesis stage.
    """
    page.insert_textbox((50, 80, 550, 200), text, fontsize=12)
    
    # Add a Table-like structure
    page.insert_text((50, 220), "Table 1: Chunking Comparison", fontsize=14, color=(1, 0, 0))
    table_data = [
        ["Strategy", "Benefit", "Drawback"],
        ["Fixed-Length", "Simple, Fast", "Breaks context"],
        ["Recursive", "Structural", "Heuristic-based"],
        ["Semantic", "Contextual", "Expensive"],
        ["Layout-Aware", "High-Fidelity", "Requires OCR/Parsing"]
    ]
    
    y = 250
    for row in table_data:
        x = 50
        for cell in row:
            page.insert_text((x, y), cell, fontsize=10)
            x += 150
        y += 20
        
    doc.save(path)
    doc.close()

if __name__ == "__main__":
    os.makedirs("./examples/pdf_docs", exist_ok=True)
    create_sample_pdf("./examples/pdf_docs/report_finance.pdf")
    create_sample_pdf("./examples/pdf_docs/report_ops.pdf")
    create_sample_pdf("./examples/pdf_docs/report_tech.pdf")
    print("Created 3 sample PDFs in ./examples/pdf_docs/")
