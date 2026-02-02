
import os
import re

docs_dir = r"c:\Users\sesa500059\OneDrive - Schneider Electric\Desktop\sj\sjv\AutoChunks\autochunks_repo\docs"

def find_links(content):
    # Match [label](path)
    return re.findall(r'\[.*?\]\((.*?)\)', content)

all_files = []
for root, dirs, files in os.walk(docs_dir):
    for file in files:
        if file.endswith(".md"):
            all_files.append(os.path.join(root, file))

results = []

for file_path in all_files:
    rel_path = os.path.relpath(file_path, docs_dir)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        links = find_links(content)
        for link in links:
            if link.startswith("http") or link.startswith("#") or link.startswith("mailto:"):
                continue
            
            # Resolve relative path
            link_path = os.path.normpath(os.path.join(os.path.dirname(file_path), link))
            
            # Check if it's a file or anchor in file
            base_link = link_path.split("#")[0]
            if not os.path.exists(base_link) and not os.path.exists(base_link + ".md"):
                results.append(f"Dead link in {rel_path}: {link} (Resolved to: {base_link})")

if not results:
    print("No dead internal links found.")
else:
    for r in results:
        print(r)
