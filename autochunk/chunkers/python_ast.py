
from __future__ import annotations
from typing import List, Optional
import ast
from .base import BaseChunker, Chunk
from ..utils.text import count_tokens

class PythonASTChunker(BaseChunker):
    """
    AST-Based Python Code Chunker.
    
    Uses Python's Abstract Syntax Tree to split code at natural boundaries
    (classes, functions, imports) rather than arbitrary line counts.
    
    BEST-OF-BREED FEATURES:
    1. Structural Awareness: Splits at class/function boundaries.
    2. Docstring Preservation: Keeps docstrings with their functions.
    3. Import Grouping: Groups imports together.
    4. Nested Handling: Handles nested classes and functions.
    5. Context Prepending: Optionally prepends module/class context.
    """
    name = "python_ast"

    def __init__(self,
                 include_imports_in_all: bool = True,
                 split_classes: bool = True,
                 split_functions: bool = True,
                 max_tokens: int = 1000,
                 prepend_context: bool = True):
        """
        Initialize the Python AST chunker.
        
        Args:
            include_imports_in_all: If True, prepend imports to every chunk.
            split_classes: If True, split classes into separate chunks.
            split_functions: If True, split functions into separate chunks.
            max_tokens: Maximum tokens per chunk (will further split if exceeded).
            prepend_context: If True, prepend class name to method chunks.
        """
        self.include_imports_in_all = include_imports_in_all
        self.split_classes = split_classes
        self.split_functions = split_functions
        self.max_tokens = max_tokens
        self.prepend_context = prepend_context

    def chunk(self,
              doc_id: str,
              text: str,
              **params) -> List[Chunk]:
        """
        Parse Python code and split at structural boundaries.
        
        Args:
            doc_id: Document identifier
            text: Python source code
        
        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []
        
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Fallback to line-based splitting for invalid Python
            from .recursive_character import RecursiveCharacterChunker
            return RecursiveCharacterChunker().chunk(doc_id, text, base_token_size=self.max_tokens)
        
        lines = text.split("\n")
        chunks = []
        
        # Extract imports
        imports = []
        import_lines = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno'):
                    start = node.lineno - 1
                    end = getattr(node, 'end_lineno', node.lineno)
                    import_text = "\n".join(lines[start:end])
                    imports.append(import_text)
                    for i in range(start, end):
                        import_lines.add(i)
        
        import_block = "\n".join(imports)
        
        # Process top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._process_class(doc_id, node, lines, import_block, len(chunks)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(self._process_function(doc_id, node, lines, import_block, len(chunks), None))
        
        # If no structures found, treat as single chunk
        if not chunks:
            return [Chunk(
                id=f"{doc_id}#py#0",
                doc_id=doc_id,
                text=text,
                meta={"chunk_index": 0, "strategy": "python_ast", "type": "module"}
            )]
        
        return chunks

    def _process_class(self, doc_id: str, node: ast.ClassDef, lines: List[str], 
                       import_block: str, start_idx: int) -> List[Chunk]:
        """Process a class definition."""
        chunks = []
        
        start = node.lineno - 1
        end = node.end_lineno
        class_text = "\n".join(lines[start:end])
        class_name = node.name
        
        # Get class docstring
        docstring = ast.get_docstring(node) or ""
        
        if not self.split_classes:
            # Return whole class as one chunk
            full_text = class_text
            if self.include_imports_in_all and import_block:
                full_text = import_block + "\n\n" + class_text
            
            chunks.append(Chunk(
                id=f"{doc_id}#py#{start_idx}",
                doc_id=doc_id,
                text=full_text,
                meta={
                    "chunk_index": start_idx,
                    "strategy": "python_ast",
                    "type": "class",
                    "name": class_name,
                    "docstring": docstring[:200],
                    "token_count": count_tokens(full_text)
                }
            ))
            return chunks
        
        # Process methods within class
        methods_processed = set()
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_chunks = self._process_function(
                    doc_id, item, lines, import_block, 
                    start_idx + len(chunks), class_name
                )
                chunks.extend(method_chunks)
                methods_processed.add(item.lineno)
        
        # If no methods or class has significant non-method content
        if not methods_processed:
            full_text = class_text
            if self.include_imports_in_all and import_block:
                full_text = import_block + "\n\n" + class_text
            
            chunks.append(Chunk(
                id=f"{doc_id}#py#{start_idx}",
                doc_id=doc_id,
                text=full_text,
                meta={
                    "chunk_index": start_idx,
                    "strategy": "python_ast",
                    "type": "class",
                    "name": class_name,
                    "token_count": count_tokens(full_text)
                }
            ))
        
        return chunks

    def _process_function(self, doc_id: str, node, lines: List[str],
                          import_block: str, idx: int, class_name: Optional[str]) -> List[Chunk]:
        """Process a function definition."""
        start = node.lineno - 1
        end = node.end_lineno
        func_text = "\n".join(lines[start:end])
        func_name = node.name
        
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Build context prefix
        context_prefix = ""
        if self.prepend_context and class_name:
            context_prefix = f"# Method of class: {class_name}\n"
        
        # Build full text
        full_text = func_text
        if context_prefix:
            full_text = context_prefix + full_text
        if self.include_imports_in_all and import_block:
            full_text = import_block + "\n\n" + full_text
        
        # Check if needs further splitting
        if count_tokens(full_text) > self.max_tokens:
            # Split large functions by logical blocks
            from .recursive_character import RecursiveCharacterChunker
            sub_chunker = RecursiveCharacterChunker()
            sub_chunks = sub_chunker.chunk(
                f"{doc_id}_func_{func_name}", 
                func_text, 
                base_token_size=self.max_tokens
            )
            
            chunks = []
            for i, sc in enumerate(sub_chunks):
                chunk_text = sc.text
                if self.include_imports_in_all and import_block and i == 0:
                    chunk_text = import_block + "\n\n" + chunk_text
                if context_prefix and i == 0:
                    chunk_text = context_prefix + chunk_text
                
                chunks.append(Chunk(
                    id=f"{doc_id}#py#{idx + i}",
                    doc_id=doc_id,
                    text=chunk_text,
                    meta={
                        "chunk_index": idx + i,
                        "strategy": "python_ast",
                        "type": "function_part",
                        "name": func_name,
                        "class_name": class_name,
                        "part": i + 1,
                        "token_count": count_tokens(chunk_text)
                    }
                ))
            return chunks
        
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name
        
        return [Chunk(
            id=f"{doc_id}#py#{idx}",
            doc_id=doc_id,
            text=full_text,
            meta={
                "chunk_index": idx,
                "strategy": "python_ast",
                "type": "method" if class_name else "function",
                "name": func_name,
                "qualified_name": qualified_name,
                "class_name": class_name,
                "docstring": docstring[:200] if docstring else "",
                "token_count": count_tokens(full_text)
            }
        )]
