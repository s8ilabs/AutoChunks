# Dashboard Walkthrough

The AutoChunks Dashboard is your command center for understanding precisely why a chunking strategy succeeds or fails.

## 1. Setup and Configuration

Upon launching `autochunks serve`, you are greeted by the Control Center.

*   **Documents Path**: Point this to your document repository (supports PDF, MD, HTML, TXT).
*   **Optimization Mode**:
    *   **Light (Fast Sample Sweep)**: Optimized for development. It caps document sampling to find the winning strategy quickly.
    *   **Full (Exhaustive Corpus Audit)**: Production-grade tuning. Uses a wider document sample for high-confidence benchmarking.
*   **Embedding Provider**:
    *   **Local (Production Simulation)**: Uses real vector embeddings via the `sentence-transformers` library. This allows you to match your production environment (e.g., BGE-Large). 
    *   **OpenAI**: Uses the OpenAI Embeddings API. Requires an API key which is entered securely in the dashboard.
    *   **Ollama**: Connects to a locally running Ollama instance.
    *   **Hashing (Architectural/CI)**: A high-performance mock provider used for structural testing and CI/CD integration without model overhead.

## 2. The Sweep

When you click Start Optimization, the engine enters the tournament state.

1.  **Ingestion**: Files are read and indexed via SHA-256 fingerprinting.
2.  **QA Generation**: Synthetic ground-truth questions are created for evaluation.
3.  **Parallel Evaluation**: The system fans out to test all selected strategies simultaneously across the vectorized search pipeline.

## 3. The Leaderboard

Once complete, the dashboard displays the Winning Strategy and a ranked list of all candidates.

*   **Objective Score**: The weighted composite score used to determine the winner based on your Optimization Goal.
*   **nDCG / MRR**: Detailed retrieval benchmarks and ranking positions.
*   **Chunk Count**: Analyzes the fragmentation overhead; some strategies may score highly but produce an excessive number of chunks, increasing storage costs.

## 4. Fidelity Inspector

This tools is designed for qualitative debugging. Click Compare on any candidate to open the Inspector.

*   **Side-by-Side View**: See exactly where the Winner cut the text vs. any other candidate strategy.
*   **Visual Diff**: Identifies structural errors, such as whether a sentence was bifurcated or if a semantic splitter triggered correctly.
*   **Sync Scroll**: Synchronized scrolling across panes to track long-form documents.

## 5. Deploy as Code

The Code Export tab provides framework-specific snippets for LangChain, Haystack, or LlamaIndex. Copy these configurations to instantly apply the optimized strategy to your production data pipeline.
