# Enterprise Security and Offline Mode

AutoChunks is designed for high-security environments, including Air-Gapped networks and strict corporate firewalls.

## 1. Local-First Processing Architecture
By default, AutoChunks operates as a Zero-Telemetry system. All data processing—including text extraction, layout analysis, and semantic evaluation—occurs within your local memory and compute boundaries.

*   **Proprietary Protection**: Document text never traverses an external API unless you explicitly configure an OpenAI or external provider.
*   **Encrypted Metadata**: Internal metadata (checksums, metrics) is stored in the `.ac_cache` directory.
*   **Process Isolation**: Evaluation occurs in a sandboxed ThreadPool, ensuring that failures in specific chunking libraries do not compromise the main optimization state.

## 2. Secure Credential Handling
For teams using cloud providers (OpenAI, Gemini), AutoChunks implements multi-layer protection for API keys.

*   **Secret Tunneling**: Backend models use Pydantic `SecretStr` to wrap all API keys. These objects allow the key to be used for requests while preventing them from appearing in logs, error messages, or string representations.
*   **UI Masking**: All credential fields in the dashboard use password-masking (type="password") to prevent shoulder-surfing and accidental exposure during screen sharing.
*   **Zero-Persistence Plan**: The best_plan.yaml artifacts contain your optimized strategy and model IDs but never persist your API keys to disk.

## 3. Cryptographic Fingerprinting (SHA-256)
We use SHA-256 content hashing to ensure data integrity and skip redundant compute.
*   **Binary Fingerprinting**: Documents are hashed at the raw byte level before extraction.
*   **Semantic Checksumming**: Extracted text blocks are re-hashed to detect minute changes in layout analysis algorithms, ensuring that evaluation metrics are always tied to the exact current version of the content.

## 4. Trusted Organization Whitelist
To mitigate supply-chain risks in open-source model loading, AutoChunks implements a strict Organization-level whitelist for Hugging Face downloads. By default, the system only permits downloads from official and verified organizations:
*   `sentence-transformers`
*   `BAAI` (Beijing Academy of AI)
*   `ds4sd` (IBM Deep Search)
*   `RapidAI`
*   `RapidOCR`

Any attempt to load a model from an unverified source will raise a SecurityError.

## 5. Air-Gapped and Offline Usage

For servers with no internet access, AutoChunks supports full offline operation via the Local Model Path feature.

### Step 1: Pre-Download Models
Download the required embedding or judge model folders on an internet-connected machine.

### Step 2: Protocol Transfer
Transfer the model folders to your secure environment (e.g., `/opt/secure/models/bge-small-en-v1.5`).

### Step 3: Configure Paths
Point the AutoChunker to the absolute local path. The engine will detect the directory and skip all external network requests.

**CLI Entry:**
```bash
autochunks optimize --docs ./docs --embedding-model /opt/secure/models/bge-small-en-v1.5
```

**Dashboard Entry:**
In the Model ID field, enter the absolute file path to the model directory. Ensure the Local Models Cache Path is set correctly in your persistent configuration if using a shared environment.
