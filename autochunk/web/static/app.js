
const API_BASE = "";

const elements = {
    viewLanding: document.getElementById('view-landing'),
    viewWorkspace: document.getElementById('view-workspace'),
    backToConfig: document.getElementById('backToConfig'),
    docsPath: document.getElementById('docsPath'),
    fileInfo: document.getElementById('fileInfo'),
    runBtn: document.getElementById('runBtn'),
    navToWorkspace: document.getElementById('navToWorkspace'),
    progressState: document.getElementById('progressState'),
    resultState: document.getElementById('resultState'),
    winnerName: document.getElementById('winnerName'),
    winnerParams: document.getElementById('winnerParams'),
    metricBoard: document.getElementById('metricBoard'),
    candidateList: document.getElementById('candidateList'),
    logTerminal: document.getElementById('logTerminal'),
    logToggle: document.getElementById('logToggle'),
    logChevron: document.getElementById('logChevron'),
    logSection: document.getElementById('logSection'),
    thinkingMsg: document.getElementById('thinkingMsg'),
    fidelityModal: document.getElementById('fidelityModal'),
    fidelityModal: document.getElementById('fidelityModal'),
    pane1Title: document.getElementById('pane1Title'),
    pane1Content: document.getElementById('pane1Content'),
    pane2Title: document.getElementById('pane2Title'),
    pane2Content: document.getElementById('pane2Content')
};

// --- UI Logic ---

function showView(viewId) {
    elements.viewLanding.classList.toggle('hidden', viewId !== 'landing');
    elements.viewWorkspace.classList.toggle('hidden', viewId !== 'workspace');

    // Smooth scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

elements.backToConfig.addEventListener('click', () => {
    showView('landing');
});

elements.navToWorkspace.addEventListener('click', () => {
    showView('workspace');
});

elements.logToggle.addEventListener('click', () => {
    elements.logTerminal.classList.toggle('minimized');
    const isMinimized = elements.logTerminal.classList.contains('minimized');
    elements.logChevron.style.transform = isMinimized ? 'rotate(0deg)' : 'rotate(180deg)';
});

document.getElementById('fileInfoToggle').addEventListener('click', () => {
    const container = document.getElementById('fileInfoContainer');
    const chevron = document.getElementById('fileInfoChevron');
    container.classList.toggle('hidden');
    const isHidden = container.classList.contains('hidden');
    chevron.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';
});

document.getElementById('embeddingProvider').addEventListener('change', (e) => {
    const group = document.getElementById('modelConfigGroup');
    const modelInput = document.getElementById('embeddingModel');

    group.style.display = e.target.value === 'hashing' ? 'none' : 'block';

    // Set smart defaults and handle API Key visibility
    const apiKeyLabel = document.getElementById('embeddingApiKeyLabel');
    const apiKeyInput = document.getElementById('embeddingApiKey');

    if (e.target.value === 'openai') {
        modelInput.value = 'text-embedding-3-small';
        apiKeyLabel.style.display = 'block';
        apiKeyInput.style.display = 'block';
    } else if (e.target.value === 'ollama') {
        modelInput.value = 'llama3';
        apiKeyLabel.style.display = 'none';
        apiKeyInput.style.display = 'none';
    } else if (e.target.value === 'local') {
        modelInput.value = 'BAAI/bge-small-en-v1.5';
        apiKeyLabel.style.display = 'none';
        apiKeyInput.style.display = 'none';
    } else {
        apiKeyLabel.style.display = 'none';
        apiKeyInput.style.display = 'none';
    }
});

// Optimization Objective logic handled by the Goal mapping

// RAGAS Toggle Handler
document.getElementById('ragasEnabled').addEventListener('change', (e) => {
    const configGroup = document.getElementById('ragasConfigGroup');
    if (e.target.checked) {
        configGroup.classList.remove('hidden');
    } else {
        configGroup.classList.add('hidden');
    }
});

// RAGAS Provider Key visibility
document.getElementById('ragasLlmProvider').addEventListener('change', (e) => {
    const keyLabel = document.getElementById('ragasApiKeyLabel');
    const keyInput = document.getElementById('ragasApiKey');
    if (e.target.value === 'openai') {
        keyLabel.style.display = 'block';
        keyInput.style.display = 'block';
    } else {
        keyLabel.style.display = 'none';
        keyInput.style.display = 'none';
    }
});

// --- API Interactions ---

async function fetchDocs() {
    try {
        const path = elements.docsPath.value;
        const res = await fetch(`/api/docs/list?path=${encodeURIComponent(path)}`);
        const data = await res.json();

        elements.fileInfo.innerHTML = data.files.map(f => `
            <div class="file-item">
                <i data-lucide="file-text" style="width: 14px; height: 14px;"></i>
                ${f}
            </div>
        `).join('');
        lucide.createIcons();
    } catch (e) {
        console.error("Failed to fetch docs", e);
    }
}

async function startOptimization() {
    // Get selected candidates
    const selected = Array.from(document.querySelectorAll('.chunker-check:checked'))
        .map(el => el.value);

    // Capture Sweep Params
    const sweepSizes = Array.from(document.querySelectorAll('input[name="sweep_size"]:checked')).map(e => parseInt(e.value));
    const sweepOverlaps = Array.from(document.querySelectorAll('input[name="sweep_overlap"]:checked')).map(e => parseFloat(e.value));

    const goal = document.getElementById('optimizationGoal').value;
    const goalMapping = {
        'balanced_ndcg': { objective: 'balanced', metrics: ['ndcg@10'] },
        'balanced_recall': { objective: 'balanced', metrics: ['recall@50'] },
        'latency_mrr': { objective: 'latency', metrics: ['mrr@10'] },
        'quality_max': { objective: 'quality', metrics: ['ndcg@10'] },
        'cost_efficiency': { objective: 'cost', metrics: ['ndcg@10'] }
    };
    const selectedGoal = goalMapping[goal] || goalMapping['balanced_ndcg'];

    const req = {
        documents_path: elements.docsPath.value,
        mode: document.getElementById('optimizeMode').value,
        objective: selectedGoal.objective,
        metrics: selectedGoal.metrics,
        proxy_enabled: document.getElementById('proxyEnabled').checked,
        proxy_percent: 10,
        embedding_provider: document.getElementById('embeddingProvider').value,
        embedding_model_or_path: document.getElementById('embeddingModel').value,
        embedding_api_key: document.getElementById('embeddingApiKey').value || null,
        selected_candidates: selected,
        sweep_params: {
            chunk_sizes: sweepSizes.length > 0 ? sweepSizes : [512],
            overlap_ratios: sweepOverlaps.length > 0 ? sweepOverlaps : [0.1]
        },
        local_models_path: document.getElementById('localModelsPath').value,
        telemetry_enabled: document.getElementById('telemetryEnabled').checked,
        // RAGAS Configuration
        analyze_ragas: document.getElementById('ragasEnabled').checked,
        ragas_llm_provider: document.getElementById('ragasLlmProvider').value,
        ragas_llm_model: document.getElementById('ragasLlmModel').value || null,
        ragas_api_key: document.getElementById('ragasApiKey').value || null
    };

    // UI Switch to Workspace
    showView('workspace');
    elements.resultState.style.display = 'none';
    elements.progressState.style.display = 'block';

    // Ensure logs are expanded and follow the progress naturally
    elements.logTerminal.classList.remove('minimized');
    elements.logTerminal.innerHTML = '<div class="log-entry"><span class="log-time">[System]</span><span class="log-msg">Orchestrating optimization pipeline...</span></div>';
    resetStepper();

    try {
        const res = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req)
        });
        const { job_id } = await res.json();
        pollJob(job_id);
    } catch (e) {
        alert("Server error. Check backend logs.");
        showView('landing');
    }
}

function resetStepper() {
    [1, 2, 3].forEach(i => {
        const step = document.getElementById(`step${i}`);
        step.classList.remove('active', 'done');
    });
}

function updateStepper(activeStep) {
    if (window.lastActiveStep === activeStep) return;
    window.lastActiveStep = activeStep;

    const steps = ["Setup", "QA Generation", "Parallel Evaluation"];
    for (let i = 1; i <= 3; i++) {
        const step = document.getElementById(`step${i}`);
        if (i < activeStep) {
            step.classList.remove('active');
            step.classList.add('done');
        } else if (i === activeStep) {
            step.classList.add('active');
            step.classList.remove('done');

            // Only update msg if not already specialized (like with a counter)
            const currentMsg = elements.thinkingMsg.innerText;
            if (!currentMsg.includes('(') || activeStep !== 3) {
                elements.thinkingMsg.innerText = `AutoChunks: ${steps[i - 1]}...`;
            }

            addSystemLog(`Moving to Stage ${i}: ${steps[i - 1]}`);
        } else {
            step.classList.remove('active', 'done');
        }
    }
}

function addSystemLog(msg) {
    const time = new Date().toLocaleTimeString([], { hour12: false });
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.innerHTML = `<span class="log-time">[${time}]</span><span class="log-msg">${msg}</span>`;
    // Only add if last log message is different to avoid spam
    const lastLog = elements.logTerminal.lastElementChild;
    if (!lastLog || !lastLog.innerText.includes(msg)) {
        elements.logTerminal.appendChild(div);
        elements.logTerminal.scrollTop = elements.logTerminal.scrollHeight;
    }
}

async function pollJob(jobId) {
    const interval = setInterval(async () => {
        const res = await fetch(`/api/job/${jobId}`);

        if (res.status === 404) {
            clearInterval(interval);
            return;
        }

        const data = await res.json();

        // Update Logs
        updateLogs(data.logs);

        // Update Stepper
        updateStepper(data.step);

        if (data.status === 'running') {
            if (data.partial_results && data.partial_results.length > 0) {
                const partials = data.partial_results;
                const leader = [...partials].sort((a, b) => (b.score || 0) - (a.score || 0))[0];

                const total = data.total_candidates || 0;
                const current = partials.length;
                if (total > 0 && data.step === 3) {
                    elements.thinkingMsg.innerText = `AutoChunks: Parallel Evaluation... (${current} of ${total})`;
                }

                currentGlobalResult = {
                    candidates: partials,
                    best_strategy: leader.name,
                    best_params: leader.params,
                    metrics: leader.metrics || {}
                };
                renderCandidateList('objective_score');
            }
        }
        if (data.status === 'completed') {
            clearInterval(interval);
            document.getElementById('step3').classList.add('done');
            showResults(data.result);
        } else if (data.status === 'failed' || data.status === 'cancelled') {
            clearInterval(interval);
            alert(data.status === 'cancelled' ? "Cancelled" : "Failed");
            showView('landing');
        }
    }, 1500);
}

function updateLogs(logs) {
    if (!logs) return;
    const existingCount = elements.logTerminal.querySelectorAll('.log-entry').length;
    if (logs.length > existingCount) {
        const newLogs = logs.slice(existingCount);
        newLogs.forEach(log => {
            if (log.msg.includes("Moving to Stage") && elements.logTerminal.innerText.includes(log.msg)) return;

            const div = document.createElement('div');
            div.className = 'log-entry';

            let levelClass = '';
            if (log.msg.includes('[INFO]')) levelClass = 'log-info';
            if (log.msg.includes('[SUCCESS]')) levelClass = 'log-success';
            if (log.msg.includes('[ERROR]') || log.msg.includes('[CRITICAL]')) levelClass = 'log-error';
            if (log.msg.includes('[WARNING]')) levelClass = 'log-warning';
            if (log.msg.includes('[DEBUG]')) levelClass = 'log-debug';

            div.innerHTML = `
                <span class="log-time">[${log.time}]</span>
                <span class="log-msg ${levelClass}">${log.msg}</span>
            `;
            elements.logTerminal.appendChild(div);
        });
        setTimeout(() => {
            elements.logTerminal.scrollTop = elements.logTerminal.scrollHeight;
        }, 50);
    }
}

let currentGlobalResult = null;
let currentDeployTab = 'langchain';

function showResults(result) {
    currentGlobalResult = result;
    setTimeout(() => {
        elements.progressState.style.display = 'none';
        elements.resultState.style.display = 'block';

        elements.logTerminal.classList.add('minimized');
        elements.logChevron.style.transform = 'rotate(0deg)';

        elements.winnerName.innerText = result.best_strategy.replace('_', ' ').toUpperCase();
        elements.winnerParams.innerText = JSON.stringify(result.best_params).replace(/[{}"]/g, '');

        const METRIC_TIPS = {
            'ndcg@k': 'Ranking Quality',
            'mrr@k': 'Top Rank Quality',
            'recall@k': 'Retrieval Breadth',
            'coverage': 'Proportion answerable',
            'count': 'Total chunks',
            'objective_score': 'Weighted composite score',
            'avg_quality_score': 'Quality Scorer evaluates chunks across 5 dimensions: Coherence (semantic consistency), Completeness (self-contained context), Density (info-rich vs fluff), Boundary (grammatical integrity), and Size (optimal token range).',
            'context_precision': 'RAGAS: Measures if ground truth is ranked highly in retrieved context',
            'context_recall': 'RAGAS: Measures if all relevant info is present in context'
        };

        // Standard metrics
        const standardMetrics = ['ndcg@k', 'mrr@k', 'recall@k', 'coverage', 'count', 'objective_score', 'avg_quality_score'];
        // RAGAS metrics (show with special styling)
        const ragasMetrics = ['context_precision', 'context_recall'];

        let metricsHtml = Object.entries(result.metrics).map(([k, v]) => {
            if (standardMetrics.includes(k)) {
                let label = k.toUpperCase().replace('@K', '');
                let value = typeof v === 'number' ? (v > 1 ? v.toFixed(0) : (v * 100).toFixed(1) + '%') : v;
                let tip = METRIC_TIPS[k] || '';
                return `
                    <div class="metric-card">
                        <span class="metric-value">${value}</span>
                        <span class="metric-label" ${tip ? `data-tooltip="${tip}"` : ''}>${label}</span>
                    </div>
                `;
            }
            return '';
        }).join('');

        // Add RAGAS metrics with special amber styling
        const hasRagasMetrics = ragasMetrics.some(m => result.metrics[m] !== undefined);
        if (hasRagasMetrics) {
            metricsHtml += '<div style="grid-column: 1 / -1; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(245, 158, 11, 0.2);"></div>';
            ragasMetrics.forEach(k => {
                const v = result.metrics[k];
                if (v !== undefined) {
                    let label = k.replace('context_', '').toUpperCase();
                    let value = typeof v === 'number' ? (v * 100).toFixed(1) + '%' : v;
                    let tip = METRIC_TIPS[k] || '';
                    metricsHtml += `
                        <div class="metric-card" style="border-color: rgba(245, 158, 11, 0.3);">
                            <span class="metric-value" style="color: #f59e0b;">${value}</span>
                            <span class="metric-label" style="color: #fbbf24;" ${tip ? `data-tooltip="${tip}"` : ''}>
                                <i data-lucide="sparkles" style="width: 10px; vertical-align: middle; margin-right: 2px;"></i>
                                ${label}
                            </span>
                        </div>
                    `;
                }
            });
        }

        elements.metricBoard.innerHTML = metricsHtml;

        renderCandidateList('objective_score');

        document.querySelectorAll('.sort-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                renderCandidateList(e.target.dataset.sort);
            });
        });

        updateDeployCode();
        lucide.createIcons();
    }, 500);
}

function renderCandidateList(sortKey) {
    if (!currentGlobalResult) return;

    const sorted = [...currentGlobalResult.candidates].sort((a, b) => {
        const aVal = a.metrics[sortKey] || 0;
        const bVal = b.metrics[sortKey] || 0;
        return sortKey === 'count' ? aVal - bVal : bVal - aVal;
    });

    elements.candidateList.innerHTML = sorted.map((cand) => {
        const originalIdx = currentGlobalResult.candidates.indexOf(cand);
        const score = ((cand.metrics.objective_score || 0) * 100).toFixed(1);
        const ndcg = ((cand.metrics['ndcg@k'] || 0) * 100).toFixed(1);
        const mrr = ((cand.metrics['mrr@k'] || 0) * 100).toFixed(1);

        // RAGAS metrics (if available)
        const hasRagas = cand.metrics.context_precision !== undefined;
        const ctxPrec = hasRagas ? ((cand.metrics.context_precision || 0) * 100).toFixed(0) : null;

        let badges = cand.metrics.post_processed ? `<span class="tag-badge" style="background: rgba(52, 211, 153, 0.1); color: #34d399; font-size: 0.6rem; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">Optimized</span>` : '';
        const bridgeBadge = cand.name.startsWith('langchain_') ? `<span class="tag-badge" style="background: rgba(96, 165, 250, 0.1); color: #60a5fa; font-size: 0.6rem; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">Bridge</span>` : '';
        const ragasBadge = hasRagas ? `<span class="tag-badge" style="background: rgba(245, 158, 11, 0.1); color: #f59e0b; font-size: 0.6rem; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">✦ RAGAS</span>` : '';

        return `
            <div class="candidate-item">
                <div style="flex: 2;">
                    <div style="font-weight: 600; display: flex; align-items: center; flex-wrap: wrap;">
                        ${cand.name.replace('_', ' ').toUpperCase()}
                        ${bridgeBadge}
                        ${badges}
                        ${ragasBadge}
                    </div>
                    <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.25rem;">${JSON.stringify(cand.params).replace(/[{}\"]/g, '').substring(0, 50)}...</div>
                </div>
                <div class="cand-metric">
                    <span class="cand-metric-val" style="color: var(--accent-teal);">${score}%</span>
                    <span class="cand-metric-label">Score</span>
                </div>
                <div class="cand-metric">
                    <span class="cand-metric-val">${ndcg}%</span>
                    <span class="cand-metric-label">nDCG</span>
                </div>
                <div class="cand-metric">
                    <span class="cand-metric-val">${((cand.metrics.avg_quality_score || 0) * 100).toFixed(0)}</span>
                    <span class="cand-metric-label">Quality</span>
                </div>
                ${hasRagas ? `
                <div class="cand-metric" style="border-left: 1px solid rgba(245, 158, 11, 0.2); padding-left: 0.75rem;">
                    <span class="cand-metric-val" style="color: #f59e0b;">${ctxPrec}%</span>
                    <span class="cand-metric-label" style="color: #fbbf24;">Ctx Prec</span>
                </div>
                ` : `
                <div class="cand-metric">
                    <span class="cand-metric-val">${mrr}%</span>
                    <span class="cand-metric-label">MRR</span>
                </div>
                `}
                <div class="compare-badge" style="cursor: pointer; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px; font-size: 0.7rem;" onclick="openFidelity('${cand.name}', ${originalIdx})">
                    Compare
                </div>
            </div>
        `;
    }).join('');
}

function switchDeployTab(tab) {
    currentDeployTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.innerText.toLowerCase().includes(tab));
    });
    updateDeployCode();
}

function updateDeployCode() {
    if (!currentGlobalResult) return;
    const codeElem = document.querySelector('#deployCode code');
    if (currentDeployTab === 'langchain') {
        codeElem.innerText = `from autochunk import AutoChunkLangChainAdapter\nsplitter = AutoChunkLangChainAdapter(plan="best_plan.yaml")\nchunks = splitter.split_documents(documents)`;
    } else if (currentDeployTab === 'llamaindex') {
        codeElem.innerText = `from autochunk import AutoChunkLlamaIndexAdapter\nnode_parser = AutoChunkLlamaIndexAdapter(plan="best_plan.yaml")\nnodes = node_parser.get_nodes_from_documents(documents)`;
    } else {
        codeElem.innerText = `from autochunk import AutoChunkHaystackAdapter\nchunker = AutoChunkHaystackAdapter(plan="best_plan.yaml")`;
    }
}

function copyDeployCode() {
    const code = document.querySelector('#deployCode code').innerText;
    navigator.clipboard.writeText(code);
    const btn = document.querySelector('.copy-btn');
    const originalHtml = btn.innerHTML;
    btn.innerHTML = '<span>Check!</span>';
    setTimeout(() => btn.innerHTML = originalHtml, 2000);
}

function openFidelity(name, index) {
    if (!currentGlobalResult) return;
    const candidateA = currentGlobalResult.candidates.find(c =>
        c.name === currentGlobalResult.best_strategy &&
        JSON.stringify(c.params) === JSON.stringify(currentGlobalResult.best_params)
    );
    const candidateB = currentGlobalResult.candidates[index];

    const fmt = (c) => {
        const size = c.params.size || c.params.base_token_size || '?';
        let overlapPercent = 0;
        if (c.params.overlap_ratio !== undefined) {
            overlapPercent = Math.round(c.params.overlap_ratio * 100);
        } else if (c.params.overlap !== undefined && typeof size === 'number') {
            overlapPercent = Math.round((c.params.overlap / size) * 100);
        }
        return `${c.name.toUpperCase()} (${size}|${overlapPercent}%)`;
    };

    elements.pane1Title.innerText = `Leader: ${fmt(candidateA)}`;
    elements.pane2Title.innerText = `Candidate: ${fmt(candidateB)}`;

    renderFidelityPane(elements.pane1Content, candidateA.chunk_samples);
    renderFidelityPane(elements.pane2Content, candidateB.chunk_samples);
    elements.fidelityModal.classList.remove('hidden');
}

function renderFidelityPane(container, samples) {
    if (!samples || samples.length === 0) {
        container.innerHTML = '<div style="padding: 2rem; text-align: center; opacity: 0.5;">No samples available for this candidate.</div>';
        return;
    }
    container.innerHTML = samples.map((s, i) => `
        <div class="visual-chunk collapsed" data-index="${i + 1}">
            <div class="token-badge">${s.tokens || 0} tokens</div>
            <div class="chunk-text">${s.text}</div>
            ${s.is_junction ? '<div style="position: absolute; bottom: 8px; right: 12px; font-size: 0.6rem; color: #fbbf24; opacity: 0.8;">✦ Junction</div>' : ''}
        </div>
    `).join('');

    // Attach click listeners for expand/collapse
    container.querySelectorAll('.visual-chunk').forEach(el => {
        el.addEventListener('click', () => {
            el.classList.toggle('collapsed');
        });
    });
}

function closeFidelity() {
    elements.fidelityModal.classList.add('hidden');
}

function updateNetworkAudit() {
    const localPath = document.getElementById('localModelsPath').value.trim();
    const provider = document.getElementById('embeddingProvider').value;
    const modelId = document.getElementById('embeddingModel').value;
    const auditPanel = document.getElementById('networkAudit');
    const auditStatus = document.getElementById('auditStatus');
    const auditDetails = document.getElementById('auditDetails');
    const networkLinks = document.getElementById('networkLinks');

    if (provider === 'local') {
        auditPanel.classList.remove('hidden');
        if (localPath) {
            auditPanel.classList.add('audit-safe');
            auditStatus.innerHTML = `Using local cache at <code>${localPath}</code>.`;
            auditDetails.classList.add('hidden');
        } else {
            auditPanel.classList.remove('audit-safe');
            auditStatus.innerHTML = `Notice: Models will be downloaded.`;
            auditDetails.classList.remove('hidden');
            networkLinks.innerHTML = `<li>${modelId}: <a href="https://huggingface.co/${modelId}" target="_blank">HF Repo</a></li>`;
        }
    } else {
        auditPanel.classList.add('hidden');
    }
    lucide.createIcons();
}

// --- Event Listeners ---

elements.runBtn.addEventListener('click', startOptimization);
elements.docsPath.addEventListener('change', fetchDocs);
document.getElementById('localModelsPath').addEventListener('input', updateNetworkAudit);

// Sync Embedding Provider to RAGAS Info
document.getElementById('embeddingProvider').addEventListener('change', (e) => {
    updateNetworkAudit();
    const val = e.target.value;
    const label = val === 'hashing' ? 'Hashing (Mock)' : 'Local (Start BGE)';
    document.getElementById('ragasRetrievalStatus').innerHTML = `Retrieval via: <strong>${label}</strong>`;
});

document.getElementById('embeddingModel').addEventListener('input', updateNetworkAudit);

// Initial Load
fetchDocs();
updateNetworkAudit();
setInterval(fetchDocs, 10000);
lucide.createIcons();
