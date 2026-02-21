document.addEventListener('DOMContentLoaded', () => {
    const docSelect = document.getElementById('document-select');
    const useSubagentsToggle = document.getElementById('use-subagents');
    const subagentPanel = document.getElementById('subagent-panel');
    const submitBtn = document.getElementById('submit-btn');
    const queryInput = document.getElementById('query-input');
    const chatHistory = document.getElementById('chat-history');

    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = document.getElementById('loading-spinner');

    useSubagentsToggle.addEventListener('change', (e) => {
        subagentPanel.classList.toggle('hidden', !e.target.checked);
    });

    queryInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = `${this.scrollHeight}px`;
    });

    async function fetchDocuments() {
        try {
            const res = await fetch('/api/documents');
            const data = await res.json();

            docSelect.innerHTML = '';
            if (data.documents && data.documents.length > 0) {
                data.documents.forEach((doc) => {
                    const opt = document.createElement('option');
                    opt.value = doc;
                    opt.textContent = doc;
                    docSelect.appendChild(opt);
                });
                return;
            }

            const opt = document.createElement('option');
            opt.disabled = true;
            opt.textContent = 'No docs in embed-docs/';
            docSelect.appendChild(opt);
        } catch (error) {
            console.error('Failed to fetch documents:', error);
        }
    }

    function scrollChatToBottom() {
        chatHistory.parentElement.scrollTop = chatHistory.parentElement.scrollHeight;
    }

    function appendMessage(role, content, metrics = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        let htmlContent = '';
        if (role === 'bot') {
            htmlContent = marked.parse(content);
        } else {
            htmlContent = document.createElement('div');
            htmlContent.innerText = role === 'user' ? content : `System: ${content}`;
            htmlContent = htmlContent.innerHTML;
        }

        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'markdown-body';
        bodyDiv.innerHTML = htmlContent;
        msgDiv.appendChild(bodyDiv);

        if (metrics) {
            const m = metrics;
            const metaDiv = document.createElement('div');
            metaDiv.className = 'metadata-block';
            let metaHtml = `<strong>Metrics (${m.mode || 'unknown'})</strong><ul>`;

            if (m.execution_time != null) metaHtml += `<li>Time: ${m.execution_time.toFixed(2)}s</li>`;
            if (m.total_tokens) metaHtml += `<li>Tokens: ${m.total_tokens} (I: ${m.total_input_tokens}/O: ${m.total_output_tokens})</li>`;
            if (m.iterations) metaHtml += `<li>Iterations: ${m.iterations}</li>`;
            if (m.subagent_calls) metaHtml += `<li>Subagent Calls: ${m.subagent_calls}</li>`;

            metaHtml += '</ul>';
            metaDiv.innerHTML = metaHtml;
            msgDiv.appendChild(metaDiv);
        }

        chatHistory.appendChild(msgDiv);
        scrollChatToBottom();
    }

    function appendLiveBox(container, html, options = {}) {
        const box = document.createElement('div');
        box.className = 'iteration-box';

        if (options.variant) {
            box.classList.add(`iteration-${options.variant}`);
        }

        box.innerHTML = html;
        container.appendChild(box);
        container.scrollTop = container.scrollHeight;
        return box;
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    function buildLiveContainer() {
        const liveContainer = document.createElement('div');
        liveContainer.className = 'message system live-message';

        const status = document.createElement('div');
        status.className = 'loading-indicator';
        status.innerHTML = 'Initializing RLM Engine <div class="dot"></div><div class="dot"></div><div class="dot"></div>';

        const streamGrid = document.createElement('div');
        streamGrid.className = 'stream-grid';

        function createStreamPanel(title) {
            const panel = document.createElement('section');
            panel.className = 'stream-panel';

            const header = document.createElement('div');
            header.className = 'stream-panel-header';

            const titleEl = document.createElement('h4');
            titleEl.className = 'stream-title';
            titleEl.textContent = title;

            const expandBtn = document.createElement('button');
            expandBtn.type = 'button';
            expandBtn.className = 'stream-expand-btn';
            expandBtn.setAttribute('aria-expanded', 'false');
            expandBtn.textContent = 'Expand';
            expandBtn.addEventListener('click', () => {
                const expanded = panel.classList.toggle('is-expanded');
                expandBtn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
                expandBtn.textContent = expanded ? 'Collapse' : 'Expand';
            });

            const body = document.createElement('div');
            body.className = 'stream-body';

            header.appendChild(titleEl);
            header.appendChild(expandBtn);
            panel.appendChild(header);
            panel.appendChild(body);

            return { panel, body };
        }

        const agentPanel = createStreamPanel('Agent Output');
        const subagentPanel = createStreamPanel('Subagent Output');

        const agentBody = agentPanel.body;
        const subagentBody = subagentPanel.body;

        streamGrid.appendChild(agentPanel.panel);
        streamGrid.appendChild(subagentPanel.panel);

        liveContainer.appendChild(status);
        liveContainer.appendChild(streamGrid);

        return {
            liveContainer,
            status,
            agentBody,
            subagentBody,
        };
    }

    fetchDocuments();

    submitBtn.addEventListener('click', async () => {
        const selectedDocs = Array.from(docSelect.selectedOptions).map((opt) => opt.value);
        const query = queryInput.value.trim();

        if (selectedDocs.length === 0) {
            alert('Please select at least one document.');
            return;
        }

        if (!query) {
            return;
        }

        const modelName = document.getElementById('model-input').value.trim() || null;
        const useSubagents = useSubagentsToggle.checked;
        const subBackend = document.getElementById('subagent-backend').value.trim() || null;
        const subModel = document.getElementById('subagent-model').value.trim() || null;
        const sysPrompt = document.getElementById('system-prompt').value.trim() || null;

        const payload = {
            documents: selectedDocs,
            query,
            model_name: modelName,
            use_subagents: useSubagents,
            system_prompt: sysPrompt,
            subagent_backend: subBackend,
            subagent_model: subModel,
        };

        appendMessage('user', query);
        queryInput.value = '';
        queryInput.style.height = 'auto';
        submitBtn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');

        const {
            liveContainer,
            status,
            agentBody,
            subagentBody,
        } = buildLiveContainer();
        chatHistory.appendChild(liveContainer);
        scrollChatToBottom();

        try {
            const res = await fetch('/api/query/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.body) {
                throw new Error('ReadableStream not yet supported in this browser.');
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let sawVerboseTraceEvents = false;
            let latestVerboseFinalAnswer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n\n');
                buffer = parts.pop() || '';

                for (const part of parts) {
                    if (!part.startsWith('data: ')) {
                        continue;
                    }

                    try {
                        const data = JSON.parse(part.substring(6));

                        if (data.type === 'error') {
                            liveContainer.remove();
                            appendMessage('error', data.detail || 'Server error occurred.');
                            break;
                        }

                        if (data.type === 'load_doc') {
                            appendLiveBox(agentBody, `<strong>Loading Document:</strong> ${escapeHtml(data.doc)} ...`);
                        } else if (data.type === 'load_complete') {
                            status.classList.add('hidden');
                            appendLiveBox(
                                agentBody,
                                `<em>Loaded ${escapeHtml(data.num_docs)} document(s) into context. Executing query...</em>`,
                            );
                        } else if (data.type === 'verbose') {
                            if (data.action === 'iteration_start') {
                                sawVerboseTraceEvents = true;
                                appendLiveBox(
                                    agentBody,
                                    `<div class="iteration-header">
                                        <span>Iteration #${data.iteration}</span>
                                        <span class="spinner mini-spinner"></span>
                                    </div>
                                    <div class="verbose-content" style="opacity:0.8; white-space:pre-wrap;">Thinking...</div>`,
                                );
                            } else if (data.action === 'completion') {
                                sawVerboseTraceEvents = true;
                                const boxes = agentBody.querySelectorAll('.iteration-box');
                                if (boxes.length > 0) {
                                    const lastBox = boxes[boxes.length - 1];
                                    const contentDiv = lastBox.querySelector('.verbose-content');
                                    const activeSpinner = lastBox.querySelector('.mini-spinner');
                                    if (contentDiv) {
                                        contentDiv.innerText = String(data.response);
                                    }
                                    if (activeSpinner) {
                                        activeSpinner.style.display = 'none';
                                    }
                                }
                            } else if (data.action === 'code_execution') {
                                sawVerboseTraceEvents = true;
                                let html = `<strong>Code Execution:</strong><pre><code>${escapeHtml(data.code)}</code></pre>`;
                                if (data.stdout && data.stdout.trim()) {
                                    html += `<strong>Output:</strong><pre><code>${escapeHtml(data.stdout)}</code></pre>`;
                                }
                                if (data.stderr && data.stderr.trim()) {
                                    html += `<strong>Error:</strong><pre><code class="stream-error">${escapeHtml(data.stderr)}</code></pre>`;
                                }
                                appendLiveBox(agentBody, html, { variant: 'agent' });
                            } else if (data.action === 'subcall') {
                                sawVerboseTraceEvents = true;
                                appendLiveBox(
                                    subagentBody,
                                    `<strong>Subagent Call (${escapeHtml(data.model)}):</strong>
                                    <div class="stream-detail">Prompt: ${escapeHtml(data.prompt)}</div>
                                    <div class="stream-detail">Response: ${escapeHtml(data.response)}</div>`,
                                    { variant: 'subagent' },
                                );
                            } else if (data.action === 'final_answer') {
                                latestVerboseFinalAnswer = String(data.answer || '');
                            }
                        } else if (data.type === 'iteration') {
                            // Fallback path: render logger iteration payloads when verbose callbacks are unavailable.
                            if (!sawVerboseTraceEvents && data.data) {
                                const iter = data.data;
                                const iterNo = iter.iteration || '?';
                                const responseText = escapeHtml(iter.response || '');
                                appendLiveBox(
                                    agentBody,
                                    `<div class="iteration-header">
                                        <span>Iteration #${iterNo}</span>
                                    </div>
                                    <div class="verbose-content" style="white-space:pre-wrap;">${responseText}</div>`,
                                    { variant: 'agent' },
                                );

                                const codeBlocks = Array.isArray(iter.code_blocks) ? iter.code_blocks : [];
                                for (const codeBlock of codeBlocks) {
                                    const result = codeBlock?.result || {};
                                    let html = `<strong>Code Execution:</strong><pre><code>${escapeHtml(codeBlock?.code || '')}</code></pre>`;
                                    if (result.stdout && String(result.stdout).trim()) {
                                        html += `<strong>Output:</strong><pre><code>${escapeHtml(result.stdout)}</code></pre>`;
                                    }
                                    if (result.stderr && String(result.stderr).trim()) {
                                        html += `<strong>Error:</strong><pre><code class="stream-error">${escapeHtml(result.stderr)}</code></pre>`;
                                    }
                                    appendLiveBox(agentBody, html, { variant: 'agent' });

                                    const calls = Array.isArray(result.rlm_calls) ? result.rlm_calls : [];
                                    for (const call of calls) {
                                        appendLiveBox(
                                            subagentBody,
                                            `<strong>Subagent Call (${escapeHtml(call?.root_model || 'unknown')}):</strong>
                                            <div class="stream-detail">Prompt: ${escapeHtml(call?.prompt || '')}</div>
                                            <div class="stream-detail">Response: ${escapeHtml(call?.response || '')}</div>`,
                                            { variant: 'subagent' },
                                        );
                                    }
                                }
                            }
                        } else if (data.type === 'metadata') {
                            appendLiveBox(agentBody, '<em>Logging subagent metadata...</em>');
                        } else if (data.type === 'done') {
                            let finalResponse = String(data.response || '');
                            if (!finalResponse.trim() || finalResponse.trim().toLowerCase() === 'verbose final_answer') {
                                finalResponse = latestVerboseFinalAnswer || finalResponse;
                            }
                            appendLiveBox(agentBody, '<em>Run complete.</em>');
                            appendMessage('bot', finalResponse, data.metrics);
                        }

                        scrollChatToBottom();
                    } catch (error) {
                        console.error('Failed to parse SSE JSON:', error, part);
                    }
                }
            }
        } catch (error) {
            console.error(error);
            if (liveContainer.parentNode) {
                liveContainer.remove();
            }
            appendMessage('error', `Network error: ${error.message}`);
        } finally {
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    });

    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitBtn.click();
        }
    });
});
