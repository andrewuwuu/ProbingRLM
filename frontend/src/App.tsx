import { For, Show, createSignal, onMount } from "solid-js";
import { createStore } from "solid-js/store";
import { marked } from "marked";

type MessageRole = "system" | "user" | "bot" | "error";
type StreamVariant = "agent" | "subagent";

interface QueryMetrics {
  mode?: string;
  execution_time?: number;
  total_tokens?: number;
  total_input_tokens?: number;
  total_output_tokens?: number;
  iterations?: number;
  subagent_calls?: number;
  [key: string]: unknown;
}

interface StandardMessage {
  id: string;
  role: MessageRole;
  content: string;
  metrics?: QueryMetrics;
}

interface StreamEntry {
  id: string;
  html: string;
  variant?: StreamVariant;
}

interface LiveMessage {
  id: string;
  role: "live";
  status: string;
  docsExpected: number;
  docsLoaded: number;
  docsLoadComplete: boolean;
  runComplete: boolean;
  agentEntries: StreamEntry[];
  subagentEntries: StreamEntry[];
  expandedAgent: boolean;
  expandedSubagent: boolean;
}

type ChatMessage = StandardMessage | LiveMessage;

interface StreamPayload {
  type?: string;
  detail?: string;
  doc?: string;
  num_docs?: number;
  requested_docs?: number;
  loaded_docs?: number;
  backend?: string;
  action?: string;
  iteration?: number | string;
  response?: string;
  answer?: string;
  code?: string;
  stdout?: string;
  stderr?: string;
  model?: string;
  prompt?: string;
  execution_time?: number;
  error?: string;
  data?: Record<string, unknown>;
  metrics?: QueryMetrics;
}

const nextId = () =>
  globalThis.crypto?.randomUUID?.() ??
  `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const escapeHtml = (value: unknown): string =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const parseMarkdown = (value: string): string => String(marked.parse(value));
const QUERY_INPUT_MAX_HEIGHT = 220;
const toSafeCount = (value: unknown): number => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return 0;
  }
  return Math.floor(parsed);
};

const getExpectedDocCount = (message: Partial<LiveMessage>): number => toSafeCount(message.docsExpected);
const getLoadedDocCount = (message: Partial<LiveMessage>): number => {
  const loaded = toSafeCount(message.docsLoaded);
  const expected = getExpectedDocCount(message);
  if (expected <= 0) {
    return loaded;
  }
  return Math.min(loaded, expected);
};

function App() {
  const [state, setState] = createStore({
    documents: [] as string[],
    selectedDocs: [] as string[],
    modelName: "",
    useSubagents: false,
    subagentBackend: "",
    subagentModel: "",
    systemPrompt: "",
    directChunkingEnabled: true,
    directChunkOverlapTokens: 256,
    directChunkMaxChunks: 64,
    openrouterMiddleOutFallback: true,
    subagentRootCompactionEnabled: true,
    subagentCompactionThresholdPct: 0.75,
    query: "",
    sending: false,
    messages: [
      {
        id: nextId(),
        role: "system",
        content:
          "Ready. Select documents and enter your query. Logs are written to rlm_logs/ for visualization."
      } as StandardMessage
    ] as ChatMessage[]
  });
  const [bannerError, setBannerError] = createSignal("");

  let chatContainerRef: HTMLDivElement | undefined;
  let queryInputRef: HTMLTextAreaElement | undefined;
  let documentSelectRef: HTMLSelectElement | undefined;

  const getSelectedDocuments = (): string[] => {
    const fromDom = documentSelectRef
      ? Array.from(documentSelectRef.selectedOptions).map((option) => option.value.trim())
      : [];
    if (fromDom.length > 0) {
      return fromDom.filter((name) => name.length > 0);
    }
    return state.selectedDocs
      .map((name) => String(name).trim())
      .filter((name) => name.length > 0);
  };

  const scrollToBottom = () => {
    queueMicrotask(() => {
      if (!chatContainerRef) return;
      chatContainerRef.scrollTop = chatContainerRef.scrollHeight;
    });
  };

  const appendMessage = (message: ChatMessage) => {
    setState("messages", (messages) => [...messages, message]);
    scrollToBottom();
  };

  const removeMessage = (messageId: string) => {
    setState("messages", (messages) => messages.filter((message) => message.id !== messageId));
  };

  const findLiveIndex = (messageId: string): number =>
    state.messages.findIndex((m) => m.role === "live" && m.id === messageId);

  const updateLiveMessage = (
    messageId: string,
    updater: (message: LiveMessage) => Partial<LiveMessage>
  ) => {
    const idx = findLiveIndex(messageId);
    if (idx < 0) return;
    const msg = state.messages[idx] as LiveMessage;
    const normalized: LiveMessage = {
      ...msg,
      status: typeof msg.status === "string" ? msg.status : "",
      docsExpected: getExpectedDocCount(msg),
      docsLoaded: getLoadedDocCount(msg),
      docsLoadComplete: Boolean(msg.docsLoadComplete),
      runComplete: Boolean(msg.runComplete),
      agentEntries: Array.isArray(msg.agentEntries) ? msg.agentEntries : [],
      subagentEntries: Array.isArray(msg.subagentEntries) ? msg.subagentEntries : [],
      expandedAgent: Boolean(msg.expandedAgent),
      expandedSubagent: Boolean(msg.expandedSubagent)
    };
    const updates = updater(normalized);
    setState("messages", idx, updates);
    scrollToBottom();
  };

  const appendLiveEntry = (
    messageId: string,
    panel: "agentEntries" | "subagentEntries",
    html: string,
    variant?: StreamVariant
  ) => {
    updateLiveMessage(messageId, (msg) => ({
      [panel]: [...(msg[panel] || []), { id: nextId(), html, variant }]
    }));
  };

  const setLiveStatus = (messageId: string, status: string) => {
    updateLiveMessage(messageId, () => ({ status }));
  };

  const updateLiveProgress = (
    messageId: string,
    updater: (message: LiveMessage) => Partial<LiveMessage>
  ) => {
    const idx = findLiveIndex(messageId);
    if (idx < 0) return;
    const msg = state.messages[idx] as LiveMessage;
    const normalized: LiveMessage = {
      ...msg,
      status: typeof msg.status === "string" ? msg.status : "",
      docsExpected: getExpectedDocCount(msg),
      docsLoaded: getLoadedDocCount(msg),
      docsLoadComplete: Boolean(msg.docsLoadComplete),
      runComplete: Boolean(msg.runComplete),
      agentEntries: Array.isArray(msg.agentEntries) ? msg.agentEntries : [],
      subagentEntries: Array.isArray(msg.subagentEntries) ? msg.subagentEntries : [],
      expandedAgent: Boolean(msg.expandedAgent),
      expandedSubagent: Boolean(msg.expandedSubagent)
    };
    const updates = updater(normalized);
    // Use the callback form for the specific index to avoid path-string inference issues
    setState("messages", idx, (prev) => ({ ...prev, ...updates } as LiveMessage));
    scrollToBottom();
  };

  const toggleExpanded = (messageId: string, key: "expandedAgent" | "expandedSubagent") => {
    updateLiveMessage(messageId, (msg) => ({
      [key]: !msg[key]
    }));
  };

  const fetchDocuments = async () => {
    setBannerError("");
    try {
      const response = await fetch("/api/documents");
      if (!response.ok) {
        throw new Error(`Failed to fetch documents (${response.status})`);
      }
      const data = (await response.json()) as { documents?: string[] };
      const documents = Array.isArray(data.documents) ? data.documents : [];
      setState("documents", documents);
      setState("selectedDocs", []);
    } catch (error) {
      setBannerError(error instanceof Error ? error.message : "Failed to load documents.");
    }
  };

  onMount(() => {
    void fetchDocuments();
  });

  const handleSubmit = async () => {
    const query = state.query.trim();
    const selectedDocuments = getSelectedDocuments();
    setState("selectedDocs", selectedDocuments);
    if (!selectedDocuments.length) {
      appendMessage({ id: nextId(), role: "error", content: "Select at least one document." });
      return;
    }
    if (!query) {
      appendMessage({ id: nextId(), role: "error", content: "Enter a query before sending." });
      return;
    }
    if (state.sending) {
      appendMessage({
        id: nextId(),
        role: "system",
        content: "A request is already running. Wait for completion before sending a new one."
      });
      return;
    }

    const payload = {
      documents: selectedDocuments,
      query,
      model_name: state.modelName.trim() || null,
      use_subagents: state.useSubagents,
      system_prompt: state.systemPrompt.trim() || null,
      direct_chunking_enabled: state.directChunkingEnabled,
      direct_chunk_overlap_tokens: state.directChunkOverlapTokens,
      direct_chunk_max_chunks: state.directChunkMaxChunks,
      openrouter_middle_out_fallback: state.openrouterMiddleOutFallback,
      subagent_root_compaction_enabled: state.subagentRootCompactionEnabled,
      subagent_compaction_threshold_pct: state.subagentCompactionThresholdPct,
      subagent_backend: state.subagentBackend.trim() || null,
      subagent_model: state.subagentModel.trim() || null
    };

    appendMessage({ id: nextId(), role: "user", content: query });
    setState("query", "");
    if (queryInputRef) {
      queryInputRef.style.height = "auto";
      queryInputRef.style.overflowY = "hidden";
    }

    const liveMessageId = nextId();
    appendMessage({
      id: liveMessageId,
      role: "live",
      status: "Waiting for server load events...",
      docsExpected: 0,
      docsLoaded: 0,
      docsLoadComplete: false,
      runComplete: false,
      agentEntries: [],
      subagentEntries: [],
      expandedAgent: false,
      expandedSubagent: false
    });

    setState("sending", true);

    try {
      const response = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed (${response.status})`);
      }
      if (!response.body) {
        throw new Error("Readable streams are not supported in this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let sawVerboseTraceEvents = false;
      let latestVerboseFinalAnswer = "";
      let stopStream = false;

      const parseStreamPayload = (rawPart: string): StreamPayload | null => {
        const part = rawPart.trim();
        if (!part) {
          return null;
        }
        if (part.startsWith(":")) {
          setLiveStatus(liveMessageId, "Stream keepalive received. Waiting for events...");
          return null;
        }

        const dataText = part
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        const jsonText = dataText || part;
        try {
          return JSON.parse(jsonText) as StreamPayload;
        } catch {
          return null;
        }
      };

      while (!stopStream) {
        const { done, value } = await reader.read();
        const decodedChunk = decoder
          .decode(value, { stream: !done })
          .replaceAll("\r\n", "\n")
          .replaceAll("\r", "\n");
        buffer += decodedChunk;
        const parts = buffer.split("\n\n");
        const trailingPart = parts.pop() || "";
        if (done) {
          if (trailingPart.trim()) {
            parts.push(trailingPart);
          }
          buffer = "";
        } else {
          buffer = trailingPart;
        }

        for (const rawPart of parts) {
          const data = parseStreamPayload(rawPart);
          if (!data) {
            continue;
          }

          if (data.type === "error") {
            removeMessage(liveMessageId);
            appendMessage({
              id: nextId(),
              role: "error",
              content: data.detail || "Server error occurred."
            });
            stopStream = true;
            break;
          }

          if (data.type === "stream_open") {
            setLiveStatus(liveMessageId, "SSE stream connected. Loading PDFs...");
            appendLiveEntry(
              liveMessageId,
              "agentEntries",
              "<em>Connected to live stream.</em>"
            );
            continue;
          }

          if (data.type === "load_start") {
            const requestedDocs = toSafeCount(data.requested_docs);
            updateLiveProgress(liveMessageId, () => ({
              docsExpected: requestedDocs,
              docsLoaded: 0,
              docsLoadComplete: false,
              status: `Loading PDFs (0/${requestedDocs})...`
            }));
            appendLiveEntry(
              liveMessageId,
              "agentEntries",
              `<em>Server started loading ${escapeHtml(requestedDocs)} document(s).</em>`
            );
            continue;
          }

          if (data.type === "model_resolved") {
            setLiveStatus(
              liveMessageId,
              `Using ${String(data.backend || "backend")} / ${String(data.model || "model")}`
            );
            appendLiveEntry(
              liveMessageId,
              "agentEntries",
              `<em>Model selected: ${escapeHtml(data.backend || "unknown")} / ${escapeHtml(
                data.model || "unknown"
              )}</em>`
            );
            continue;
          }

          if (data.type === "load_doc") {
            updateLiveProgress(liveMessageId, (message) => {
              const requestedDocs = toSafeCount(data.requested_docs) || message.docsExpected;
              const loadedDocs = toSafeCount(data.loaded_docs) || message.docsLoaded + 1;
              const boundedLoaded = requestedDocs > 0 ? Math.min(loadedDocs, requestedDocs) : loadedDocs;
              const progress = requestedDocs > 0 ? `${boundedLoaded}/${requestedDocs}` : String(boundedLoaded);
              return {
                docsExpected: requestedDocs,
                docsLoaded: boundedLoaded,
                status: `Loading PDFs (${progress})...`
              };
            });
            appendLiveEntry(
              liveMessageId,
              "agentEntries",
              `<strong>Loaded document:</strong> ${escapeHtml(data.doc)}`
            );
            continue;
          }

          if (data.type === "load_doc_empty") {
            updateLiveProgress(liveMessageId, (message) => {
              const requestedDocs = toSafeCount(data.requested_docs) || message.docsExpected;
              const loadedDocs = toSafeCount(data.loaded_docs);
              const progress = requestedDocs > 0 ? `${loadedDocs}/${requestedDocs}` : String(loadedDocs);
              return {
                docsExpected: requestedDocs,
                docsLoaded: loadedDocs,
                status: `Loading PDFs (${progress})...`
              };
            });
            appendLiveEntry(
              liveMessageId,
              "agentEntries",
              `<strong>Document had no extractable text:</strong> ${escapeHtml(data.doc)}`
            );
            continue;
          }

          if (data.type === "load_complete") {
            const loadedFromServer =
              typeof data.num_docs === "number" && Number.isFinite(data.num_docs)
                ? Math.max(0, data.num_docs)
                : null;
            updateLiveProgress(liveMessageId, (message) => {
              const loadedCount = loadedFromServer ?? message.docsLoaded;
              const requestedDocs = toSafeCount(data.requested_docs) || message.docsExpected;
              return {
                docsExpected: requestedDocs,
                docsLoaded: loadedCount,
                docsLoadComplete: true,
                status: "PDF loading complete. Running query..."
              };
            });
            appendLiveEntry(
              liveMessageId,
              "agentEntries",
              `<em>Loaded ${escapeHtml(loadedFromServer ?? "all selected")} document(s). Executing query...</em>`
            );
            continue;
          }

          if (data.type === "verbose") {
            if (data.action === "iteration_start_live" || data.action === "iteration_start") {
              sawVerboseTraceEvents = true;
              setLiveStatus(liveMessageId, `Running iteration #${String(data.iteration ?? "?")}...`);
              appendLiveEntry(
                liveMessageId,
                "agentEntries",
                `<div class="iteration-header"><span>Iteration #${escapeHtml(data.iteration)}</span><span class="spinner mini-spinner"></span></div><div class="verbose-content">Thinking...</div>`,
                "agent"
              );
            } else if (data.action === "completion") {
              sawVerboseTraceEvents = true;
              setLiveStatus(liveMessageId, "Model response received. Executing tools...");
              appendLiveEntry(
                liveMessageId,
                "agentEntries",
                `<div class="verbose-content">${escapeHtml(data.response || "")}</div>`,
                "agent"
              );
            } else if (data.action === "iteration_error") {
              sawVerboseTraceEvents = true;
              setLiveStatus(liveMessageId, "Iteration failed.");
              appendLiveEntry(
                liveMessageId,
                "agentEntries",
                `<div class="stream-error"><strong>Iteration error:</strong> ${escapeHtml(
                  data.detail || "Unknown error"
                )}</div>`,
                "agent"
              );
            } else if (data.action === "code_execution") {
              sawVerboseTraceEvents = true;
              setLiveStatus(liveMessageId, "Executing generated code...");
              let html = `<strong>Code execution:</strong><pre><code>${escapeHtml(data.code || "")}</code></pre>`;
              if ((data.stdout || "").trim()) {
                html += `<strong>Output:</strong><pre><code>${escapeHtml(data.stdout || "")}</code></pre>`;
              }
              if ((data.stderr || "").trim()) {
                html += `<strong>Error:</strong><pre><code class="stream-error">${escapeHtml(data.stderr || "")}</code></pre>`;
              }
              appendLiveEntry(liveMessageId, "agentEntries", html, "agent");
            } else if (data.action === "subcall_start") {
              sawVerboseTraceEvents = true;
              setLiveStatus(
                liveMessageId,
                `Subagent call in progress (${String(data.model || "unknown")})...`
              );
              appendLiveEntry(
                liveMessageId,
                "subagentEntries",
                `<div class="iteration-header"><span>Subagent call started (${escapeHtml(
                  data.model || "unknown"
                )})</span><span class="spinner mini-spinner"></span></div><div class="stream-detail">Prompt: ${escapeHtml(
                  data.prompt || ""
                )}</div>`,
                "subagent"
              );
            } else if (data.action === "subcall_complete") {
              sawVerboseTraceEvents = true;
              const timeText =
                typeof data.execution_time === "number"
                  ? ` (${Number(data.execution_time).toFixed(2)}s)`
                  : "";
              if (data.error) {
                setLiveStatus(
                  liveMessageId,
                  `Subagent call failed (${String(data.model || "unknown")}).`
                );
                appendLiveEntry(
                  liveMessageId,
                  "subagentEntries",
                  `<strong>Subagent call failed (${escapeHtml(
                    data.model || "unknown"
                  )})${escapeHtml(timeText)}:</strong><div class="stream-error">${escapeHtml(data.error)}</div>`,
                  "subagent"
                );
              } else {
                setLiveStatus(
                  liveMessageId,
                  `Subagent call completed (${String(data.model || "unknown")})${timeText}.`
                );
                appendLiveEntry(
                  liveMessageId,
                  "subagentEntries",
                  `<strong>Subagent call completed (${escapeHtml(
                    data.model || "unknown"
                  )})${escapeHtml(timeText)}:</strong><div class="stream-detail">Response: ${escapeHtml(
                    data.response || ""
                  )}</div>`,
                  "subagent"
                );
              }
            } else if (data.action === "subcall") {
              sawVerboseTraceEvents = true;
              appendLiveEntry(
                liveMessageId,
                "subagentEntries",
                `<strong>Subagent call (${escapeHtml(data.model || "unknown")}):</strong><div class="stream-detail">Prompt: ${escapeHtml(data.prompt || "")}</div><div class="stream-detail">Response: ${escapeHtml(data.response || "")}</div>`,
                "subagent"
              );
            } else if (data.action === "final_answer") {
              latestVerboseFinalAnswer = String(data.answer || "");
            }
            continue;
          }

          if (data.type === "iteration" && !sawVerboseTraceEvents && data.data) {
            const iteration = data.data;
            const iterNo = iteration.iteration ?? "?";
            const responseText = escapeHtml(iteration.response ?? "");
            appendLiveEntry(
              liveMessageId,
              "agentEntries",
              `<div class="iteration-header"><span>Iteration #${escapeHtml(iterNo)}</span></div><div class="verbose-content">${responseText}</div>`,
              "agent"
            );

            const codeBlocks = Array.isArray(iteration.code_blocks) ? iteration.code_blocks : [];
            for (const codeBlock of codeBlocks) {
              const result =
                codeBlock && typeof codeBlock === "object" && "result" in codeBlock
                  ? (codeBlock.result as Record<string, unknown>)
                  : {};
              let html = `<strong>Code execution:</strong><pre><code>${escapeHtml(codeBlock?.code ?? "")}</code></pre>`;
              if ((result.stdout || "").toString().trim()) {
                html += `<strong>Output:</strong><pre><code>${escapeHtml(result.stdout || "")}</code></pre>`;
              }
              if ((result.stderr || "").toString().trim()) {
                html += `<strong>Error:</strong><pre><code class="stream-error">${escapeHtml(result.stderr || "")}</code></pre>`;
              }
              appendLiveEntry(liveMessageId, "agentEntries", html, "agent");

              const calls = Array.isArray(result.rlm_calls) ? result.rlm_calls : [];
              for (const call of calls) {
                appendLiveEntry(
                  liveMessageId,
                  "subagentEntries",
                  `<strong>Subagent call (${escapeHtml(call?.root_model || "unknown")}):</strong><div class="stream-detail">Prompt: ${escapeHtml(call?.prompt || "")}</div><div class="stream-detail">Response: ${escapeHtml(call?.response || "")}</div>`,
                  "subagent"
                );
              }
            }
            continue;
          }

          if (data.type === "metadata") {
            setLiveStatus(liveMessageId, "Finalizing metadata...");
            appendLiveEntry(liveMessageId, "agentEntries", "<em>Logging subagent metadata...</em>");
            continue;
          }

          if (data.type === "done") {
            updateLiveProgress(liveMessageId, (message) => ({
              docsLoaded: Math.max(message.docsLoaded, message.docsExpected),
              docsLoadComplete: true,
              runComplete: true,
              status: "Run complete."
            }));
            let finalResponse = String(data.response || "");
            if (
              !finalResponse.trim() ||
              finalResponse.trim().toLowerCase() === "verbose final_answer"
            ) {
              finalResponse = latestVerboseFinalAnswer || finalResponse;
            }
            const metrics = data.metrics || {};
            if (
              metrics.mode === "rlm_subagents" &&
              Number(metrics.subagent_calls || 0) === 0
            ) {
              appendLiveEntry(
                liveMessageId,
                "subagentEntries",
                "<em>No subagent calls were made in this run. The root model completed directly.</em>",
                "subagent"
              );
            }
            appendLiveEntry(liveMessageId, "agentEntries", "<em>Run complete.</em>", "agent");
            appendMessage({
              id: nextId(),
              role: "bot",
              content: finalResponse,
              metrics: data.metrics
            });
            continue;
          }

          appendLiveEntry(
            liveMessageId,
            "agentEntries",
            `<em>Server event:</em> <code>${escapeHtml(data.type || "unknown")}</code>`
          );
        }
        if (done) {
          break;
        }
      }
    } catch (error) {
      removeMessage(liveMessageId);
      appendMessage({
        id: nextId(),
        role: "error",
        content: error instanceof Error ? error.message : "Network error."
      });
    } finally {
      setState("sending", false);
    }
  };

  return (
    <div class="app-shell">
      <div class="background-glow glow-one" />
      <div class="background-glow glow-two" />

      <header class="app-header panel">
        <div>
          <h1>
            Probing<span>RLM</span>
          </h1>
          <p>Recursive Reasoning Engine</p>
        </div>
        <button type="button" class="secondary-btn" onClick={() => void fetchDocuments()}>
          Refresh Documents
        </button>
      </header>

      <Show when={bannerError()}>
        <section class="banner-error">{bannerError()}</section>
      </Show>

      <main class="content-grid">
        <aside class="sidebar panel">
          <h2>Configuration</h2>

          <div class="form-group">
            <label for="document-select">Select Documents</label>
            <select
              id="document-select"
              multiple
              ref={documentSelectRef}
              value={state.selectedDocs}
              onChange={(event) => {
                const selected = Array.from(event.currentTarget.selectedOptions).map(
                  (option) => option.value
                );
                setState("selectedDocs", selected);
              }}
            >
              <Show
                when={state.documents.length > 0}
                fallback={
                  <option disabled value="">
                    No docs in embed-docs/
                  </option>
                }
              >
                <For each={state.documents}>
                  {(documentName) => <option value={documentName}>{documentName}</option>}
                </For>
              </Show>
            </select>
            <small>Use Ctrl/Cmd click for multi-select.</small>
          </div>

          <div class="form-group">
            <label for="model-input">Model Name (optional)</label>
            <input
              id="model-input"
              type="text"
              placeholder="e.g. gpt-4.1-mini"
              value={state.modelName}
              onInput={(event) => setState("modelName", event.currentTarget.value)}
            />
          </div>

          <label class="toggle-row" for="use-subagents">
            <input
              id="use-subagents"
              type="checkbox"
              checked={state.useSubagents}
              onChange={(event) => setState("useSubagents", event.currentTarget.checked)}
            />
            <span>Enable Subagents</span>
          </label>

          <label class="toggle-row" for="direct-chunking-enabled">
            <input
              id="direct-chunking-enabled"
              type="checkbox"
              checked={state.directChunkingEnabled}
              onChange={(event) => setState("directChunkingEnabled", event.currentTarget.checked)}
            />
            <span>Enable Direct Chunking</span>
          </label>

          <label class="toggle-row" for="openrouter-middle-out">
            <input
              id="openrouter-middle-out"
              type="checkbox"
              checked={state.openrouterMiddleOutFallback}
              onChange={(event) => setState("openrouterMiddleOutFallback", event.currentTarget.checked)}
            />
            <span>Enable OpenRouter Middle-Out Retry</span>
          </label>

          <div class="form-group">
            <label for="direct-chunk-overlap">Chunk Overlap Tokens</label>
            <input
              id="direct-chunk-overlap"
              type="number"
              min="0"
              step="1"
              value={state.directChunkOverlapTokens}
              onInput={(event) => {
                const parsed = Number.parseInt(event.currentTarget.value, 10);
                setState(
                  "directChunkOverlapTokens",
                  Number.isFinite(parsed) && parsed >= 0 ? parsed : 0
                );
              }}
            />
          </div>

          <div class="form-group">
            <label for="direct-chunk-max">Max Chunks</label>
            <input
              id="direct-chunk-max"
              type="number"
              min="1"
              step="1"
              value={state.directChunkMaxChunks}
              onInput={(event) => {
                const parsed = Number.parseInt(event.currentTarget.value, 10);
                setState(
                  "directChunkMaxChunks",
                  Number.isFinite(parsed) && parsed > 0 ? parsed : 1
                );
              }}
            />
          </div>

          <Show when={state.useSubagents}>
            <div class="subagent-panel">
              <label class="toggle-row" for="subagent-root-compaction">
                <input
                  id="subagent-root-compaction"
                  type="checkbox"
                  checked={state.subagentRootCompactionEnabled}
                  onChange={(event) =>
                    setState("subagentRootCompactionEnabled", event.currentTarget.checked)
                  }
                />
                <span>Enable Root Compaction Guard</span>
              </label>

              <div class="form-group">
                <label for="subagent-compaction-threshold">Root Compaction Threshold (0.10-0.99)</label>
                <input
                  id="subagent-compaction-threshold"
                  type="number"
                  min="0.10"
                  max="0.99"
                  step="0.01"
                  value={state.subagentCompactionThresholdPct}
                  onInput={(event) => {
                    const parsed = Number.parseFloat(event.currentTarget.value);
                    let normalized = Number.isFinite(parsed) ? parsed : 0.75;
                    if (normalized < 0.1) normalized = 0.1;
                    if (normalized > 0.99) normalized = 0.99;
                    setState("subagentCompactionThresholdPct", Number(normalized.toFixed(2)));
                  }}
                />
              </div>

              <div class="form-group">
                <label for="subagent-backend">Subagent Backend</label>
                <input
                  id="subagent-backend"
                  type="text"
                  placeholder="e.g. openai"
                  value={state.subagentBackend}
                  onInput={(event) => setState("subagentBackend", event.currentTarget.value)}
                />
              </div>
              <div class="form-group">
                <label for="subagent-model">Subagent Model</label>
                <input
                  id="subagent-model"
                  type="text"
                  placeholder="e.g. gpt-4.1-mini"
                  value={state.subagentModel}
                  onInput={(event) => setState("subagentModel", event.currentTarget.value)}
                />
              </div>
            </div>
          </Show>

          <div class="form-group">
            <label for="system-prompt">System Prompt (optional)</label>
            <textarea
              id="system-prompt"
              rows="4"
              placeholder="Optional system instruction..."
              value={state.systemPrompt}
              onInput={(event) => setState("systemPrompt", event.currentTarget.value)}
            />
          </div>
        </aside>

        <section class="main-panel panel">
          <div class="chat-container" ref={chatContainerRef}>
            <div class="chat-history">
              <For each={state.messages}>
                {(message) => (
                  <Show
                    when={message.role === "live" && (message as LiveMessage)}
                    fallback={
                      <article class={`message ${(message as StandardMessage).role}`}>
                        <Show
                          when={(message as StandardMessage).role === "bot"}
                          fallback={<div>{(message as StandardMessage).content}</div>}
                        >
                          <div
                            class="markdown-body"
                            innerHTML={parseMarkdown((message as StandardMessage).content)}
                          />
                        </Show>

                        <Show when={(message as StandardMessage).metrics}>
                          {(metrics) => (
                            <section class="metadata-block">
                              <strong>Metrics ({metrics().mode || "unknown"})</strong>
                              <ul>
                                <Show when={typeof metrics().execution_time === "number"}>
                                  <li>Time: {Number(metrics().execution_time).toFixed(2)}s</li>
                                </Show>
                                <Show when={metrics().total_tokens}>
                                  <li>
                                    Tokens: {metrics().total_tokens} (I: {metrics().total_input_tokens || 0}
                                    /O: {metrics().total_output_tokens || 0})
                                  </li>
                                </Show>
                                <Show when={metrics().iterations}>
                                  <li>Iterations: {metrics().iterations}</li>
                                </Show>
                                <Show when={metrics().subagent_calls !== undefined}>
                                  <li>Subagent Calls: {metrics().subagent_calls}</li>
                                </Show>
                              </ul>
                            </section>
                          )}
                        </Show>
                      </article>
                    }
                  >
                    {(live) => (
                      <article class="message system live-message">
                        {(() => {
                          const expectedDocs = getExpectedDocCount(live());
                          const loadedDocs = getLoadedDocCount(live());
                          return (
                            <div class="live-status-row">
                              <span class={`status-pill ${live().docsLoadComplete ? "is-done" : "is-active"}`}>
                                <Show
                                  when={live().docsLoadComplete}
                                  fallback={`PDFs loading (${loadedDocs}/${expectedDocs})`}
                                >
                                  {`PDFs loaded (${loadedDocs}/${expectedDocs})`}
                                </Show>
                              </span>
                              <span
                                class={`status-pill ${live().runComplete
                                  ? "is-done"
                                  : live().docsLoadComplete
                                    ? "is-active"
                                    : "is-pending"
                                  }`}
                              >
                                <Show
                                  when={live().runComplete}
                                  fallback={<Show when={live().docsLoadComplete} fallback="Waiting for PDFs">Query running</Show>}
                                >
                                  Query complete
                                </Show>
                              </span>
                            </div>
                          );
                        })()}
                        <Show when={live().status}>
                          <div class="loading-indicator">{live().status}</div>
                        </Show>

                        <div class="stream-grid">
                          <section
                            class={`stream-panel ${live().expandedAgent ? "is-expanded" : ""}`}
                          >
                            <div class="stream-panel-header">
                              <h4 class="stream-title">Agent Output</h4>
                              <button
                                type="button"
                                class="stream-expand-btn"
                                aria-expanded={live().expandedAgent}
                                onClick={() => toggleExpanded(live().id, "expandedAgent")}
                              >
                                {live().expandedAgent ? "Collapse" : "Expand"}
                              </button>
                            </div>
                            <div class="stream-body">
                              <For each={live().agentEntries}>
                                {(entry) => (
                                  <div
                                    class={`iteration-box ${entry.variant ? `iteration-${entry.variant}` : ""
                                      }`}
                                    innerHTML={entry.html}
                                  />
                                )}
                              </For>
                            </div>
                          </section>

                          <section
                            class={`stream-panel ${live().expandedSubagent ? "is-expanded" : ""}`}
                          >
                            <div class="stream-panel-header">
                              <h4 class="stream-title">Subagent Output</h4>
                              <button
                                type="button"
                                class="stream-expand-btn"
                                aria-expanded={live().expandedSubagent}
                                onClick={() => toggleExpanded(live().id, "expandedSubagent")}
                              >
                                {live().expandedSubagent ? "Collapse" : "Expand"}
                              </button>
                            </div>
                            <div class="stream-body">
                              <For each={live().subagentEntries}>
                                {(entry) => (
                                  <div
                                    class={`iteration-box ${entry.variant ? `iteration-${entry.variant}` : ""
                                      }`}
                                    innerHTML={entry.html}
                                  />
                                )}
                              </For>
                            </div>
                          </section>
                        </div>
                      </article>
                    )}
                  </Show>
                )}
              </For>
            </div>
          </div>

          <section class="input-area">
            <textarea
              ref={queryInputRef}
              rows="3"
              placeholder="Ask a question about the selected documents..."
              value={state.query}
              onInput={(event) => {
                setState("query", event.currentTarget.value);
                event.currentTarget.style.height = "auto";
                const nextHeight = Math.min(event.currentTarget.scrollHeight, QUERY_INPUT_MAX_HEIGHT);
                event.currentTarget.style.height = `${nextHeight}px`;
                event.currentTarget.style.overflowY =
                  event.currentTarget.scrollHeight > QUERY_INPUT_MAX_HEIGHT ? "auto" : "hidden";
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSubmit();
                }
              }}
            />
            <button
              type="button"
              class="primary-btn"
              disabled={state.sending}
              onClick={() => void handleSubmit()}
            >
              <Show
                when={state.sending}
                fallback={<span class="btn-label">Send Query</span>}
              >
                <span class="spinner" />
              </Show>
            </button>
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
