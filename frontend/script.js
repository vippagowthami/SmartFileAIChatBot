// Configuration
let CONFIG = {
    BACKEND_URL: "http://127.0.0.1:8001",
    USE_RAG: true,
    NUM_RETRIEVAL: 5,
    MODEL: "llama3",
    QUERY_TIMEOUT_MS: 130000,
    THEME: "light",
    TEMPERATURE: 0.1
};

let documents = [];
let isWaitingForResponse = false;
let chatHistory = [];
let currentChatId = null;

// ==================== Init ====================
document.addEventListener("DOMContentLoaded", () => {
    loadSettings();
    setStatus("Connecting...", "");
    checkHealth();
    loadChatHistory();
    loadIndexedFiles();

    const saved = localStorage.getItem("currentChatId");
    if (saved && chatHistory.length > 0) {
        const sessionMsgs = chatHistory.filter(m => m.chatId === saved);
        if (sessionMsgs.length > 0) { loadChatSession(saved); return; }
    }
    newChat();
    updateStatistics();
    setInterval(updateStatistics, 30000);
});

// ==================== Health ====================
async function checkHealth() {
    try {
        const res = await fetchWithTimeout(`${CONFIG.BACKEND_URL}/health`, 7000);
        const data = await res.json();
        if (res.ok) {
            setStatus(data.ollama?.available ? "Connected" : "Connected (local mode)", "connected");
        }
        return res.ok;
    } catch {
        setStatus("Backend offline — start the server", "error");
        return false;
    }
}

function fetchWithTimeout(url, timeoutMs = 10000, options = {}) {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), timeoutMs);
    return fetch(url, { ...options, signal: ctrl.signal }).finally(() => clearTimeout(id));
}

function setStatus(message, className = "") {
    const el = document.getElementById("status-indicator");
    el.textContent = message;
    el.className = `status ${className}`;
}

// ==================== Chat ====================
async function sendMessage() {
    const input = document.getElementById("message-input");
    const message = input.value.trim();
    if (!message || isWaitingForResponse) return;

    input.value = "";
    input.disabled = true;
    addMessageToChat(message, "user");

    const loadingId = addLoadingIndicator();
    isWaitingForResponse = true;
    setStatus("Thinking...", "");

    try {
        const useRAG = document.getElementById("use-rag").checked;

        const response = await fetchWithTimeout(
            `${CONFIG.BACKEND_URL}/query`,
            CONFIG.QUERY_TIMEOUT_MS,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    question: message, 
                    use_rag: useRAG,
                    temperature: CONFIG.TEMPERATURE ?? 0.1
                })
            }
        );

        removeLoadingIndicator(loadingId);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to get response");
        }

        const data = await response.json();
        const formattedResponse = formatResponse(data);
        addMessageToChat(formattedResponse, "ai", data);
        setStatus("Connected", "connected");

    } catch (error) {
        removeLoadingIndicator(loadingId);
        const msg = error.name === "AbortError"
            ? "Request timed out — the model took too long. Try a shorter question."
            : `Error: ${error.message}`;
        addMessageToChat(msg, "ai-error");
        setStatus("Error occurred", "error");
    } finally {
        isWaitingForResponse = false;
        input.disabled = false;
        input.focus();
    }
}

function handleKeyPress(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function addMessageToChat(content, type, data = null) {
    const chatContainer = document.getElementById("chat-container");
    const welcome = chatContainer.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}`;

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "message-bubble";

    if (type === "ai-error") {
        bubbleDiv.style.color = "#d32f2f";
        bubbleDiv.textContent = content;
    } else if (type === "ai") {
        bubbleDiv.innerHTML = content;
    } else {
        bubbleDiv.textContent = content;
    }

    messageDiv.appendChild(bubbleDiv);
    saveChatMessage(content, type, data);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addLoadingIndicator() {
    const chatContainer = document.getElementById("chat-container");
    const messageDiv = document.createElement("div");
    messageDiv.className = "message ai";
    messageDiv.id = "loading-" + Date.now();

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "message-bubble";
    bubbleDiv.innerHTML = `
        <div class="loading">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>`;

    messageDiv.appendChild(bubbleDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageDiv.id;
}

function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ==================== Response Formatting ====================
/**
 * Converts plain text with basic markdown to safe HTML.
 * Handles: **bold**, *italic*, `code`, numbered lists, bullet lists, newlines.
 */
function markdownToHtml(text) {
    // Escape HTML first
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");

    // Code blocks (```...```)
    html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Numbered list items (1. 2. etc.)
    html = html.replace(/(?:^|\n)(\d+)\.\s+(.+)/g, (_, n, item) =>
        `\n<li>${item}</li>`
    );

    // Bullet list items (- or *)
    html = html.replace(/(?:^|\n)[-•]\s+(.+)/g, (_, item) =>
        `\n<li>${item}</li>`
    );

    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>[\s\S]*?<\/li>(\s*<li>[\s\S]*?<\/li>)*)/g, "<ul>$1</ul>");

    // Paragraphs: double newline → paragraph break
    html = html.replace(/\n\n+/g, "</p><p>");

    // Single newlines → <br>
    html = html.replace(/\n/g, "<br>");

    return `<p>${html}</p>`;
}

function formatResponse(data) {
    let html = markdownToHtml(data.answer);

    if (data.retrieved_sources && data.retrieved_sources.length > 0) {
        const seen = new Set();
        const uniqueSources = [];
        data.retrieved_sources.forEach(src => {
            const name = (src.source || "").split(/[/\\]/).pop();
            if (name && !seen.has(name)) {
                seen.add(name);
                uniqueSources.push({ name, similarity: src.similarity });
            }
        });

        if (uniqueSources.length > 0) {
            const badges = uniqueSources.map(s => {
                const pct = s.similarity != null ? ` (${Math.round(s.similarity * 100)}%)` : "";
                return `<span class="source-badge">📄 ${s.name}${pct}</span>`;
            }).join(" ");
            html += `<div class="source-info">${badges}</div>`;
        }
    }

    // Timing info
    if (data.timings && data.timings.total) {
        html += `<div class="timing-info">⏱ ${data.timings.total.toFixed(2)}s</div>`;
    }

    return html;
}

// ==================== File Upload ====================
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const uploadProgress = document.getElementById("upload-progress");
    uploadProgress.className = "upload-progress";
    uploadProgress.classList.remove("is-processing");
    uploadProgress.style.color = "";
    setUploadProgress(0, `Preparing ${file.name}`, "Starting upload...");

    try {
        const responseData = await uploadFileWithProgress(file, (percent, phaseText, detailText) => {
            setUploadProgress(percent, phaseText, detailText);
        });

        const details = responseData.details || {};
        const chunks = details.chunks_created ?? "?";
        const elapsed = details.total_time != null ? `${details.total_time}s` : "";
        setUploadProgress(100, `✅ ${file.name} indexed`, `${chunks} chunks${elapsed ? " in " + elapsed : ""}`);
        uploadProgress.style.color = "#388e3c";
        uploadProgress.classList.remove("is-processing");

        if (!documents.includes(file.name)) {
            documents.push(file.name);
            updateDocumentsList();
        }

        loadIndexedFiles();
        await updateStatistics();

        setTimeout(() => {
            uploadProgress.className = "upload-progress hidden";
            document.getElementById("file-input").value = "";
        }, 3000);

    } catch (error) {
        uploadProgress.classList.remove("is-processing");
        setUploadProgress(0, "❌ Upload failed", error.message);
        uploadProgress.style.color = "#d32f2f";
        document.getElementById("file-input").value = "";
    }
}

function setUploadProgress(percent, title, detail) {
    const uploadProgress = document.getElementById("upload-progress");
    const progressRing = document.getElementById("upload-progress-ring-fill");
    const progressPercent = document.getElementById("upload-progress-percent");
    const progressText = document.getElementById("upload-progress-text");
    const progressSubtext = document.getElementById("upload-progress-subtext");

    const clamped = Math.max(0, Math.min(100, Math.round(percent)));
    const angle = Math.round((clamped / 100) * 360);
    progressRing.style.background = `conic-gradient(var(--primary-color) ${angle}deg, #e7eef5 ${angle}deg)`;
    progressPercent.textContent = `${clamped}%`;
    progressText.textContent = title;
    progressSubtext.textContent = detail || "";
    uploadProgress.classList.toggle("is-processing", clamped >= 100 && title.toLowerCase().includes("processing"));
}

function uploadFileWithProgress(file, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append("file", file);

        xhr.open("POST", `${CONFIG.BACKEND_URL}/upload`, true);

        xhr.upload.onprogress = (event) => {
            if (event.lengthComputable) {
                const percent = (event.loaded / event.total) * 75; // 0-75% for upload
                onProgress(percent, `Uploading ${file.name}`,
                    `${Math.round(event.loaded / 1024)} KB of ${Math.round(event.total / 1024)} KB`);
            } else {
                onProgress(40, `Uploading ${file.name}`, "Uploading...");
            }
        };

        xhr.onload = () => onProgress(85, `Processing ${file.name}`, "Embedding chunks into ChromaDB...");

        xhr.onreadystatechange = () => {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch {
                        reject(new Error("Invalid response from server"));
                    }
                } else {
                    let msg = "Upload failed";
                    try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
                    reject(new Error(msg));
                }
            }
        };

        xhr.onerror = () => reject(new Error("Network error while uploading"));
        xhr.onloadstart = () => onProgress(0, `Uploading ${file.name}`, "Starting...");
        xhr.send(formData);
    });
}

function updateDocumentsList() {
    const listEl = document.getElementById("documents-list");
    if (documents.length === 0) {
        listEl.innerHTML = '<p class="empty-state">No documents uploaded</p>';
        return;
    }
    listEl.innerHTML = documents
        .map(doc => `<div class="document-item">📄 ${doc}</div>`)
        .join("");
}

async function loadIndexedFiles() {
    try {
        const res = await fetchWithTimeout(`${CONFIG.BACKEND_URL}/indexed-files`, 10000);
        if (!res.ok) return;
        const data = await res.json();
        documents = Array.isArray(data.files) ? data.files : [];
        updateDocumentsList();
    } catch (e) {
        console.error("Failed to load indexed files:", e);
    }
}

// ==================== Statistics ====================
async function updateStatistics() {
    try {
        const res = await fetchWithTimeout(`${CONFIG.BACKEND_URL}/statistics`, 10000);
        const data = await res.json();
        const stats = data.statistics;
        const db = stats.database;

        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val ?? 0;
        };

        set("doc-count", db.indexed_files ?? db.total_documents ?? 0);
        set("indexed-file-count", db.indexed_files ?? db.total_documents ?? 0);
        set("chunk-count", db.total_chunks ?? 0);

        const modelEl = document.getElementById("model-name");
        if (modelEl) modelEl.textContent = stats.llm_model || CONFIG.MODEL;

    } catch (e) {
        console.error("Failed to update statistics:", e);
    }
}

// ==================== Database ====================
async function clearDatabase() {
    if (!confirm("Clear ALL documents? This cannot be undone.")) return;
    try {
        const res = await fetch(`${CONFIG.BACKEND_URL}/clear-database`, { method: "POST" });
        if (!res.ok) throw new Error("Failed to clear database");
        documents = [];
        updateDocumentsList();
        await updateStatistics();
        addMessageToChat("📚 Database cleared successfully.", "ai");
    } catch (e) {
        addMessageToChat(`Error clearing database: ${e.message}`, "ai-error");
    }
}

// ==================== Settings ====================
function toggleSettings() {
    const modal = document.getElementById("settings-modal");
    modal.classList.toggle("hidden");
    if (!modal.classList.contains("hidden")) {
        modal.style.display = "flex";
        loadSettingsIntoModal();
    } else {
        modal.style.display = "none";
    }
}

function loadSettingsIntoModal() {
    const el = (id) => document.getElementById(id);
    if (el("backend-url")) el("backend-url").value = CONFIG.BACKEND_URL;
    if (el("model-select")) el("model-select").value = CONFIG.MODEL;
    if (el("num-retrieval")) el("num-retrieval").value = CONFIG.NUM_RETRIEVAL;
    if (el("theme-select")) el("theme-select").value = CONFIG.THEME || "light";
    if (el("temperature")) el("temperature").value = CONFIG.TEMPERATURE ?? 0.1;
    if (el("temp-value")) el("temp-value").innerText = CONFIG.TEMPERATURE ?? 0.1;
}

function saveSettings() {
    const el = (id) => document.getElementById(id);
    if (el("backend-url")) CONFIG.BACKEND_URL = el("backend-url").value || CONFIG.BACKEND_URL;
    if (el("model-select")) CONFIG.MODEL = el("model-select").value;
    if (el("num-retrieval")) CONFIG.NUM_RETRIEVAL = parseInt(el("num-retrieval").value);
    if (el("theme-select")) CONFIG.THEME = el("theme-select").value;
    if (el("temperature")) CONFIG.TEMPERATURE = parseFloat(el("temperature").value);
    
    localStorage.setItem("chatbot-config", JSON.stringify(CONFIG));
    applyTheme();
    toggleSettings();
    addMessageToChat("⚙️ Settings saved.", "ai");
}

function applyTheme() {
    if (CONFIG.THEME === "dark") {
        document.body.classList.add("dark-theme");
    } else {
        document.body.classList.remove("dark-theme");
    }
}

function loadSettings() {
    const saved = localStorage.getItem("chatbot-config");
    if (saved) {
        try {
            const parsed = JSON.parse(saved);
            CONFIG = { ...CONFIG, ...parsed };
            // Fix legacy port
            if (typeof CONFIG.BACKEND_URL === "string" && CONFIG.BACKEND_URL.includes(":8000")) {
                CONFIG.BACKEND_URL = CONFIG.BACKEND_URL.replace(":8000", ":8001");
            }
            localStorage.setItem("chatbot-config", JSON.stringify(CONFIG));
            applyTheme();
        } catch {}
    }
}

function changeModel() {
    CONFIG.MODEL = document.getElementById("model-select").value;
    localStorage.setItem("chatbot-config", JSON.stringify(CONFIG));
}

// ==================== Chat History ====================
function saveChatMessage(content, type, data) {
    if (!currentChatId) currentChatId = Date.now().toString();
    chatHistory.push({
        chatId: currentChatId,
        timestamp: new Date().toISOString(),
        type,
        content,
        data
    });
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    localStorage.setItem("currentChatId", currentChatId);
    syncChatHistoryToBackend();
}

async function loadChatHistory() {
    const saved = localStorage.getItem("chatHistory");
    let local = [];
    try { local = saved ? JSON.parse(saved) : []; } catch {}

    try {
        const res = await fetchWithTimeout(`${CONFIG.BACKEND_URL}/chat-history`, 10000);
        if (res.ok) {
            const data = await res.json();
            if (Array.isArray(data.messages)) {
                if (data.messages.length === 0 && local.length > 0) {
                    chatHistory = local;
                    syncChatHistoryToBackend();
                } else {
                    chatHistory = data.messages;
                }
                localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
                return;
            }
        }
    } catch {}
    chatHistory = local;
}

function syncChatHistoryToBackend() {
    fetch(`${CONFIG.BACKEND_URL}/chat-history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: chatHistory }),
    }).catch(() => {});
}

function showChatHistory() {
    const existing = document.getElementById("history-modal");
    if (existing) existing.remove();

    const modal = document.createElement("div");
    modal.className = "modal";
    modal.id = "history-modal";
    modal.style.display = "flex";

    const grouped = {};
    chatHistory.forEach(msg => {
        if (!grouped[msg.chatId]) grouped[msg.chatId] = [];
        grouped[msg.chatId].push(msg);
    });

    let items = "";
    Object.keys(grouped).reverse().forEach(chatId => {
        const msgs = grouped[chatId];
        const first = msgs.find(m => m.type === "user");
        const preview = first ? first.content.substring(0, 60) : "(No messages)";
        const date = new Date(msgs[0].timestamp).toLocaleDateString();
        items += `<div class="history-item-row">
                    <div class="history-item-content" onclick="loadChatSession('${chatId}');this.closest('.modal').remove()">
                        <strong>${date}</strong><br/><small>${preview}…</small>
                    </div>
                    <button class="history-delete-btn" onclick="deleteChatSession('${chatId}', this)" title="Delete Chat">🗑️</button>
                  </div>`;
    });

    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2>Chat History</h2>
                <button class="close-btn" onclick="this.closest('.modal').remove()">✕</button>
            </div>
            <div style="max-height:400px;overflow-y:auto">
                ${items || "<p style='padding:16px'>No chat history yet</p>"}
            </div>
            ${items ? `<div class="modal-footer" style="justify-content: flex-start; padding-top:12px">
                <button class="clear-db-btn" onclick="clearAllChatHistory()">🗑️ Clear All History</button>
            </div>` : ""}
        </div>`;
    document.body.appendChild(modal);
}

function deleteChatSession(chatId, btnEl) {
    if (!confirm("Delete this chat session?")) return;
    chatHistory = chatHistory.filter(m => m.chatId !== chatId);
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    syncChatHistoryToBackend();
    
    if (btnEl) {
        btnEl.closest('.history-item-row').remove();
    }
    
    if (currentChatId === chatId) {
        newChat();
    }
}

function clearAllChatHistory() {
    if (!confirm("Are you sure you want to delete ALL chat history? This cannot be undone.")) return;
    chatHistory = [];
    localStorage.removeItem("chatHistory");
    currentChatId = null;
    localStorage.removeItem("currentChatId");
    syncChatHistoryToBackend();
    
    const modal = document.getElementById("history-modal");
    if (modal) modal.remove();
    
    newChat();
    addMessageToChat("🗑️ All chat history has been deleted.", "ai");
}

function loadChatSession(chatId) {
    const chatContainer = document.getElementById("chat-container");
    chatContainer.innerHTML = "";
    chatHistory.filter(m => m.chatId === chatId).forEach(msg => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${msg.type}`;
        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.innerHTML = msg.content;
        msgDiv.appendChild(bubble);
        chatContainer.appendChild(msgDiv);
    });
    currentChatId = chatId;
    localStorage.setItem("currentChatId", currentChatId);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ==================== Chat Control ====================
function newChat() {
    currentChatId = null;
    localStorage.removeItem("currentChatId");
    document.getElementById("chat-container").innerHTML = `
        <div class="welcome-message">
            <h2>Smart File AI Chatbot</h2>
            <p>Upload a document and ask questions — or ask anything general!</p>
            <div class="welcome-tips">
                <p><strong>Tips:</strong></p>
                <ul>
                    <li>Upload PDF, DOCX, or TXT files for source-based answers</li>
                    <li>Ask general questions even without uploading files</li>
                    <li>Toggle <em>Use RAG</em> off for pure general answers</li>
                    <li>All processing is local — no internet required</li>
                </ul>
            </div>
        </div>`;
    document.getElementById("message-input").focus();
}
