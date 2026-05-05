// Configuration
let CONFIG = {
    BACKEND_URL: "http://127.0.0.1:8001",
    USE_RAG: true,
    NUM_RETRIEVAL: 5,
    MODEL: "llama3",
    QUERY_TIMEOUT_MS: 130000,
    THEME: "light",
    TEMPERATURE: 0.1,
    VOICE: {
        ENABLED: true,
        WAKE_ENABLED: false,
        WAKE_TEXT: "Hey Smart",
        STT_MODEL: "base",
        TTS_ENGINE: "piper",
        AUTO_PLAY: false,
        SILENCE_MS: 1500,
    },
};

let documents = [];
let isWaitingForResponse = false;
let chatHistory = [];
let currentChatId = null;
let currentAudio = null;
let wakePollTimer = null;
let wakeEventCursor = 0;
let aiMessageCounter = 0;
const ttsCache = new Map();
// New map to track per-message audio objects
const ttsAudioMap = new Map();

// Voice runtime state
let voiceState = "idle"; // idle | listening | processing | speaking
let mediaRecorder = null;
let mediaStream = null;
let audioContext = null;
let analyser = null;
let waveformArray = null;
let waveformAnimationFrame = null;
let silenceStartAt = 0;
let recordedChunks = [];
let recordingTrigger = "manual";

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
        if (sessionMsgs.length > 0) {
            loadChatSession(saved);
        } else {
            newChat();
        }
    } else {
        newChat();
    }

    setSidebarPane("documents");
    updateStatistics();
    setInterval(updateStatistics, 30000);

    initVoiceUI();
    syncVoiceSettingsToBackend();
    updateWakePolling();
});

function setSidebarPane(pane) {
    const docsPane = document.getElementById("pane-documents");
    const statsPane = document.getElementById("pane-statistics");
    const docsBtn = document.getElementById("menu-documents");
    const statsBtn = document.getElementById("menu-statistics");
    if (!docsPane || !statsPane || !docsBtn || !statsBtn) return;

    if (pane === "statistics") {
        docsPane.classList.add("hidden");
        statsPane.classList.remove("hidden");
        docsBtn.classList.remove("active");
        statsBtn.classList.add("active");
        return;
    }

    statsPane.classList.add("hidden");
    docsPane.classList.remove("hidden");
    statsBtn.classList.remove("active");
    docsBtn.classList.add("active");
}

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
    if (!el) return;
    el.textContent = message;
    el.className = `status ${className}`;
}

// ==================== Voice UI ====================
function initVoiceUI() {
    const autoPlay = document.getElementById("autoplay-toggle");
    if (autoPlay) autoPlay.checked = !!CONFIG.VOICE.AUTO_PLAY;
    setVoiceState("idle");
    setWakeIndicator(false, "");
}

function setVoiceState(nextState, label = "") {
    const btn = document.getElementById("voice-btn");
    const icon = document.getElementById("voice-icon");
    const visualizer = document.getElementById("voice-visualizer");
    const stateText = document.getElementById("voice-state-text");
    if (!btn || !icon || !visualizer || !stateText) return;

    voiceState = nextState;
    btn.classList.remove("state-idle", "state-listening", "state-processing", "state-speaking");

    if (nextState === "idle") {
        btn.classList.add("state-idle");
        icon.textContent = "🎤";
        stateText.textContent = label || "Listening...";
        visualizer.classList.add("hidden");
        return;
    }

    if (nextState === "listening") {
        btn.classList.add("state-listening");
        icon.textContent = "🔴";
        stateText.textContent = label || "Listening...";
        visualizer.classList.remove("hidden");
        return;
    }

    if (nextState === "processing") {
        btn.classList.add("state-processing");
        icon.textContent = "⏳";
        stateText.textContent = label || "Transcribing...";
        visualizer.classList.remove("hidden");
        return;
    }

    btn.classList.add("state-speaking");
    icon.innerHTML = '<span class="speaking-waves"><span></span><span></span><span></span></span>';
    stateText.textContent = label || "Speaking...";
    visualizer.classList.add("hidden");
}

function toggleAutoPlay(checked) {
    CONFIG.VOICE.AUTO_PLAY = !!checked;
    localStorage.setItem("chatbot-config", JSON.stringify(CONFIG));
}

function setWakeIndicator(show, text) {
    const el = document.getElementById("wake-indicator");
    if (!el) return;
    el.textContent = text || "Listening...";
    if (show) {
        el.classList.remove("hidden");
    } else {
        el.classList.add("hidden");
    }
}

async function onVoiceButtonClick() {
    if (!CONFIG.VOICE.ENABLED) {
        setStatus("Voice mode is disabled in settings", "error");
        return;
    }

    if (voiceState === "processing") return;

    if (voiceState === "listening") {
        await stopRecordingAndTranscribe();
        return;
    }

    await startRecording("manual");
}

let speechRecognitionInstance = null;

async function startRecording(trigger = "manual") {
    if (voiceState !== "idle") return;

    recordingTrigger = trigger;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        setStatus("Speech Recognition not supported in this browser.", "error");
        return;
    }

    try {
        if (!speechRecognitionInstance) {
            speechRecognitionInstance = new SpeechRecognition();
            speechRecognitionInstance.continuous = false;
            speechRecognitionInstance.interimResults = true;
            
            speechRecognitionInstance.onresult = (event) => {
                let interimTranscript = '';
                let finalTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                const input = document.getElementById("message-input");
                if (input) {
                    input.value = finalTranscript || interimTranscript || "";
                }
            };
            
            speechRecognitionInstance.onend = () => {
                cleanupRecorder();
                setVoiceState("idle");
                const input = document.getElementById("message-input");
                if (input && input.value.trim() !== "") {
                    sendMessage();
                } else {
                    setStatus("Transcription ready - edit and send", "connected");
                }
            };
            
            speechRecognitionInstance.onerror = (event) => {
                cleanupRecorder();
                setVoiceState("idle");
                if (event.error !== 'no-speech') {
                    setStatus(`Microphone error: ${event.error}`, "error");
                }
            };
        }

        speechRecognitionInstance.start();
        setVoiceState("listening", trigger === "wake" ? "Listening... (wake word)" : "Listening...");
        
    } catch (err) {
        cleanupRecorder();
        setVoiceState("idle");
        setStatus(`Microphone error: ${err.message || err}`, "error");
    }
}

function drawWaveform() {
    // Disabled visualizer for fast native STT
}

async function stopRecordingAndTranscribe() {
    if (speechRecognitionInstance && voiceState === "listening") {
        speechRecognitionInstance.stop();
    }
}

async function transcribeAudio(audioBlob) {
    return "";
}

function cleanupRecorder() {
    if (speechRecognitionInstance) {
        try { speechRecognitionInstance.stop(); } catch (e) {}
    }
}

function updateWakePolling() {
    if (wakePollTimer) {
        clearInterval(wakePollTimer);
        wakePollTimer = null;
    }

    const shouldPoll = CONFIG.VOICE.ENABLED && CONFIG.VOICE.WAKE_ENABLED;
    setWakeIndicator(shouldPoll, shouldPoll ? "Listening..." : "");

    if (!shouldPoll) return;

    wakePollTimer = setInterval(async () => {
        if (voiceState !== "idle") return;
        try {
            const res = await fetchWithTimeout(`${CONFIG.BACKEND_URL}/wake-word/events?after_id=${wakeEventCursor}`, 5000);
            if (!res.ok) return;
            const data = await res.json();
            const events = Array.isArray(data.events) ? data.events : [];
            if (events.length === 0) return;
            wakeEventCursor = Math.max(wakeEventCursor, ...events.map(e => e.id || 0));
            await startRecording("wake");
        } catch {
            // Keep polling without noisy errors in UI
        }
    }, 1200);
}

async function syncVoiceSettingsToBackend() {
    try {
        await fetchWithTimeout(`${CONFIG.BACKEND_URL}/voice/settings`, 10000, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                enable_voice_mode: CONFIG.VOICE.ENABLED,
                enable_wake_word: CONFIG.VOICE.WAKE_ENABLED,
                wake_word_text: CONFIG.VOICE.WAKE_TEXT,
                stt_model: CONFIG.VOICE.STT_MODEL,
                tts_engine: CONFIG.VOICE.TTS_ENGINE,
                auto_play_responses: CONFIG.VOICE.AUTO_PLAY,
            }),
        });
    } catch {
        // Keep UI usable even if voice backend is unavailable.
    }
}

// Track which message is currently speaking
let currentSpeakingMessageId = null;

function stopCurrentAudio() {
    if (currentAudio) {
        if (typeof currentAudio.pause === 'function') currentAudio.pause();
        currentAudio = null;
    }
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
    document.querySelectorAll(".speaker-btn.is-active").forEach(btn => btn.classList.remove("is-active"));
    if (currentSpeakingMessageId) {
        updateTTSControls(currentSpeakingMessageId, 'idle');
        currentSpeakingMessageId = null;
    }
    if (voiceState === "speaking") setVoiceState("idle");
}

function stripMarkdownForTTS(text) {
    return String(text || "")
        .replace(/```[\s\S]*?```/g, " ")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/\*\*([^*]+)\*\*/g, "$1")
        .replace(/\*([^*]+)\*/g, "$1")
        .replace(/\[(.*?)\]\((.*?)\)/g, "$1")
        .replace(/[>#_~]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

async function playMessageAudio(messageId) {
    const msg = chatHistory.find(m => m.id === messageId);
    if (!msg) return;

    // If this message is already speaking, toggle pause/resume
    if (currentSpeakingMessageId === messageId && 'speechSynthesis' in window) {
        if (window.speechSynthesis.paused) {
            resumeTTS(messageId);
        } else if (window.speechSynthesis.speaking) {
            pauseTTS(messageId);
        }
        return;
    }

    const cleanText = stripMarkdownForTTS(msg.rawText || msg.content || "");
    if (!cleanText) return;

    stopCurrentAudio();
    currentSpeakingMessageId = messageId;
    setVoiceState("speaking", "Speaking...");
    updateTTSControls(messageId, 'playing');

    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(cleanText);

        utterance.onend = () => {
            updateTTSControls(messageId, 'idle');
            currentSpeakingMessageId = null;
            setVoiceState("idle");
        };

        utterance.onerror = () => {
            updateTTSControls(messageId, 'idle');
            currentSpeakingMessageId = null;
            setVoiceState("idle");
        };

        currentAudio = {
            pause: () => window.speechSynthesis.pause(),
            resume: () => window.speechSynthesis.resume(),
            cancel: () => window.speechSynthesis.cancel(),
        };

        window.speechSynthesis.speak(utterance);
    } else {
        updateTTSControls(messageId, 'idle');
        currentSpeakingMessageId = null;
        setVoiceState("idle");
        setStatus("Text-to-speech not supported in browser", "error");
    }
}

function pauseTTS(messageId) {
    if ('speechSynthesis' in window && window.speechSynthesis.speaking) {
        window.speechSynthesis.pause();
        updateTTSControls(messageId, 'paused');
        setVoiceState("idle", "Paused");
    }
}

function resumeTTS(messageId) {
    if ('speechSynthesis' in window && window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
        updateTTSControls(messageId, 'playing');
        setVoiceState("speaking", "Speaking...");
    }
}

function stopTTS(messageId) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
    currentAudio = null;
    updateTTSControls(messageId, 'idle');
    currentSpeakingMessageId = null;
    setVoiceState("idle");
}

function updateTTSControls(messageId, state) {
    const container = document.getElementById(`tts-controls-${messageId}`);
    if (!container) return;

    const speakBtn = container.querySelector('.tts-speak-btn');
    const pauseBtn = container.querySelector('.tts-pause-btn');
    const resumeBtn = container.querySelector('.tts-resume-btn');
    const stopBtn = container.querySelector('.tts-stop-btn');

    if (!speakBtn) return;

    if (state === 'idle') {
        speakBtn.classList.remove('hidden');
        if (pauseBtn) pauseBtn.classList.add('hidden');
        if (resumeBtn) resumeBtn.classList.add('hidden');
        if (stopBtn) stopBtn.classList.add('hidden');
        container.classList.remove('is-playing');
    } else if (state === 'playing') {
        speakBtn.classList.add('hidden');
        if (pauseBtn) pauseBtn.classList.remove('hidden');
        if (resumeBtn) resumeBtn.classList.add('hidden');
        if (stopBtn) stopBtn.classList.remove('hidden');
        container.classList.add('is-playing');
    } else if (state === 'paused') {
        speakBtn.classList.add('hidden');
        if (pauseBtn) pauseBtn.classList.add('hidden');
        if (resumeBtn) resumeBtn.classList.remove('hidden');
        if (stopBtn) stopBtn.classList.remove('hidden');
        container.classList.add('is-playing');
    }
}

async function speakText(text, buttonEl = null, messageId = null) {
    if (!CONFIG.VOICE.ENABLED) return;

    if (messageId) {
        stopCurrentAudio();
        currentSpeakingMessageId = messageId;
        setVoiceState("speaking", "Speaking...");
        updateTTSControls(messageId, 'playing');

        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);

            utterance.onend = () => {
                updateTTSControls(messageId, 'idle');
                currentSpeakingMessageId = null;
                setVoiceState("idle");
            };

            utterance.onerror = () => {
                updateTTSControls(messageId, 'idle');
                currentSpeakingMessageId = null;
                setVoiceState("idle");
            };

            currentAudio = {
                pause: () => window.speechSynthesis.pause(),
                resume: () => window.speechSynthesis.resume(),
                cancel: () => window.speechSynthesis.cancel(),
            };

            window.speechSynthesis.speak(utterance);
        } else {
            updateTTSControls(messageId, 'idle');
            currentSpeakingMessageId = null;
            setVoiceState("idle");
            setStatus("Text-to-speech not supported in browser", "error");
        }
        return;
    }

    // Fallback for calls without messageId
    stopCurrentAudio();
    setVoiceState("speaking", "Speaking...");

    if (buttonEl) buttonEl.classList.add("is-active");

    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);

        utterance.onend = () => {
            if (buttonEl) buttonEl.classList.remove("is-active");
            setVoiceState("idle");
        };

        utterance.onerror = () => {
            if (buttonEl) buttonEl.classList.remove("is-active");
            setVoiceState("idle");
        };

        currentAudio = {
            pause: () => window.speechSynthesis.pause(),
            resume: () => window.speechSynthesis.resume(),
            cancel: () => window.speechSynthesis.cancel(),
        };

        window.speechSynthesis.speak(utterance);
    } else {
        setVoiceState("idle");
        if (buttonEl) buttonEl.classList.remove("is-active");
        setStatus("Text-to-speech not supported in browser", "error");
    }
}

// ==================== Chat ====================
async function sendMessage() {
    const input = document.getElementById("message-input");
    const message = input.value.trim();
    if (!message || isWaitingForResponse) return;

    stopCurrentAudio();
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
                    temperature: CONFIG.TEMPERATURE ?? 0.1,
                }),
            }
        );

        removeLoadingIndicator(loadingId);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to get response");
        }

        const data = await response.json();
        const formattedResponse = formatResponse(data);
        const msgId = addMessageToChat(formattedResponse, "ai", data);
        setStatus("Connected", "connected");

        if (CONFIG.VOICE.AUTO_PLAY) {
            const clean = stripMarkdownForTTS(data.answer || "");
            if (clean) {
                try {
                    await speakText(clean, null, msgId);
                } catch (e) {
                    setStatus(`TTS error: ${e.message}`, "error");
                }
            }
        }
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

    let messageId = null;
    if (type === "ai") {
        aiMessageCounter += 1;
        messageId = `${Date.now()}-${aiMessageCounter}`;

        const actions = document.createElement("div");
        actions.className = "message-actions";
        actions.id = `tts-controls-${messageId}`;
        actions.innerHTML = `
            <button class="tts-speak-btn tts-ctrl-btn" onclick="playMessageAudio('${messageId}')" title="Speak">🔊 Speak</button>
            <button class="tts-pause-btn tts-ctrl-btn hidden" onclick="pauseTTS('${messageId}')" title="Pause">⏸️ Pause</button>
            <button class="tts-resume-btn tts-ctrl-btn hidden" onclick="resumeTTS('${messageId}')" title="Resume">▶️ Resume</button>
            <button class="tts-stop-btn tts-ctrl-btn hidden" onclick="stopTTS('${messageId}')" title="Stop">⏹️ Stop</button>
        `;
        bubbleDiv.appendChild(actions);
    }

    saveChatMessage(content, type, data, messageId);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageId;
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
function markdownToHtml(text) {
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");

    html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    html = html.replace(/(?:^|\n)(\d+)\.\s+(.+)/g, (_, n, item) => `\n<li>${item}</li>`);
    html = html.replace(/(?:^|\n)[-•]\s+(.+)/g, (_, item) => `\n<li>${item}</li>`);
    html = html.replace(/(<li>[\s\S]*?<\/li>(\s*<li>[\s\S]*?<\/li>)*)/g, "<ul>$1</ul>");
    html = html.replace(/\n\n+/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");

    return `<p>${html}</p>`;
}

function formatResponse(data) {
    let html = markdownToHtml(data.answer);

    if (data.retrieved_sources && data.retrieved_sources.length > 0) {
        const seen = new Set();
        const uniqueSources = [];
        data.retrieved_sources.forEach(src => {
            const name = (src.source || "").split(/[\\/]/).pop();
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
                const percent = (event.loaded / event.total) * 75;
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
                    try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {
                        // Ignore parse failures
                    }
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

    if (el("voice-enabled")) el("voice-enabled").value = CONFIG.VOICE.ENABLED ? "on" : "off";
    if (el("wake-enabled")) el("wake-enabled").value = CONFIG.VOICE.WAKE_ENABLED ? "on" : "off";
    if (el("wake-text")) el("wake-text").value = CONFIG.VOICE.WAKE_TEXT;
    if (el("stt-model")) el("stt-model").value = CONFIG.VOICE.STT_MODEL;
    if (el("tts-engine")) el("tts-engine").value = CONFIG.VOICE.TTS_ENGINE;
    if (el("voice-autoplay")) el("voice-autoplay").value = CONFIG.VOICE.AUTO_PLAY ? "on" : "off";
}

function saveSettings() {
    const el = (id) => document.getElementById(id);
    if (el("backend-url")) CONFIG.BACKEND_URL = el("backend-url").value || CONFIG.BACKEND_URL;
    if (el("model-select")) CONFIG.MODEL = el("model-select").value;
    if (el("num-retrieval")) CONFIG.NUM_RETRIEVAL = parseInt(el("num-retrieval").value);
    if (el("theme-select")) CONFIG.THEME = el("theme-select").value;
    if (el("temperature")) CONFIG.TEMPERATURE = parseFloat(el("temperature").value);

    if (el("voice-enabled")) CONFIG.VOICE.ENABLED = el("voice-enabled").value === "on";
    if (el("wake-enabled")) CONFIG.VOICE.WAKE_ENABLED = el("wake-enabled").value === "on";
    if (el("wake-text")) CONFIG.VOICE.WAKE_TEXT = el("wake-text").value || "Hey Smart";
    if (el("stt-model")) CONFIG.VOICE.STT_MODEL = el("stt-model").value;
    if (el("tts-engine")) CONFIG.VOICE.TTS_ENGINE = el("tts-engine").value;
    if (el("voice-autoplay")) CONFIG.VOICE.AUTO_PLAY = el("voice-autoplay").value === "on";

    const autoPlay = document.getElementById("autoplay-toggle");
    if (autoPlay) autoPlay.checked = !!CONFIG.VOICE.AUTO_PLAY;

    localStorage.setItem("chatbot-config", JSON.stringify(CONFIG));
    applyTheme();
    updateWakePolling();
    syncVoiceSettingsToBackend();
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
            CONFIG = {
                ...CONFIG,
                ...parsed,
                VOICE: {
                    ...CONFIG.VOICE,
                    ...(parsed.VOICE || {}),
                },
            };
            if (typeof CONFIG.BACKEND_URL === "string" && CONFIG.BACKEND_URL.includes(":8000")) {
                CONFIG.BACKEND_URL = CONFIG.BACKEND_URL.replace(":8000", ":8001");
            }
            localStorage.setItem("chatbot-config", JSON.stringify(CONFIG));
            applyTheme();
        } catch {
            // Ignore malformed local settings
        }
    }
}

function changeModel() {
    CONFIG.MODEL = document.getElementById("model-select").value;
    localStorage.setItem("chatbot-config", JSON.stringify(CONFIG));
}

// ==================== Chat History ====================
function saveChatMessage(content, type, data, messageId = null) {
    if (!currentChatId) currentChatId = Date.now().toString();
    chatHistory.push({
        id: messageId,
        chatId: currentChatId,
        timestamp: new Date().toISOString(),
        type,
        content,
        rawText: data?.answer || content,
        data,
    });
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    localStorage.setItem("currentChatId", currentChatId);
    syncChatHistoryToBackend();
    renderRecentChats();
}

async function loadChatHistory() {
    const saved = localStorage.getItem("chatHistory");
    let local = [];
    try { local = saved ? JSON.parse(saved) : []; } catch {
        local = [];
    }

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
                renderRecentChats();
                return;
            }
        }
    } catch {
        // Fallback to local cache
    }
    chatHistory = local;
    renderRecentChats();
}

function renderRecentChats() {
    const container = document.getElementById("recent-chats");
    if (!container) return;

    const grouped = {};
    chatHistory.forEach(msg => {
        if (!grouped[msg.chatId]) grouped[msg.chatId] = [];
        grouped[msg.chatId].push(msg);
    });

    const entries = Object.keys(grouped).reverse();
    if (entries.length === 0) {
        container.innerHTML = '<div class="list-empty">No recent chats</div>';
        return;
    }

    container.innerHTML = entries.map((chatId, index) => {
        const msgs = grouped[chatId];
        const firstUser = msgs.find(m => m.type === "user");
        const rawTitle = firstUser ? firstUser.content : `Chat ${entries.length - index}`;
        const safeTitle = escapeHtml(String(rawTitle).trim());
        const title = safeTitle.length > 0 ? safeTitle.substring(0, 40) : `Chat ${entries.length - index}`;
        const preview = new Date(msgs[0].timestamp).toLocaleDateString();
        const active = (chatId === currentChatId) ? "active" : "inactive";
        return `<div class="chat-item ${active}" data-chat-id="${chatId}" onclick="onSelectChat(event)">
                    <div class="chat-main">
                        <div class="chat-title" title="${safeTitle}">${title}</div>
                        <div class="chat-preview">${preview}</div>
                    </div>
                    <div class="chat-actions" onclick="event.stopPropagation()">
                        <button class="chat-more-btn" title="Chat options" onclick="toggleChatMenu(event, '${chatId}')">⋯</button>
                        <div class="chat-menu" id="chat-menu-${chatId}">
                            <button class="chat-menu-item" onclick="shareChatSession(event, '${chatId}')">Share</button>
                            <button class="chat-menu-item" onclick="deleteChatSessionById(event, '${chatId}')">Delete</button>
                            <button class="chat-menu-item" onclick="showChatHistory()">More options</button>
                        </div>
                    </div>
                </div>`;
    }).join("");
}

function onSelectChat(event) {
    const el = event.currentTarget || event.target.closest(".chat-item");
    if (!el) return;
    const chatId = el.getAttribute("data-chat-id");
    if (!chatId) return;
    loadChatSession(chatId);
    document.querySelectorAll(".chat-item").forEach(i => i.classList.remove("active"));
    el.classList.add("active");
}

function toggleChatMenu(event, chatId) {
    event.stopPropagation();
    const menu = document.getElementById(`chat-menu-${chatId}`);
    if (!menu) return;
    const isOpen = menu.classList.contains("open");
    closeAllChatMenus();
    if (!isOpen) menu.classList.add("open");
}

function closeAllChatMenus() {
    document.querySelectorAll(".chat-menu.open").forEach(menu => menu.classList.remove("open"));
}

function deleteChatSessionById(event, chatId) {
    event.stopPropagation();
    deleteChatSession(chatId);
    closeAllChatMenus();
}

async function shareChatSession(event, chatId) {
    event.stopPropagation();
    const messages = chatHistory.filter(m => m.chatId === chatId);
    const firstUser = messages.find(m => m.type === "user");
    const heading = firstUser?.content || "Shared chat";
    const content = messages
        .map(m => `${m.type.toUpperCase()}: ${String(m.rawText || m.content).replace(/<[^>]*>/g, "")}`)
        .join("\n\n");
    const text = `${heading}\n\n${content}`;
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            const temp = document.createElement("textarea");
            temp.value = text;
            document.body.appendChild(temp);
            temp.select();
            document.execCommand("copy");
            temp.remove();
        }
        setStatus("Chat copied to clipboard", "connected");
    } catch {
        setStatus("Unable to copy chat", "error");
    }
    closeAllChatMenus();
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function syncChatHistoryToBackend() {
    fetch(`${CONFIG.BACKEND_URL}/chat-history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: chatHistory }),
    }).catch(() => {});
}

document.addEventListener("click", () => {
    closeAllChatMenus();
});

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
        const preview = first ? String(first.content).substring(0, 60) : "(No messages)";
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
        btnEl.closest(".history-item-row").remove();
    }

    if (currentChatId === chatId) {
        newChat();
    }
    renderRecentChats();
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
    renderRecentChats();
}

function loadChatSession(chatId) {
    const chatContainer = document.getElementById("chat-container");
    chatContainer.innerHTML = "";
    chatHistory.filter(m => m.chatId === chatId).forEach(msg => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${msg.type}`;
        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        if (msg.type === "ai" || msg.type === "ai-error") {
            bubble.innerHTML = msg.content;
            if (msg.type === "ai" && msg.id) {
                const actions = document.createElement("div");
                actions.className = "message-actions";
                actions.id = `tts-controls-${msg.id}`;
                actions.innerHTML = `
                    <button class="tts-speak-btn tts-ctrl-btn" onclick="playMessageAudio('${msg.id}')" title="Speak">🔊 Speak</button>
                    <button class="tts-pause-btn tts-ctrl-btn hidden" onclick="pauseTTS('${msg.id}')" title="Pause">⏸️ Pause</button>
                    <button class="tts-resume-btn tts-ctrl-btn hidden" onclick="resumeTTS('${msg.id}')" title="Resume">▶️ Resume</button>
                    <button class="tts-stop-btn tts-ctrl-btn hidden" onclick="stopTTS('${msg.id}')" title="Stop">⏹️ Stop</button>
                `;
                bubble.appendChild(actions);
            }
        } else {
            bubble.textContent = msg.content;
        }
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
                    <li>Use voice button to dictate; edit transcript before sending</li>
                </ul>
            </div>
        </div>`;
    document.getElementById("message-input").focus();
    renderRecentChats();
}
