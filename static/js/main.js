const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const metadataCard = document.getElementById("metadata-card");
const summaryBtn = document.getElementById("summary-btn");
const chatWindow = document.getElementById("chat-window");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const summaryContent = document.getElementById("summary-content");

let currentPaperId = null;

// --- Upload handling ---

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) uploadFile(file);
});

async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showStatus("Only PDF files are supported.", true);
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        showStatus("PDF exceeds the 20MB limit.", true);
        return;
    }

    showStatus("Processing PDF... this can take a minute for image-heavy papers.");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            showStatus(data.error || "Upload failed.", true);
            return;
        }

        currentPaperId = data.paper_id;
        showStatus("");
        showMetadata(data);
        enableChat();
        summaryBtn.disabled = false;
    } catch (err) {
        showStatus("Network error during upload.", true);
    }
}

function showStatus(message, isError = false) {
    uploadStatus.textContent = message;
    uploadStatus.classList.toggle("error", isError);
}

function showMetadata(data) {
    document.getElementById("meta-chunks").textContent = data.text_chunks;
    document.getElementById("meta-images").textContent = data.images;
    document.getElementById("meta-tables").textContent = data.tables;
    metadataCard.classList.remove("hidden");
}

function enableChat() {
    questionInput.disabled = false;
    sendBtn.disabled = false;
}

// --- Chat handling ---

sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuestion();
});

async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    if (!currentPaperId) {
        addBubble("Please upload a paper first.", "error");
        return;
    }

    addBubble(question, "user");
    questionInput.value = "";

    const spinner = document.createElement("div");
    spinner.className = "loading-spinner";
    spinner.textContent = "Thinking...";
    chatWindow.appendChild(spinner);
    scrollToBottom();

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, paper_id: currentPaperId }),
        });
        const data = await response.json();

        spinner.remove();

        if (!response.ok) {
            addBubble(data.error || "Something went wrong.", "error");
            return;
        }

        addBubble(data.final_answer, "ai", data.source_type);
    } catch (err) {
        spinner.remove();
        addBubble("Network error - the agent may have timed out.", "error");
    }
}

function addBubble(text, kind, sourceType = null) {
    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${kind}`;
    bubble.textContent = text;

    if (sourceType) {
        const tag = document.createElement("span");
        tag.className = "citation-tag";
        tag.textContent = sourceType.replace("_query", "");
        bubble.appendChild(document.createElement("br"));
        bubble.appendChild(tag);
    }

    chatWindow.appendChild(bubble);
    scrollToBottom();
}

function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// --- Summary handling ---

summaryBtn.addEventListener("click", async () => {
    if (!currentPaperId) return;

    summaryContent.innerHTML = '<p class="placeholder-text">Generating summary...</p>';

    try {
        const response = await fetch(`/summary?paper_id=${currentPaperId}`);
        const data = await response.json();

        if (!response.ok) {
            summaryContent.innerHTML = `<p class="placeholder-text">${data.error}</p>`;
            return;
        }

        summaryContent.textContent = data.summary;
    } catch (err) {
        summaryContent.innerHTML = '<p class="placeholder-text">Failed to load summary.</p>';
    }
});