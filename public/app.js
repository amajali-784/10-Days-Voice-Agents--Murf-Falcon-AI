// Frontend JavaScript for Murf AI Voice Agent
let socket;
let mediaRecorder;
let audioChunks = [];

// DOM Elements
const connectBtn = document.getElementById('connect-btn');
const connectionStatus = document.getElementById('connection-status');
const messagesContainer = document.getElementById('messages');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const startRecordingBtn = document.getElementById('start-recording');
const stopRecordingBtn = document.getElementById('stop-recording');

// Connect to the server
connectBtn.addEventListener('click', () => {
    if (socket) {
        socket.disconnect();
        connectBtn.textContent = 'Connect to Voice Agent';
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.className = 'disconnected';
        return;
    }

    socket = io();

    socket.on('connect', () => {
        connectionStatus.textContent = 'Connected';
        connectionStatus.className = 'connected';
        connectBtn.textContent = 'Disconnect';
        addMessage('System', 'Connected to voice agent', new Date());
    });

    socket.on('disconnect', () => {
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.className = 'disconnected';
        connectBtn.textContent = 'Connect to Voice Agent';
        addMessage('System', 'Disconnected from voice agent', new Date());
    });

    // Handle responses from the agent
    socket.on('agentResponse', (data) => {
        addMessage('Agent', data.text, data.timestamp);
    });
});

// Send text message
sendBtn.addEventListener('click', sendMessage);
textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const text = textInput.value.trim();
    if (text && socket) {
        socket.emit('voiceData', { text });
        addMessage('You', text, new Date());
        textInput.value = '';
    }
}

// Voice recording functionality
startRecordingBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            // In a real implementation, we would send this to the server
            // For now, we'll just simulate sending text
            const simulatedText = "Hello, this is a voice message";
            if (socket) {
                socket.emit('voiceData', { text: simulatedText });
                addMessage('You (voice)', simulatedText, new Date());
            }

            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
        addMessage('System', 'Recording started...', new Date());
    } catch (error) {
        console.error('Error accessing microphone:', error);
        addMessage('System', 'Error accessing microphone: ' + error.message, new Date());
    }
});

stopRecordingBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        startRecordingBtn.disabled = false;
        stopRecordingBtn.disabled = true;
        addMessage('System', 'Recording stopped', new Date());
    }
});

// Helper function to add messages to the UI
function addMessage(sender, text, timestamp) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender === 'You' || sender === 'You (voice)' ? 'user-message' : sender === 'Agent' ? 'agent-message' : ''}`;

    const senderSpan = document.createElement('strong');
    senderSpan.textContent = sender;

    const textDiv = document.createElement('div');
    textDiv.textContent = text;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'timestamp';
    timeDiv.textContent = timestamp.toLocaleTimeString();

    messageDiv.appendChild(senderSpan);
    messageDiv.appendChild(document.createElement('br'));
    messageDiv.appendChild(textDiv);
    messageDiv.appendChild(timeDiv);

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}