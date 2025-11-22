const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Handle WebSocket connections
io.on('connection', (socket) => {
    console.log('User connected:', socket.id);

    // Handle voice data from client
    socket.on('voiceData', (data) => {
        // In a real implementation, we would send this to Murf API
        // For now, we'll simulate a response
        console.log('Received voice data from client');

        // Simulate processing delay
        setTimeout(() => {
            // Send back a response (in a real app, this would be actual TTS audio)
            socket.emit('agentResponse', {
                text: "Hello! I'm your Murf AI voice agent. How can I assist you today?",
                timestamp: new Date()
            });
        }, 1000);
    });

    socket.on('disconnect', () => {
        console.log('User disconnected:', socket.id);
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});