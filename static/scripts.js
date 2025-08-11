
document.addEventListener('DOMContentLoaded', () => {
    if (typeof USER_ID === 'undefined') return;

    // --- Page Elements ---
    const avatarSelectionGrid = document.getElementById('avatar-selection-grid');
    const step1Div = document.getElementById('step1-select-avatar');
    const step2Div = document.getElementById('step2-interaction');
    const statusBar = document.getElementById('statusBar');
    const transcriptionDisplay = document.getElementById('transcription-display');
    const llmResponseDisplay = document.getElementById('llm-response-display');
    const avatarVideo = document.getElementById('avatar-video');
    const changeAvatarBtn = document.getElementById('changeAvatarBtn');
    const sidebarLinks = document.querySelectorAll('.sidebar a');

    // --- State ---
    let ws;
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let selectedAvatarId = null;
    let videoQueue = [];
    let isPlaying = false;

    // --- WebSocket Setup ---
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        ws = new WebSocket(`${protocol}://${window.location.host}/ws/${USER_ID}`);

        ws.onopen = () => {
            statusBar.textContent = 'Connection established. Please select an avatar.';
            statusBar.className = 'status-bar info';
        };
        ws.onmessage = e => handleWebSocketMessage(JSON.parse(e.data));
        ws.onclose = () => {
            statusBar.textContent = 'Connection lost. Please refresh.';
            statusBar.className = 'status-bar error';
            isRecording = false;
            if (mediaRecorder?.state === 'recording') mediaRecorder.stop();
        };
        ws.onerror = e => {
            statusBar.textContent = 'Connection error.';
            statusBar.className = 'status-bar error';
        };
    }

    function handleWebSocketMessage(data) {
        switch (data.type) {
            case 'avatar_selected':
                data.success ? showInteractionScreen() : alert('Error selecting avatar.');
                break;
            case 'transcription':
                transcriptionDisplay.innerHTML = `<span class="user">You:</span> ${data.text || '...'}`;
                statusBar.textContent = data.text ? 'Thinking...' : 'Could not understand audio.';
                statusBar.className = data.text ? 'status-bar processing' : 'status-bar info';
                break;
            case 'llm_chunk':
                if (llmResponseDisplay.innerHTML.includes('Assistant:')) llmResponseDisplay.innerHTML += data.text;
                else llmResponseDisplay.innerHTML = `<span class="assistant">Assistant:</span> ${data.text}`;
                break;
            case 'llm_end':
                statusBar.textContent = 'Hold SPACE to speak';
                statusBar.className = 'status-bar info';
                break;
            case 'video_chunk':
                if (data.url) {
                    videoQueue.push(data.url);
                    if (!isPlaying) playNextVideo();
                }
                break;
            case 'error':
                statusBar.textContent = `Error: ${data.message}`;
                statusBar.className = 'status-bar error';
                break;
        }
    }

    function sendMessage(data) {
        if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
    }

    // --- Audio ---
    async function setupAudioRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                if (ws?.readyState === WebSocket.OPEN) ws.send(audioBlob);
                audioChunks = [];
            };
        } catch (err) {
            statusBar.textContent = 'Microphone access denied.';
            alert('This app requires microphone access.');
        }
    }

    function startRecording() {
        if (mediaRecorder && !isRecording) {
            mediaRecorder.start();
            isRecording = true;
            statusBar.textContent = 'Listening...';
            statusBar.className = 'status-bar listening';
            transcriptionDisplay.textContent = '';
            llmResponseDisplay.textContent = '';
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            statusBar.textContent = 'Processing...';
            statusBar.className = 'status-bar processing';
        }
    }

    // --- Dual Video Playback (buffered + transition) ---
    const videoA = avatarVideo;
    const videoB = avatarVideo.cloneNode();
    videoB.id = 'buffer-video';
    Object.assign(videoB.style, {
        position: 'absolute', top: '0', left: '0', width: '100%', height: '100%', transition: 'opacity 0.2s ease-in-out', opacity: 0, zIndex: 0
    });
    videoA.style.transition = 'opacity 0.2s ease-in-out';
    videoA.style.zIndex = 1;
    // videoA.style.objectFit = 'contain';
    // videoA.style.maxHeight = '800px';
    avatarVideo.parentNode.style.position = 'relative';
    avatarVideo.parentNode.appendChild(videoB);

    let currentVideo = videoA;
    let bufferVideo = videoB;

    function playNextVideo() {
        if (!videoQueue.length) {
            const fallback = document.querySelector('.avatar-card-select.selected')?.dataset.videoUrl;
            if (fallback) switchToVideo(fallback, true);
            isPlaying = false;
            return;
        }
        isPlaying = true;
        const nextUrl = videoQueue.shift();
        switchToVideo(nextUrl, false);
    }

    function switchToVideo(url, loop) {
        bufferVideo.src = url;
        bufferVideo.loop = loop;
        bufferVideo.oncanplay = () => {
            bufferVideo.play().then(() => {
                currentVideo.style.opacity = 0;
                bufferVideo.style.opacity = 1;
                setTimeout(() => {
                    currentVideo.pause();
                    currentVideo.src = '';
                    [currentVideo, bufferVideo] = [bufferVideo, currentVideo];
                    bufferVideo.style.opacity = 0;
                    bufferVideo.onended = playNextVideo;
                }, 200);
            }).catch(console.error);
        };
        bufferVideo.load();
    }

    videoA.onended = playNextVideo;
    videoB.onended = playNextVideo;

    // --- UI ---
    async function loadAvatars() {
        try {
            const res = await fetch('/api/avatars');
            const avatars = await res.json();
            avatarSelectionGrid.innerHTML = avatars.length ? '' : '<p>No avatars found.</p>';
            avatars.forEach(avatar => {
                const card = document.createElement('div');
                card.className = 'avatar-card-select';
                card.dataset.avatarId = avatar.id;
                card.dataset.videoUrl = avatar.video_url;
                card.innerHTML = `<img src="${avatar.image_url}" alt="${avatar.id}"><p>${avatar.id}</p>`;
                card.addEventListener('click', () => selectAvatar(avatar.id, card));
                avatarSelectionGrid.appendChild(card);
            });
        } catch (err) {
            avatarSelectionGrid.innerHTML = '<p>Error loading avatars.</p>';
        }
    }

    function selectAvatar(id, card) {
        selectedAvatarId = id;
        document.querySelectorAll('.avatar-card-select').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        switchToVideo(card.dataset.videoUrl, true);
        sendMessage({ type: 'select_avatar', avatar_id: id });
    }

    function showInteractionScreen() {
        step1Div.style.display = 'none';
        step2Div.style.display = 'block';
        statusBar.textContent = 'Hold SPACE to speak';
        statusBar.className = 'status-bar info';
    }

    function showAvatarSelectionScreen() {
        step1Div.style.display = 'block';
        step2Div.style.display = 'none';
        statusBar.textContent = 'Please select an avatar.';
        statusBar.className = 'status-bar info';
        selectedAvatarId = null;
    }

    changeAvatarBtn.addEventListener('click', showAvatarSelectionScreen);

    window.addEventListener('keydown', e => {
        if (e.code === 'Space' && !isRecording && selectedAvatarId) {
            e.preventDefault();
            startRecording();
        }
    });
    window.addEventListener('keyup', e => {
        if (e.code === 'Space' && isRecording) {
            e.preventDefault();
            stopRecording();
        }
    });

    sidebarLinks.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const pageId = e.target.dataset.page;
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
            sidebarLinks.forEach(l => l.classList.remove('active'));
            e.target.classList.add('active');
        });
    });

    function init() {
        loadAvatars();
        connectWebSocket();
        setupAudioRecording();
    }
    init();
});
