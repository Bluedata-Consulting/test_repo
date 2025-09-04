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
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const scheduleManagement = document.getElementById('schedule-management');
    const scheduleList = document.getElementById('schedule-list');

    // --- State ---
    let ws;
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let selectedAvatarId = null;
    let videoQueue = [];
    let isPlaying = false;
    let webcamStream = null;

    let savedSchedule = {
        time: '',
        purpose: '',
        enabled: false
    };

    // --- Helper Function: Convert Data URI to Blob ---
    function dataURItoBlob(dataURI) {
        let byteString;
        if (dataURI.split(',')[0].indexOf('base64') >= 0) {
            byteString = atob(dataURI.split(',')[1]);
        } else {
            byteString = unescape(dataURI.split(',')[1]);
        }
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ia = new Uint8Array(byteString.length);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ia], { type: mimeString });
    }

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
        ws.onerror = () => {
            statusBar.textContent = 'Connection error.';
            statusBar.className = 'status-bar error';
        };
    }

    function handleWebSocketMessage(data) {
        switch (data.type) {
            case 'avatar_selected':
                if (data.success) {
                    // Update selected avatar ID
                    selectedAvatarId = data.avatar.id;
                    showInteractionScreen();
                } else {
                    alert('Error selecting avatar.');
                }
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
            case 'schedule_set':
                if (data.success) {
                    savedSchedule.enabled = true;
                    updateScheduleUI();
                    statusBar.textContent = 'Schedule has been set! An avatar has been automatically selected.';
                    statusBar.className = 'status-bar success';
                    showNotification('Schedule has been set! An avatar has been automatically selected.', 'success');
                    updateScheduleList(); // Update the schedule list when schedule is set
                }
                break;
            case 'schedule_triggered':
                // Show notification when schedule is triggered
                showNotification(`⏰ Schedule triggered: ${data.message}`, 'info');
                statusBar.textContent = data.message;
                statusBar.className = 'status-bar success';
                break;
            case 'error':
                statusBar.textContent = `Error: ${data.message}`;
                statusBar.className = 'status-bar error';
                showNotification(`❌ Error: ${data.message}`, 'error');
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
        } catch (err) {
            statusBar.textContent = 'Microphone access denied.';
            showNotification('🎙️ Microphone access denied. This app requires microphone access.', 'error');
        }
    }

    function startRecording() {
        if (mediaRecorder && !isRecording) {
            const visionToggle = document.getElementById('vision-toggle');
            if (visionToggle.checked && webcamStream) {
                const imageSrc = captureImage();
                if (imageSrc) {
                    const imageData = dataURItoBlob(imageSrc);
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        sendAudioWithImage(audioBlob, imageData);
                        audioChunks = [];
                    };
                }
            } else {
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    if (ws?.readyState === WebSocket.OPEN) ws.send(audioBlob);
                    audioChunks = [];
                };
            }
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

    // --- Vision Mode Functions ---
    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } });
            webcamVideo.srcObject = webcamStream;
            statusBar.textContent = 'Camera started. Hold SPACE to speak with vision.';
            statusBar.className = 'status-bar info';
        } catch (err) {
            statusBar.textContent = 'Camera access denied.';
            statusBar.className = 'status-bar error';
            document.getElementById('vision-toggle').checked = false;
            showNotification('📷 Camera access denied.', 'error');
        }
    }

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
        statusBar.textContent = 'Camera stopped.';
        statusBar.className = 'status-bar info';
    }

    function captureImage() {
        if (!webcamStream) return null;
        const context = webcamCanvas.getContext('2d');
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        context.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
        return webcamCanvas.toDataURL('image/jpeg');
    }

    function sendAudioWithImage(audioBlob, imageData) {
        if (ws?.readyState === WebSocket.OPEN) {
            audioBlob.arrayBuffer().then(audioBuffer => {
                const audioBytes = new Uint8Array(audioBuffer);
                if (imageData) {
                    imageData.arrayBuffer().then(imageBuffer => {
                        const imageBytes = new Uint8Array(imageBuffer);
                        const combinedBuffer = new Uint8Array(4 + audioBytes.length + imageBytes.length);
                        const audioLenView = new DataView(combinedBuffer.buffer);
                        audioLenView.setUint32(0, audioBytes.length, false);
                        combinedBuffer.set(audioBytes, 4);
                        combinedBuffer.set(imageBytes, 4 + audioBytes.length);
                        ws.send(combinedBuffer);
                    });
                }
            });
        }
    }

    // --- Dual Video Playback ---
    const videoA = avatarVideo;
    const videoB = avatarVideo.cloneNode();
    videoB.id = 'buffer-video';
    Object.assign(videoB.style, {
        position: 'absolute', top: '0', left: '0', width: '100%', height: '100%', transition: 'opacity 0.2s ease-in-out', opacity: 0, zIndex: 0
    });
    videoA.style.transition = 'opacity 0.2s ease-in-out';
    videoA.style.zIndex = 1;
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
            showNotification('Error loading avatars.', 'error');
        }
    }

    function selectAvatar(id, card) {
        selectedAvatarId = id;
        document.querySelectorAll('.avatar-card-select').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        switchToVideo(card.dataset.videoUrl, true);
        sendMessage({ type: 'select_avatar', avatar_id: id });
        
        // If a schedule has been set, update the UI to reflect that it's ready to be armed
        if (savedSchedule.time && savedSchedule.purpose) {
            statusBar.textContent = `Avatar selected. You can now arm the schedule for ${savedSchedule.time}.`;
            statusBar.className = 'status-bar info';
        }
    }

    function showInteractionScreen() {
        step1Div.style.display = 'none';
        step2Div.style.display = 'block';
        scheduleManagement.style.display = 'block';
        statusBar.textContent = 'Hold SPACE to speak';
        statusBar.className = 'status-bar info';
        updateScheduleList();
    }

    function showAvatarSelectionScreen() {
        step1Div.style.display = 'block';
        step2Div.style.display = 'none';
        scheduleManagement.style.display = 'none';
        statusBar.textContent = 'Please select an avatar.';
        statusBar.className = 'status-bar info';
        selectedAvatarId = null;
    }

    // --- Schedule Functions ---
    function updateScheduleUI() {
        const scheduleToggle = document.getElementById('schedule-toggle');
        if (savedSchedule.enabled) {
            scheduleToggle.checked = true;
            statusBar.textContent = `Schedule armed for ${savedSchedule.time}.`;
            statusBar.className = 'status-bar success';
        } else {
            scheduleToggle.checked = false;
            statusBar.textContent = 'Schedule disarmed.';
            statusBar.className = 'status-bar info';
        }
    }

    function formatTimeForDisplay() {
        const hour = document.getElementById('schedule-hour').value.padStart(2, '0');
        const minute = document.getElementById('schedule-minute').value;
        const ampm = document.getElementById('schedule-ampm').value;
        return `${hour}:${minute} ${ampm}`;
    }

    function updateScheduleList() {
        if (!scheduleList) return;
        
        if (savedSchedule.time && savedSchedule.purpose) {
            scheduleList.innerHTML = `
                <div class="schedule-item">
                    <div class="schedule-info">
                        <strong>${savedSchedule.time}</strong>
                        <span>${savedSchedule.purpose}</span>
                    </div>
                    <div class="schedule-status ${savedSchedule.enabled ? 'enabled' : 'disabled'}">
                        ${savedSchedule.enabled ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
            `;
        } else {
            scheduleList.innerHTML = '<p>No schedules set yet.</p>';
        }
    }

    // --- Notification System ---
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span class="notification-message">${message}</span>
            <button class="notification-close">&times;</button>
        `;
        
        // Add to document
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
        
        // Close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }

    // --- Event Listeners ---
    changeAvatarBtn.addEventListener('click', showAvatarSelectionScreen);

    const visionToggle = document.getElementById('vision-toggle');
    const scheduleToggle = document.getElementById('schedule-toggle');
    const webcamContainer = document.getElementById('webcam-container');
    
    // Schedule inputs
    const scheduleHour = document.getElementById('schedule-hour');
    const scheduleMinute = document.getElementById('schedule-minute');
    const scheduleAmpm = document.getElementById('schedule-ampm');
    const schedulePurpose = document.getElementById('main-schedule-purpose');

    // Save schedule when inputs change
    [scheduleHour, scheduleMinute, scheduleAmpm, schedulePurpose].forEach(input => {
        input.addEventListener('change', () => {
            savedSchedule.time = formatTimeForDisplay();
            savedSchedule.purpose = schedulePurpose.value;
            updateScheduleList();
        });
    });

    visionToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            webcamContainer.style.display = 'block';
            startWebcam();
            sendMessage({ type: 'enable_vision' });
        } else {
            stopWebcam();
            webcamContainer.style.display = 'none';
            sendMessage({ type: 'disable_vision' });
        }
    });

    scheduleToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            // Validate schedule data
            if (!savedSchedule.time || !savedSchedule.purpose) {
                alert('Please set a time and purpose first.');
                e.target.checked = false;
                return;
            }
            
            // Send schedule to backend - avatar will be automatically selected if needed
            sendMessage({
                type: 'schedule',
                time: savedSchedule.time,
                purpose: savedSchedule.purpose
            });
            
            // Show confirmation
            showNotification(`⏰ Schedule set for ${savedSchedule.time}: ${savedSchedule.purpose}`, 'success');
        } else {
            // Disable schedule
            savedSchedule.enabled = false;
            statusBar.textContent = 'Schedule disarmed.';
            statusBar.className = 'status-bar info';
            showNotification('Schedule disarmed.', 'info');
        }
        updateScheduleList();
    });

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

    function init() {
        loadAvatars();
        connectWebSocket();
        setupAudioRecording();
    }
    init();
});
