/// <reference types="vite/client" />

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid';
import Scheduler from './Scheduler'

const BACKEND_BASE = (import.meta.env.VITE_BACKEND_BASE as string) || 'http://127.0.0.1:8600'
const STOP_WORDS = ["stop", "stop listening", "stop recording", "please stop", "that's enough"];

type Avatar = {
  id: string
  name: string
  base_id: string
  persona: string
  language: string
  idle_video_url?: string
  lip_video_url?: string
}

export default function App() {
  const [avatars, setAvatars] = useState<Avatar[]>([])
  const [selected, setSelected] = useState<string>('')
  const [wakeWord, setWakeWord] = useState<string>('avatar')
  const [listening, setListening] = useState<boolean>(false)
  const [phase, setPhase] = useState<'idle' | 'listening' | 'speaking'>('idle')

  // Refs for media and state management
  const detectStreamRef = useRef<MediaStream | null>(null)
  const detectRecRef = useRef<MediaRecorder | null>(null)
  const detectCycleRef = useRef<boolean>(false)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const vadTimerRef = useRef<number | null>(null)
  const speakingRef = useRef<boolean>(false)
  const sessionId = useRef<string>(
    `session-${Date.now()}`
  );
  const clientId = useRef<string>(uuidv4());

  // Refs for DOM elements
  const logRef = useRef<HTMLDivElement>(null)
  const idleRef = useRef<HTMLVideoElement>(null)
  const lipRef = useRef<HTMLVideoElement>(null)
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)

  const log = (m: string) => {
    if (!logRef.current) return;
    const t = new Date().toLocaleTimeString()
    logRef.current.textContent += `[${t}] ${m}\n`
    logRef.current.scrollTop = logRef.current.scrollHeight
  }

  const playAvatarResponse = async (audioUrl: string, responseText: string) => {
    log(`LLM: ${responseText}`)

    if (lipRef.current) {
      lipRef.current.currentTime = 0
      lipRef.current.style.opacity = '1'
      await lipRef.current.play()
    }

    setPhase('speaking')
    const audio = new Audio(`${BACKEND_BASE}${audioUrl}`)
    currentAudioRef.current = audio
    await audio.play()
    await new Promise(res => { 
      audio.onended = () => {
        currentAudioRef.current = null
        res(null) 
      }
    })

    if (lipRef.current) {
      lipRef.current.pause()
      lipRef.current.style.opacity = '0'
    }
    if (idleRef.current) {
      idleRef.current.play().catch(() => {}) 
    }
    setPhase(listening ? 'listening' : 'idle')
  }

  // Cleanly stops all detection resources
  const stopAllActivity = useCallback((propagateToBackend = true) => {
    log('Stopping all activity.');
    detectCycleRef.current = false;

    // Stop wake word detection loop
    if (vadTimerRef.current) {
      clearInterval(vadTimerRef.current);
      vadTimerRef.current = null;
    }
    if (detectRecRef.current && detectRecRef.current.state !== 'inactive') {
      detectRecRef.current.stop();
    }
    detectRecRef.current = null;
    detectStreamRef.current?.getTracks().forEach(track => track.stop());
    detectStreamRef.current = null;

    // Stop any utterance capture in progress
    speakingRef.current = false;

    // Stop audio playback
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }

    // Stop lip-sync video
    if (lipRef.current) {
      lipRef.current.pause();
      lipRef.current.style.opacity = '0';
    }

    if (propagateToBackend) {
      // Signal backend to stop all activity for this client
      fetch(`${BACKEND_BASE}/api/vad_stopped`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'manual_stop',
          reason: 'User clicked stop button',
          timestamp: new Date().toISOString(),
          sessionId: clientId.current,
        }),
      }).then(res => {
        if (res.ok) {
          log('Backend stop signal sent.');
        } else {
          log('Failed to send backend stop signal.');
        }
      }).catch(e => log(`Error sending stop signal: ${String(e)}`));
    }

    setPhase('idle');
    setListening(false);
  }, []);

  // Fetch initial config and avatars on component mount
  useEffect(() => {
    (async () => {
      try {
        setPhase('idle')
        // wait for backend readiness (whisper warmup)
        let ready = false
        for (let i = 0; i < 60; i++) {
          try {
            const d = await fetch(`${BACKEND_BASE}/api/ready`).then(r=>r.json())
            if (d?.ready) { ready = true; break }
          } catch {} 
          await new Promise(res => setTimeout(res, 1000))
        }
        log(ready ? 'Models ready.' : 'Model readiness timed out.')

        const cfg = await fetch(`${BACKEND_BASE}/api/config`).then(r => r.json())
        if (cfg?.wake_word) setWakeWord(cfg.wake_word)

        const avs = await fetch(`${BACKEND_BASE}/api/avatars`).then(r => r.json())
        const avatarsWithVideos = avs.map((a: Avatar) => ({
          ...a,
          idle_video_url: `/media/${a.base_id}.mp4`,
          lip_video_url: `/media/${a.base_id}_lip.mp4`,
        }));
        setAvatars(avatarsWithVideos)
        if (avatarsWithVideos[0]) setSelected(avatarsWithVideos[0].id)
      } catch (e) {
        log(String(e))
      }
    })()

    // SSE connection
    const eventSource = new EventSource(`${BACKEND_BASE}/api/events/${clientId.current}`);
    
    eventSource.onopen = () => {
        log("SSE connection established.");
    };

    eventSource.onmessage = (event) => {
        // The backend sends a keep-alive message as a comment, which we can ignore.
        if (event.data.startsWith(':')) {
            return;
        }

        const data = JSON.parse(event.data);
        if (data.type === 'scheduled_event') {
            log("Received scheduled event");
            playAvatarResponse(data.audio_url, data.response);
        } else if (data.type === 'stop_activity') {
            log("Received stop activity event. Halting all activity locally.");
            stopAllActivity(false); // Pass false to prevent feedback loop
        }
    };

    eventSource.onerror = (err) => {
        log("SSE connection error.");
        // The browser will automatically try to reconnect. No need to close the event source here.
    };

    return () => {
        eventSource.close();
    };

  }, [stopAllActivity])

  // Update video sources when selected avatar changes
  useEffect(() => {
    const a = avatars.find(v => v.id === selected)
    if (!a || !idleRef.current || !lipRef.current) return
    idleRef.current.src = `${BACKEND_BASE}${a.idle_video_url}`
    lipRef.current.src = `${BACKEND_BASE}${a.lip_video_url}`
    idleRef.current.play().catch(() => {}) 
  }, [avatars, selected])

  // Central function to send audio and handle avatar response
  const sendToBackend = async (blob: Blob) => {
    const fd = new FormData()
    fd.append('audio', blob, 'speech.webm')
    fd.append('avatar_id', selected)
    fd.append('client_id', clientId.current)
    try {
      const r = await fetch(`${BACKEND_BASE}/api/interact`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(`Server responded with ${r.status}`);
      const d = await r.json()
      log(`Transcription: ${d.transcription}`)

      // VAD Stop Word Check
      const transcriptionLower = (d.transcription || "").trim().toLowerCase();
      if (STOP_WORDS.some(word => transcriptionLower.includes(word))) {
        log("Stop word detected. Halting capture.");
        stopAllActivity();
        log("Listening stopped.");
        // Send vad:stopped event to backend
        fetch(`${BACKEND_BASE}/api/vad_stopped`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: 'vad:stopped',
            reason: 'stop_word',
            timestamp: new Date().toISOString(),
            sessionId: sessionId.current,
          }),
        });
        return;
      }

      await playAvatarResponse(d.audio_url, d.response);

    } catch (e) {
      log(`Error during interaction: ${String(e)}`)
    } finally {
      setPhase(listening ? 'listening' : 'idle')
    }
  }

  // Hold-to-talk functionality
  const holdToTalk = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const rec = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      const chunks: BlobPart[] = []
      rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data) }
      rec.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        if (blob.size > 100) await sendToBackend(blob);
        stream.getTracks().forEach(t => t.stop())
      }
      rec.start()
      setListening(true)
      setPhase('listening')
      const stop = () => {
        rec.stop()
        setListening(false)
        setPhase('idle')
        window.removeEventListener('mouseup', stop)
      }
      window.addEventListener('mouseup', stop)
    } catch(e) {
      log(`Could not start recording: ${String(e)}`);
    }
  }

  // After wake word, capture the user's command
  const captureUtterance = useCallback(async () => {
    log('Wake word detected! Capturing utterance...');
    setPhase('listening')
    // Grace delay so the user sees the green indicator before we start
    await new Promise(res => setTimeout(res, 280))
    const stream = detectStreamRef.current
    if (!stream) {
      log('Error: Detection stream not available for utterance capture.')
      return
    }

    const rec = new MediaRecorder(stream, { mimeType: 'audio/webm' })
    const chunks: BlobPart[] = []
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data) }
    const finished = new Promise<void>(res => { rec.onstop = () => res() })
    rec.start()

    // VAD Setup
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext()
    const ctx = audioCtxRef.current
    // Check if context is running, resume if suspended
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    const src = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser)
    const data = new Uint8Array(analyser.frequencyBinCount)
    let silenceFrames = 0
    speakingRef.current = true

    const vadLoop = () => {
      if (!speakingRef.current) return
      analyser.getByteTimeDomainData(data)
      let sum = 0
      for (const v of data) { sum += (v - 128) * (v - 128) }
      const rms = Math.sqrt(sum / data.length)
      if (rms < 6) silenceFrames++; else silenceFrames = 0;
      if (silenceFrames > 25) {
        if (rec.state === 'recording') rec.stop()
        speakingRef.current = false
        if (vadTimerRef.current) {
          clearInterval(vadTimerRef.current)
          vadTimerRef.current = null
        }
      }
    }
    vadTimerRef.current = window.setInterval(vadLoop, 24)

    setTimeout(() => { if (speakingRef.current && rec.state === 'recording') { rec.stop(); speakingRef.current = false; } }, 7000)
    await finished
    src.disconnect()

    const blob = new Blob(chunks, { type: 'audio/webm' })
    if (blob.size > 100) {
      await sendToBackend(blob)
    } else {
      log("No utterance detected after wake word.")
    }
  }, [selected, sendToBackend]) // Dependencies updated

  // Starts the wake word detection cycle
  const startDetect = useCallback(async () => {
    if (detectStreamRef.current) return;
    log('Starting wake word detection...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      detectStreamRef.current = stream;
      detectCycleRef.current = true;
      setPhase('listening')
      setListening(true)

      const runCycle = () => {
        if (!detectCycleRef.current || !detectStreamRef.current) return;

        const rec = new MediaRecorder(detectStreamRef.current, { mimeType: 'audio/webm;codecs=opus' });
        detectRecRef.current = rec;
        const chunks: BlobPart[] = [];
        rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data) };

        rec.onstop = async () => {
          try {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            if (blob.size > 100) {
              const fd = new FormData();
              fd.append('audio', blob, 'wake.webm');
              const r = await fetch(`${BACKEND_BASE}/api/wake`, { method: 'POST', body: fd });
              const d = await r.json();

              const wakeText = (d.text || "").toLowerCase();
              if (STOP_WORDS.some(word => wakeText.includes(word))) {
                log("Stop word detected in wake phrase. Halting.");
                stopAllActivity();
                return;
              }

              if (d.detected) {
                setPhase('listening')
                await captureUtterance();
                log("Interaction finished. Resuming wake word detection...");
              }
            }
          } catch (e) {
            log(`Wake detection error: ${String(e)}`);
          } finally {
            if (detectCycleRef.current) {
              setTimeout(runCycle, 100); // Schedule the next check
            }
          }
        };
        rec.start();
        setTimeout(() => { if (rec.state === 'recording') rec.stop() }, 1800);
      };
      runCycle();
    } catch (err) {
      log(`Failed to get microphone: ${String(err)}`);
      setListening(false);
      setPhase('idle')
    }
  }, [captureUtterance, stopAllActivity]);

  // Main toggle for the always-on feature
  const toggleAlways = async () => {
    if (listening) {
      stopAllActivity()
    } else {
      await startDetect()
    }
  }

  return (
    <div style={{ fontFamily: 'system-ui', color: '#eaeef6', background: 'radial-gradient(1200px 800px at 20% -10%, #0c1530 0%, #0b0f19 40%, #0b0f19 100%)', minHeight: '100vh' }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 12, background: 'rgba(18, 24, 41, 0.9)', position: 'sticky', top: 0, borderBottom: '1px solid #2a3350', backdropFilter: 'saturate(120%) blur(4px)' }}>
        <h2 style={{ margin: 0 }}>Edge Avatar React</h2>
        <select value={selected} onChange={(e) => setSelected(e.target.value)} style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #2a3350', background: '#131b32', color: '#eaeef6' }}>
          {avatars.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
        </select>
        <button onMouseDown={holdToTalk} disabled={listening} style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #2a3350', background: '#131b32', color: '#eaeef6', cursor: listening ? 'not-allowed' : 'pointer' }}>Hold to Talk</button>
        <button onClick={toggleAlways} style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #2a3350', background: '#131b32', color: '#eaeef6' }}>{listening ? 'Stop Listening' : 'Start Listening'}</button>
        <button onClick={stopAllActivity} style={{ padding: '8px 12px', borderRadius: 10, border: '1px solid #ef4444', background: '#ef4444', color: '#eaeef6' }}>STOP</button>
        <div style={{ marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: 8, border: '1px solid #2a3350', padding: '6px 10px', borderRadius: 999 }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: phase === 'speaking' ? '#38bdf8' : phase === 'listening' ? '#22c55e' : '#64748b', boxShadow: phase === 'speaking' ? '0 0 10px #38bdf8' : phase === 'listening' ? '0 0 6px #22c55e' : 'none' }} />
          <span style={{ fontSize: 12, color: '#cbd5e1' }}>{phase === 'speaking' ? 'Responding' : phase === 'listening' ? 'Listening… Speak now' : 'Idle'}</span>
        </div>
      </div>

      <div style={{ position: 'relative', width: 380, height: 380, margin: '32px auto', borderRadius: '50%', overflow: 'hidden', border: `3px solid ${phase === 'speaking' ? '#38bdf8' : phase === 'listening' ? '#22c55e' : '#2a3350'}`, boxShadow: phase === 'speaking' ? '0 0 22px rgba(56,189,248,0.35)' : phase === 'listening' ? '0 0 22px rgba(34,197,94,0.35)' : '0 0 30px rgba(0,0,0,0.4)', transition: 'box-shadow 180ms ease, border-color 180ms ease' }}>
        <video ref={idleRef} playsInline muted loop style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />
        <video ref={lipRef} playsInline muted style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0, transition: 'opacity 120ms ease-in-out' }} />
      </div>

      <div style={{ maxWidth: 900, margin: '0 auto 8px', display: 'flex', justifyContent: 'flex-end' }}>
        <button onClick={() => { if (logRef.current) logRef.current.textContent = '' }} style={{ padding: '6px 10px', borderRadius: 8, border: '1px solid #2a3350', background: '#1a2238', color: '#eaeef6' }}>Clear</button>
      </div>
      <div ref={logRef} style={{ maxWidth: 900, margin: '0 auto 24px', padding: 12, background: '#121829', border: '1px solid #2a3350', borderRadius: 10, whiteSpace: 'pre-wrap', minHeight: 160, maxHeight: 320, overflowY: 'auto', lineHeight: 1.4, fontSize: 14 }} />
      <Scheduler userId={clientId.current} avatarId={selected} />
    </div>
  )
}