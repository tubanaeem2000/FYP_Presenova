import { useState, useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { submitPresentationSession } from '../services/api';

export type SessionStatus = 'IDLE' | 'STREAMING' | 'INTERRUPTED_Q&A' | 'REPORT_GENERATING' | 'FINISHED';

export interface RealtimeFeedback {
  eyeContact: number;
  posture: number;
  hint: string;
  confidence: number;
  emotion: string;
}

export interface UseLiveSessionProps {
  userId: string;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export const useLiveSession = ({ userId, videoRef }: UseLiveSessionProps) => {
  const [status, setStatus] = useState<SessionStatus>('IDLE');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<string | null>(null);
  const [realtimeFeedback, setRealtimeFeedback] = useState<RealtimeFeedback>({
    eyeContact: 0,
    posture: 0,
    hint: 'Preparing session...',
    confidence: 0,
    emotion: 'Unavailable',
  });
  const [hasHistory, setHasHistory] = useState(false);
  const [historySummary, setHistorySummary] = useState<any>(null);
  const [finalReport, setFinalReport] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<Socket | null>(null);
  const videoIntervalRef = useRef<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioIntervalRef = useRef<any>(null);

  // Initialize socket connection
  const connectSocket = () => {
    if (socketRef.current) return;

    const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:5000/api';
    const socketHost = apiBase.replace(/\/api\/?$/, '') || 'http://localhost:5000';

    const socket = io(`${socketHost}/ws/live-session`, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socket.on('connect', () => {
      console.log('Connected to live session socket namespace');
    });

    socket.on('session_started', (data: any) => {
      console.log('Session started event:', data);
      if (data.status === 'success') {
        setSessionId(data.session_id);
        setHasHistory(data.has_history);
        setHistorySummary(data.history_summary);
        setStatus('STREAMING');
      } else {
        setError('Failed to start session');
        setStatus('IDLE');
      }
    });

    socket.on('realtime_feedback', (data: any) => {
      setRealtimeFeedback({
        eyeContact: typeof data.eye_contact === 'number' ? data.eye_contact : 0,
        posture: typeof data.posture === 'number' ? data.posture : 0,
        hint: data.hint || 'Face not detected. Waiting for a valid frame.',
        confidence: typeof data.confidence === 'number' ? data.confidence : 0,
        emotion: data.emotion || 'NOT DETECTED',
      });
    });

    socket.on('interruption_trigger', (data: any) => {
      console.log('Panelist interruption triggered:', data.question);
      setCurrentQuestion(data.question);
      setStatus('INTERRUPTED_Q&A');
    });

    socket.on('interruption_resolved', (data: any) => {
      console.log('Interruption resolved:', data);
      setCurrentQuestion(null);
      setStatus('STREAMING');
    });

    socket.on('disconnect', () => {
      console.log('Socket disconnected');
    });

    socketRef.current = socket;
  };

  // Start live presentation session
  const startSession = (topic: string) => {
    setError(null);
    setFinalReport(null);
    connectSocket();
    
    if (socketRef.current) {
      socketRef.current.emit('start_session', {
        user_id: userId || 'guest',
        topic: topic,
      });
    }
  };

  // Send answer to academic panelist interruption
  const sendAnswer = (answer: string) => {
    if (socketRef.current && sessionId && answer.trim()) {
      socketRef.current.emit('submit_answer', {
        session_id: sessionId,
        answer: answer,
      });
    }
  };

  // Stop session & fetch compiled report
  const stopSession = async () => {
    if (!sessionId) return;
    setStatus('REPORT_GENERATING');
    
    // Clear intervals and recorder
    stopMediaStreaming();

    try {
      const response = await submitPresentationSession(sessionId);
      if (response.status === 'success') {
        setFinalReport(response.report);
        setStatus('FINISHED');
      } else {
        setError('Failed to compile presentation report');
        setStatus('IDLE');
      }
    } catch (err: any) {
      setError(err.message || 'Error compiling report');
      setStatus('IDLE');
    } finally {
      // Disconnect socket
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    }
  };

  const stopMediaStreaming = () => {
    if (videoIntervalRef.current) {
      clearInterval(videoIntervalRef.current);
      videoIntervalRef.current = null;
    }
    if (audioIntervalRef.current) {
      clearInterval(audioIntervalRef.current);
      audioIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
  };

  // Loop for video frames and audio chunks streaming
  useEffect(() => {
    let audioActive = true;
    let localAudioStream: MediaStream | null = null;

    if (status === 'STREAMING' && sessionId && videoRef.current) {
      // 1. Setup Canvas for Video Frame Capture (3fps to avoid network clog)
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      videoIntervalRef.current = setInterval(() => {
        if (videoRef.current && ctx && socketRef.current) {
          canvas.width = 320; // Lower resolution for fast transmission
          canvas.height = 240;
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          const frameBase64 = canvas.toDataURL('image/jpeg', 0.6); // 60% compression quality

          socketRef.current.emit('video_frame', {
            session_id: sessionId,
            frame: frameBase64,
          });
        }
      }, 333); // ~3 frames per second

      // 2. Setup Audio Streaming (Simulate microphone audio chunks)
      navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        if (!audioActive) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        localAudioStream = stream;
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = async (e) => {
          if (e.data.size > 0 && socketRef.current && status === 'STREAMING') {
            // Read blob as base64 string
            const reader = new FileReader();
            reader.readAsDataURL(e.data);
            reader.onloadend = () => {
              const base64Audio = reader.result as string;
              socketRef.current?.emit('audio_chunk', {
                session_id: sessionId,
                audio: base64Audio,
                transcript_snippet: '', // Voice STT can optionally be generated on client
              });
            };
          }
        };

        // Record in 3-second slices
        mediaRecorder.start(3000);

        audioIntervalRef.current = setInterval(() => {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.requestData();
          }
        }, 3000);
      }).catch((err) => {
        console.warn('Microphone access denied or unavailable. Streaming visuals only.', err);
      });
    } else {
      stopMediaStreaming();
    }

    return () => {
      audioActive = false;
      if (localAudioStream) {
        localAudioStream.getTracks().forEach((track) => track.stop());
      }
      stopMediaStreaming();
    };
  }, [status, sessionId, videoRef]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopMediaStreaming();
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  return {
    status,
    sessionId,
    currentQuestion,
    realtimeFeedback,
    hasHistory,
    historySummary,
    finalReport,
    error,
    startSession,
    sendAnswer,
    stopSession,
  };
};
