import { useState, useEffect, useRef } from 'react';
import { callService } from '../services/api';
import { AudioWebSocketService } from '../services/websocket';

export function useCallSession() {
  const [callId, setCallId] = useState(null);
  const [isActive, setIsActive] = useState(false);
  const [duration, setDuration] = useState('00:00');
  const [currentDetection, setCurrentDetection] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [history, setHistory] = useState([]);

  const timerRef = useRef(null);
  const elapsedRef = useRef(0);
  const wsRef = useRef(null);

  const startSession = async (mode = 'demo', scenario = 'deepfake_cloned_call') => {
    const id = `call-${Math.random().toString(36).substring(2, 8)}`;
    setCallId(id);
    setIsActive(true);
    setTimeline([]);
    setHistory([]);
    elapsedRef.current = 0;
    setDuration('00:00');

    timerRef.current = setInterval(() => {
      elapsedRef.current += 1;
      const mins = Math.floor(elapsedRef.current / 60);
      const secs = elapsedRef.current % 60;
      setDuration(`${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`);
    }, 1000);

    await callService.startCall({ call_id: id, mode, scenario });

    wsRef.current = new AudioWebSocketService(
      id,
      (data) => {
        setCurrentDetection(data);
        setTimeline((prev) => [data, ...prev.slice(0, 49)]);
        setHistory((prev) => [
          ...prev.slice(-29),
          {
            time: data.elapsed_time || `w${data.window_index}`,
            raw: data.model_confidence || 0,
            smoothed: data.smoothed_confidence || 0
          }
        ]);
      },
      (err) => console.error('WS Error:', err),
      () => stopSession()
    );

    wsRef.current.connect();
  };

  const stopSession = async () => {
    setIsActive(false);
    clearInterval(timerRef.current);
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }
    if (callId) {
      try {
        await callService.stopCall(callId);
      } catch (e) {}
    }
  };

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      if (wsRef.current) wsRef.current.disconnect();
    };
  }, []);

  return {
    callId,
    isActive,
    duration,
    currentDetection,
    timeline,
    history,
    startSession,
    stopSession
  };
}
