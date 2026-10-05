import { useRef } from 'react';

export function useAudioStream() {
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);

  const startMicrophoneCapture = async (onPcmChunk) => {
    mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: false }
    });

    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = audioContextRef.current.createMediaStreamSource(mediaStreamRef.current);

    processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
    source.connect(processorRef.current);
    processorRef.current.connect(audioContextRef.current.destination);

    processorRef.current.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      onPcmChunk(pcm.buffer);
    };
  };

  const stopMicrophoneCapture = () => {
    if (processorRef.current) {
      try { processorRef.current.disconnect(); } catch (e) {}
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (e) {}
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  };

  return {
    startMicrophoneCapture,
    stopMicrophoneCapture
  };
}
