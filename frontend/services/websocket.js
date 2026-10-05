/**
 * DeepGuard AI — Real-time WebSocket Service
 */

export class AudioWebSocketService {
  constructor(callId, onMessage, onError, onClose) {
    this.callId = callId;
    this.onMessage = onMessage;
    this.onError = onError;
    this.onClose = onClose;
    this.socket = null;
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/ws/audio/${this.callId}`;

    this.socket = new WebSocket(url);

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (this.onMessage) this.onMessage(data);
      } catch (e) {
        console.error('Error decoding WebSocket payload:', e);
      }
    };

    this.socket.onerror = (err) => {
      if (this.onError) this.onError(err);
    };

    this.socket.onclose = () => {
      if (this.onClose) this.onClose();
    };
  }

  sendAudioChunk(pcmBuffer) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(pcmBuffer);
    }
  }

  sendCommand(action, payload = {}) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ action, ...payload }));
    }
  }

  disconnect() {
    if (this.socket) {
      try {
        this.socket.close();
      } catch (e) {}
      this.socket = null;
    }
  }
}
