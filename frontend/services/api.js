/**
 * DeepGuard AI — Frontend REST Service
 */

const BASE_URL = '/api';

export const callService = {
  async getScenarios() {
    const res = await fetch(`${BASE_URL}/call/scenarios`);
    return await res.json();
  },

  async startCall(options = {}) {
    const res = await fetch(`${BASE_URL}/call/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options)
    });
    return await res.json();
  },

  async stopCall(callId) {
    const formData = new FormData();
    formData.append('call_id', callId);
    const res = await fetch(`${BASE_URL}/call/stop`, {
      method: 'POST',
      body: formData
    });
    return await res.json();
  },

  async getStatus(callId) {
    const url = callId ? `${BASE_URL}/call/status?call_id=${callId}` : `${BASE_URL}/call/status`;
    const res = await fetch(url);
    return await res.json();
  },

  async uploadAudioFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/call/upload`, {
      method: 'POST',
      body: formData
    });
    return await res.json();
  }
};

export const modelService = {
  async getInfo() {
    const res = await fetch(`${BASE_URL}/model/info`);
    return await res.json();
  },

  async getMetrics() {
    const res = await fetch(`${BASE_URL}/model/metrics`);
    return await res.json();
  }
};
