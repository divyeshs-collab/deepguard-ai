import React from 'react';

export default function AudioAnalysisCard({ detection, windowsAnalyzed = 0 }) {
  const speechDetected = detection?.speech_detected ? 'YES' : (detection ? 'NO' : '--');
  const quality = detection?.quality_status || '--';
  const latency = detection?.inference_latency_ms !== undefined ? `${detection.inference_latency_ms} ms` : '-- ms';
  const rms = detection?.acoustic_metrics?.rms || 0.0;
  const rmsPct = Math.min(100, Math.round(rms * 400));

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Acoustic Telemetry</span>
        <span className="text-xs font-mono text-slate-400">HOP: 1.0s</span>
      </div>

      <div className="space-y-3 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-slate-500">Current Window</span>
          <span className="font-mono font-medium text-slate-800">3.0 seconds (48,000 samples)</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">Windows Analyzed</span>
          <span className="font-mono font-medium text-slate-800">{windowsAnalyzed}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">Speech Detected (VAD)</span>
          <span className={`font-semibold ${detection?.speech_detected ? 'text-emerald-600' : 'text-slate-500'}`}>
            {speechDetected}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">Audio Quality</span>
          <span className="font-semibold text-slate-600">{quality}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">Inference Latency</span>
          <span className="font-mono font-medium text-emerald-600">{latency}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-slate-500">Spectrogram Shape</span>
          <span className="font-mono text-slate-600">128 Mel &times; 188 Frames</span>
        </div>
      </div>

      <div className="pt-3 border-t border-slate-100">
        <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
          <span>Energy (RMS Level)</span>
          <span className="font-mono">{rms.toFixed(4)}</span>
        </div>
        <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
          <div
            className="bg-indigo-600 h-full transition-all duration-200"
            style={{ width: `${rmsPct}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
}
