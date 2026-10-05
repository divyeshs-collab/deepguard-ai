import React from 'react';
import { Play, Square, Upload } from 'lucide-react';

export default function CallStatusCard({
  isActive,
  duration,
  scenario,
  onScenarioChange,
  onStart,
  onStop,
  onFileUpload
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Call Status</span>
        <span
          className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
            isActive
              ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
              : 'bg-slate-100 text-slate-600 border-slate-200'
          }`}
        >
          {isActive ? 'CONNECTED' : 'IDLE'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-slate-500">Call Duration</p>
          <p className="text-2xl font-bold font-mono text-slate-800 mt-0.5">{duration}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Audio Stream</p>
          <p className="text-sm font-semibold text-slate-800 mt-1.5 flex items-center">
            <span
              className={`w-2 h-2 rounded-full mr-1.5 ${
                isActive ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'
              }`}
            ></span>
            {isActive ? 'LIVE STREAM' : 'STANDBY'}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Sample Rate</p>
          <p className="text-sm font-semibold font-mono text-slate-700 mt-0.5">16 kHz Mono</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Processing Mode</p>
          <p className="text-sm font-semibold text-slate-700 mt-0.5">Real-Time (CNN)</p>
        </div>
      </div>

      <div className="pt-3 border-t border-slate-100 space-y-2.5">
        <div className="flex space-x-2">
          <button
            onClick={onStart}
            disabled={isActive}
            className="flex-1 py-2 px-3 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow-sm transition-all flex items-center justify-center space-x-1.5"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Start Call Analysis</span>
          </button>
          <button
            onClick={onStop}
            disabled={!isActive}
            className="py-2 px-3 bg-white hover:bg-slate-50 border border-slate-300 disabled:opacity-50 text-slate-700 text-xs font-semibold rounded-lg transition-all flex items-center justify-center space-x-1"
          >
            <Square className="w-3 h-3 fill-current" />
            <span>Stop</span>
          </button>
        </div>

        <div className="flex items-center space-x-2 text-xs">
          <span className="text-slate-500">Source:</span>
          <select
            value={scenario}
            onChange={(e) => onScenarioChange(e.target.value)}
            disabled={isActive}
            className="flex-1 bg-slate-50 border border-slate-200 text-slate-700 rounded-md py-1 px-2 text-xs focus:ring-1 focus:ring-slate-900 outline-none"
          >
            <option value="deepfake_cloned_call">Demo: Cloned CEO Voice Attack</option>
            <option value="human_authentic_call">Demo: Authentic Human Executive</option>
            <option value="phishing_synthetic_call">Demo: Synthetic Phishing Bot</option>
            <option value="ambient_silence">Demo: Ambient Silence Line</option>
            <option value="live_microphone">Live: Computer Microphone Feed</option>
          </select>
        </div>

        <div className="pt-1 flex items-center justify-between text-xs text-slate-500">
          <span>Or test file:</span>
          <label className="cursor-pointer font-medium text-slate-700 hover:text-slate-900 underline flex items-center space-x-1">
            <Upload className="w-3 h-3 inline" />
            <span>Upload WAV/MP3/FLAC</span>
            <input
              type="file"
              accept=".wav,.mp3,.flac"
              className="hidden"
              onChange={(e) => e.target.files[0] && onFileUpload(e.target.files[0])}
            />
          </label>
        </div>
      </div>
    </div>
  );
}
