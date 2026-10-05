import React from 'react';
import { Shield } from 'lucide-react';

export default function Header({ currentView, onViewChange }) {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-40 px-6 py-3.5 shadow-sm">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3.5">
          <div className="w-9 h-9 rounded-lg bg-slate-900 flex items-center justify-center text-white font-bold shadow-sm">
            <Shield className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight text-slate-900">DeepGuard AI</h1>
              <span className="text-xs px-2 py-0.5 rounded font-mono font-medium bg-slate-100 text-slate-600 border border-slate-200">
                v1.0.0
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium">Real-Time Deepfake Audio Detection Engine</p>
          </div>
        </div>

        <div className="flex items-center space-x-6">
          <nav className="hidden md:flex items-center space-x-1">
            <button
              onClick={() => onViewChange('live')}
              className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors ${
                currentView === 'live' ? 'bg-slate-100 text-slate-900' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Live Sentinel
            </button>
            <button
              onClick={() => onViewChange('model')}
              className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors ${
                currentView === 'model' ? 'bg-slate-100 text-slate-900' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              AI Detection Engine
            </button>
            <button
              onClick={() => onViewChange('explain')}
              className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors ${
                currentView === 'explain' ? 'bg-slate-100 text-slate-900' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Spectrogram &amp; Explainability
            </button>
          </nav>

          <div className="flex items-center space-x-2 pl-4 border-l border-slate-200">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
            <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider font-mono">
              System Online
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
