import React, { useState } from 'react';
import Header from '../components/Header';
import DashboardPage from '../pages/DashboardPage';
import ModelInfoPage from '../pages/ModelInfoPage';

export default function DeepGuardDashboard() {
  const [currentView, setCurrentView] = useState('live');

  return (
    <div className="bg-slate-50 text-slate-900 font-sans antialiased min-h-screen flex flex-col">
      <Header currentView={currentView} onViewChange={setCurrentView} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {currentView === 'live' && <DashboardPage />}
        {currentView === 'model' && <ModelInfoPage />}
        {currentView === 'explain' && <DashboardPage />}
      </main>

      <footer className="bg-white border-t border-slate-200 py-4 px-6 text-xs text-slate-500 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3 text-center md:text-left">
          <p className="max-w-4xl text-[11px] leading-relaxed text-slate-400">
            <strong className="text-slate-600 font-semibold">Regulatory Disclaimer:</strong>
            {' '}This system provides an AI-based assessment of audio characteristics associated with synthetic or manipulated speech. It does not establish that a caller is definitively using AI or deepfake technology. Results may be affected by audio quality, network conditions, speakers, recording devices, codecs, and synthetic-generation techniques. High-risk results should be validated using additional evidence and professional analysis.
          </p>
          <div className="text-[11px] font-mono text-slate-400 whitespace-nowrap">
            &copy; 2026 DeepGuard AI Sentinel
          </div>
        </div>
      </footer>
    </div>
  );
}
