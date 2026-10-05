import React from 'react';

export default function ExplainabilityView({ explanation }) {
  const fb = explanation?.frequency_breakdown || {
    high_frequency_pct: '--',
    mid_harmonic_pct: '--',
    low_fundamental_pct: '--'
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div>
          <h3 className="text-sm font-bold text-slate-900">Audio Regions Influencing Prediction</h3>
          <p className="text-xs text-slate-500">Grad-CAM neural attention across acoustic frequency bands</p>
        </div>
        <span className="text-xs font-mono text-slate-400">128 Mel Bands &times; 1024 FFT</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-slate-600">High-Frequency Region</span>
            <span className="text-xs font-bold font-mono text-slate-800">{fb.high_frequency_pct}%</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-tight">
            4.5 kHz &ndash; 8.0 kHz. Vocoder aliasing, phase discontinuities, and high-frequency flattening.
          </p>
        </div>

        <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-slate-600">Harmonic &amp; Formant Region</span>
            <span className="text-xs font-bold font-mono text-slate-800">{fb.mid_harmonic_pct}%</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-tight">
            1.0 kHz &ndash; 4.5 kHz. Unnatural pitch tracking, synthetic resonances, and comb-filter artifacts.
          </p>
        </div>

        <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-slate-600">Fundamental &amp; Glottal Region</span>
            <span className="text-xs font-bold font-mono text-slate-800">{fb.low_fundamental_pct}%</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-tight">
            0 Hz &ndash; 1.0 kHz. Vocal cord glottal pulses, conversational breathiness, and micro-dynamics.
          </p>
        </div>
      </div>
    </div>
  );
}
