import React from 'react';

export default function DetectionTimeline({ timeline = [] }) {
  const getBadgeStyle = (classification) => {
    switch (classification) {
      case 'LIKELY HUMAN':
        return 'bg-emerald-100 text-emerald-800 border-emerald-200';
      case 'SUSPICIOUS':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      case 'HIGH CONFIDENCE SYNTHETIC':
        return 'bg-rose-100 text-rose-800 border-rose-200';
      case 'ANALYZING':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      default:
        return 'bg-slate-100 text-slate-600 border-slate-200';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3 flex flex-col">
      <div className="flex items-center justify-between pb-2 border-b border-slate-100">
        <div>
          <h3 className="text-sm font-bold text-slate-900">Detection Timeline</h3>
          <p className="text-xs text-slate-500">Sequential window classification audit</p>
        </div>
        <span className="text-xs font-mono font-medium text-slate-400">
          {timeline.length} logs
        </span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2 max-h-56 pr-1 text-xs">
        {timeline.length === 0 ? (
          <div className="text-center py-10 text-slate-400 text-xs">
            No windows evaluated yet.
          </div>
        ) : (
          timeline.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between p-2 rounded-lg bg-slate-50 border border-slate-100 text-[11px]"
            >
              <div className="flex items-center space-x-2">
                <span className="font-mono text-slate-400">{item.elapsed_time || '00:00'}</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${getBadgeStyle(item.classification)}`}>
                  {item.classification}
                </span>
              </div>
              <div className="font-mono text-slate-600">
                <span className="text-slate-400">Raw:</span> {Math.round((item.model_confidence || 0) * 100)}% &bull;{' '}
                <span className="text-slate-400">Smooth:</span> {Math.round((item.smoothed_confidence || 0) * 100)}%
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
