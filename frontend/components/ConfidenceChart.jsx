import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine
} from 'recharts';

export default function ConfidenceChart({ history }) {
  const chartData = history.map((pt, i) => ({
    time: pt.time || `w${i}`,
    raw: Math.round(pt.raw * 100),
    smoothed: Math.round(pt.smoothed * 100)
  }));

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900">Live Confidence Timeline</h3>
          <p className="text-xs text-slate-500">Real-time rolling estimation of synthetic speech likelihood</p>
        </div>
        <div className="flex items-center space-x-4 text-xs font-medium">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-0.5 bg-slate-300 inline-block"></span>
            <span className="text-slate-500">Raw Window</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-1 bg-indigo-600 rounded-sm inline-block"></span>
            <span className="text-slate-700">Smoothed</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-0.5 border-t border-dashed border-amber-500 inline-block"></span>
            <span className="text-amber-600">Threshold (70%)</span>
          </div>
        </div>
      </div>

      <div className="w-full h-56 bg-slate-50 rounded-lg border border-slate-100 p-2">
        {chartData.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center text-xs text-slate-400 font-medium">
            Awaiting live stream data...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="time" stroke="#94A3B8" fontSize={11} />
              <YAxis domain={[0, 100]} stroke="#94A3B8" fontSize={11} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#E2E8F0', borderRadius: '8px', fontSize: '12px' }}
              />
              <ReferenceLine y={70} stroke="#F59E0B" strokeDasharray="3 3" label={{ value: '70% Threshold', fill: '#D97706', fontSize: 10 }} />
              <Line type="monotone" dataKey="raw" stroke="#CBD5E1" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="smoothed" stroke="#4F46E5" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
