import React from 'react';
import { Mic, ShieldCheck, AlertTriangle, ShieldAlert, VolumeX } from 'lucide-react';

export default function PrimaryDetectionCard({ detection }) {
  const classification = detection?.classification || 'AWAITING AUDIO';
  const desc = detection?.state_description || 'Initiate a call or stream audio to activate continuous 3-second neural evaluation.';
  const riskLevel = detection?.risk_level || 'STANDBY';
  const rawConf = detection?.model_confidence !== undefined ? Math.round(detection.model_confidence * 100) : '--';
  const smoothConf = detection?.smoothed_confidence !== undefined ? Math.round(detection.smoothed_confidence * 100) : '--';

  let cardStyle = 'border-slate-200 bg-white';
  let badgeStyle = 'bg-slate-100 text-slate-600';
  let textStyle = 'text-slate-900';
  let icon = <Mic className="w-6 h-6 text-slate-500" />;

  if (classification === 'LIKELY HUMAN') {
    cardStyle = 'border-emerald-200 bg-emerald-50/20';
    badgeStyle = 'bg-emerald-100 text-emerald-800 border-emerald-200';
    textStyle = 'text-emerald-700';
    icon = <ShieldCheck className="w-6 h-6 text-emerald-600" />;
  } else if (classification === 'SUSPICIOUS') {
    cardStyle = 'border-amber-200 bg-amber-50/20';
    badgeStyle = 'bg-amber-100 text-amber-800 border-amber-200';
    textStyle = 'text-amber-700';
    icon = <AlertTriangle className="w-6 h-6 text-amber-600" />;
  } else if (classification === 'HIGH CONFIDENCE SYNTHETIC') {
    cardStyle = 'border-rose-200 bg-rose-50/20';
    badgeStyle = 'bg-rose-100 text-rose-800 border-rose-200';
    textStyle = 'text-rose-700';
    icon = <ShieldAlert className="w-6 h-6 text-rose-600" />;
  } else if (classification === 'INSUFFICIENT AUDIO' || classification.includes('LIMITED')) {
    icon = <VolumeX className="w-6 h-6 text-slate-400" />;
  }

  return (
    <div className={`rounded-xl border p-5 shadow-sm space-y-4 flex flex-col justify-between relative overflow-hidden transition-all duration-300 ${cardStyle}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Classification State</span>
        <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold font-mono border ${badgeStyle}`}>
          {riskLevel}
        </span>
      </div>

      <div className="my-auto py-2 text-center space-y-1">
        <div className="w-12 h-12 rounded-full mx-auto flex items-center justify-center bg-slate-100/80 mb-2">
          {icon}
        </div>
        <h2 className={`text-2xl font-bold tracking-tight ${textStyle}`}>
          {classification}
        </h2>
        <p className="text-xs text-slate-500 max-w-xs mx-auto">
          {desc}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-100">
        <div className="bg-slate-50 rounded-lg p-2.5 text-center border border-slate-100">
          <p className="text-[11px] font-medium text-slate-500">Model Confidence</p>
          <p className="text-xl font-bold font-mono text-slate-900 mt-0.5">{rawConf}%</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-2.5 text-center border border-slate-100">
          <p className="text-[11px] font-medium text-slate-500">Smoothed Confidence</p>
          <p className="text-xl font-bold font-mono text-slate-900 mt-0.5">{smoothConf}%</p>
        </div>
      </div>
    </div>
  );
}
