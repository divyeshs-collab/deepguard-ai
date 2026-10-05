import React, { useState, useEffect } from 'react';
import { modelService } from '../services/api';

export default function ModelInfoPage() {
  const [metrics, setMetrics] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [infoData, metricsData] = await Promise.all([
          modelService.getInfo(),
          modelService.getMetrics()
        ]);
        setInfo(infoData);
        setMetrics(metricsData);
      } catch (e) {
        console.error('Failed loading model info:', e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const cm = metrics?.metrics?.classification_metrics;
  const perf = metrics?.metrics?.real_time_system_performance;
  const isEvaluated = metrics?.status === 'Evaluated' && cm;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
      <div className="flex items-center justify-between pb-4 border-b border-slate-100">
        <div>
          <h2 className="text-lg font-bold text-slate-900">AI Detection Engine Specifications</h2>
          <p className="text-xs text-slate-500">
            Architecture, training configuration, and independently verified benchmark metrics
          </p>
        </div>
        <span className="text-xs font-mono font-medium px-2.5 py-1 bg-slate-100 text-slate-700 rounded border border-slate-200">
          Model Version: v1.0.0
        </span>
      </div>

      {/* Grid of specs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
        <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-100">
          <span className="text-slate-400 block mb-1">Architecture</span>
          <span className="font-bold text-slate-900 text-sm">2D CNN Classifier</span>
          <span className="text-slate-500 block text-[11px] mt-0.5">32 &rarr; 64 &rarr; 128 &rarr; 256 &rarr; GAP &rarr; 128 &rarr; 1</span>
        </div>
        <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-100">
          <span className="text-slate-400 block mb-1">Input Representation</span>
          <span className="font-bold text-slate-900 text-sm">Log-Mel Spectrogram</span>
          <span className="text-slate-500 block text-[11px] mt-0.5">128 Mel bands, 1024 FFT, 256 Hop</span>
        </div>
        <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-100">
          <span className="text-slate-400 block mb-1">Framework</span>
          <span className="font-bold text-slate-900 text-sm">PyTorch</span>
          <span className="text-slate-500 block text-[11px] mt-0.5">Binary Cross Entropy + Adam</span>
        </div>
        <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-100">
          <span className="text-slate-400 block mb-1">Confidence Calibration</span>
          <span className="font-bold text-slate-900 text-sm">Temperature Scaling</span>
          <span className="text-slate-500 block text-[11px] mt-0.5">Validation set negative log-likelihood</span>
        </div>
      </div>

      {/* Metrics Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-900">Independent Test Set Metrics</h3>
          <span className="text-xs text-slate-500">Evaluated on held-out unseen speakers</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-center">
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <p className="text-[11px] text-slate-500">Accuracy</p>
            <p className="text-lg font-bold font-mono text-slate-900 mt-1">
              {isEvaluated ? `${(cm.accuracy * 100).toFixed(1)}%` : 'Not evaluated'}
            </p>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <p className="text-[11px] text-slate-500">Precision</p>
            <p className="text-lg font-bold font-mono text-slate-900 mt-1">
              {isEvaluated ? `${(cm.precision * 100).toFixed(1)}%` : 'Not evaluated'}
            </p>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <p className="text-[11px] text-slate-500">Recall</p>
            <p className="text-lg font-bold font-mono text-slate-900 mt-1">
              {isEvaluated ? `${(cm.recall * 100).toFixed(1)}%` : 'Not evaluated'}
            </p>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <p className="text-[11px] text-slate-500">F1 Score</p>
            <p className="text-lg font-bold font-mono text-slate-900 mt-1">
              {isEvaluated ? `${(cm.f1_score * 100).toFixed(1)}%` : 'Not evaluated'}
            </p>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <p className="text-[11px] text-slate-500">ROC-AUC</p>
            <p className="text-lg font-bold font-mono text-indigo-600 mt-1">
              {isEvaluated ? cm.roc_auc.toFixed(3) : 'Not evaluated'}
            </p>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <p className="text-[11px] text-slate-500">PR-AUC</p>
            <p className="text-lg font-bold font-mono text-indigo-600 mt-1">
              {isEvaluated ? cm.pr_auc.toFixed(3) : 'Not evaluated'}
            </p>
          </div>
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <p className="text-[11px] text-slate-500">EER (Equal Error)</p>
            <p className="text-lg font-bold font-mono text-emerald-600 mt-1">
              {isEvaluated ? `${(cm.equal_error_rate_eer * 100).toFixed(1)}%` : 'Not evaluated'}
            </p>
          </div>
        </div>
      </div>

      {/* Real-time system performance */}
      <div className="space-y-4 pt-4 border-t border-slate-100">
        <h3 className="text-sm font-bold text-slate-900">Real-Time System Benchmarks</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-500 block">Avg Inference Latency</span>
            <span className="text-base font-bold font-mono text-slate-800">
              {perf ? `${perf.avg_inference_latency_ms} ms` : 'Not benchmarked'}
            </span>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-500 block">P95 Latency</span>
            <span className="text-base font-bold font-mono text-slate-800">
              {perf ? `${perf.p95_inference_latency_ms} ms` : 'Not benchmarked'}
            </span>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-500 block">Audio Processing Latency</span>
            <span className="text-base font-bold font-mono text-slate-800">
              {perf ? `${perf.audio_processing_latency_ms} ms` : 'Not benchmarked'}
            </span>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-500 block">Throughput</span>
            <span className="text-base font-bold font-mono text-slate-800">
              {perf ? `${perf.throughput_windows_per_sec} win/s` : 'Not benchmarked'}
            </span>
          </div>
        </div>
      </div>

      {/* Visualizations Gallery */}
      <div className="space-y-4 pt-4 border-t border-slate-100">
        <h3 className="text-sm font-bold text-slate-900">Evaluation Artifacts &amp; Visualizations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="border border-slate-200 rounded-lg p-2 bg-slate-50 text-center">
            <p className="text-xs font-semibold text-slate-700 mb-1">Loss Curve</p>
            <img src="/api/model/evaluation-asset/training_loss.png" alt="Training Loss" className="rounded border border-slate-200 w-full h-auto bg-white" />
          </div>
          <div className="border border-slate-200 rounded-lg p-2 bg-slate-50 text-center">
            <p className="text-xs font-semibold text-slate-700 mb-1">Accuracy Curve</p>
            <img src="/api/model/evaluation-asset/training_accuracy.png" alt="Training Accuracy" className="rounded border border-slate-200 w-full h-auto bg-white" />
          </div>
          <div className="border border-slate-200 rounded-lg p-2 bg-slate-50 text-center">
            <p className="text-xs font-semibold text-slate-700 mb-1">Confusion Matrix</p>
            <img src="/api/model/evaluation-asset/confusion_matrix.png" alt="Confusion Matrix" className="rounded border border-slate-200 w-full h-auto bg-white" />
          </div>
          <div className="border border-slate-200 rounded-lg p-2 bg-slate-50 text-center">
            <p className="text-xs font-semibold text-slate-700 mb-1">ROC Curve</p>
            <img src="/api/model/evaluation-asset/roc_curve.png" alt="ROC Curve" className="rounded border border-slate-200 w-full h-auto bg-white" />
          </div>
        </div>
      </div>
    </div>
  );
}
