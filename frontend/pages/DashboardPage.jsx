import React, { useState } from 'react';
import CallStatusCard from '../components/CallStatusCard';
import PrimaryDetectionCard from '../components/PrimaryDetectionCard';
import AudioAnalysisCard from '../components/AudioAnalysisCard';
import ConfidenceChart from '../components/ConfidenceChart';
import DetectionTimeline from '../components/DetectionTimeline';
import ExplainabilityView from '../components/ExplainabilityView';
import { useCallSession } from '../hooks/useCallSession';
import { useAudioStream } from '../hooks/useAudioStream';
import { callService } from '../services/api';

export default function DashboardPage() {
  const [selectedScenario, setSelectedScenario] = useState('deepfake_cloned_call');
  const {
    isActive,
    duration,
    currentDetection,
    timeline,
    history,
    startSession,
    stopSession
  } = useCallSession();

  const { startMicrophoneCapture, stopMicrophoneCapture } = useAudioStream();

  const handleStart = async () => {
    if (selectedScenario === 'live_microphone') {
      await startSession('live', selectedScenario);
    } else {
      await startSession('demo', selectedScenario);
    }
  };

  const handleStop = async () => {
    stopMicrophoneCapture();
    await stopSession();
  };

  const handleFileUpload = async (file) => {
    try {
      const res = await callService.uploadAudioFile(file);
      await startSession('demo', 'custom');
    } catch (e) {
      alert('Upload failed: ' + e.message);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top 3 Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <CallStatusCard
          isActive={isActive}
          duration={duration}
          scenario={selectedScenario}
          onScenarioChange={setSelectedScenario}
          onStart={handleStart}
          onStop={handleStop}
          onFileUpload={handleFileUpload}
        />
        <PrimaryDetectionCard detection={currentDetection} />
        <AudioAnalysisCard detection={currentDetection} windowsAnalyzed={timeline.length} />
      </div>

      {/* Middle Row: Confidence Chart & Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ConfidenceChart history={history} />
        </div>
        <div>
          <DetectionTimeline timeline={timeline} />
        </div>
      </div>

      {/* Bottom Row: Explainability */}
      <ExplainabilityView explanation={currentDetection?.explanation} />
    </div>
  );
}
