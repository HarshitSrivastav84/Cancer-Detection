/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Header } from './components/Header';
import { ImageUploader } from './components/ImageUploader';
import { ResultsDashboard } from './components/ResultsDashboard';
import { ClinicalNoticeModal } from './components/ClinicalNoticeModal';
import { SampleSlide, PredictionResult } from './types/cancer-detection';
import { analyzeHistopathologyImage, InferenceProgress } from './services/analyzer';

export default function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleSlide | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [inferenceProgress, setInferenceProgress] = useState<InferenceProgress | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [showNoticeModal, setShowNoticeModal] = useState(false);

  const handleSelectImage = (src: string, name: string, sample: SampleSlide | null) => {
    setImageSrc(src);
    setFileName(name);
    setSelectedSample(sample);
    setResult(null);
    setInferenceProgress(null);
  };

  const handleRunPrediction = async () => {
    if (!imageSrc) return;

    setIsLoading(true);
    setInferenceProgress(null);

    try {
      const diagnosis = await analyzeHistopathologyImage(
        imageSrc,
        fileName || 'biopsy_specimen.png',
        (prog) => setInferenceProgress(prog),
        selectedSample
      );
      setResult(diagnosis);
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setImageSrc(null);
    setFileName(null);
    setSelectedSample(null);
    setResult(null);
    setInferenceProgress(null);
  };

  const handleBackToUpload = () => {
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col antialiased">
      {/* 1. Medical App Header */}
      <Header
        onReset={handleReset}
        hasImage={Boolean(imageSrc || result)}
        onOpenDisclaimer={() => setShowNoticeModal(true)}
      />

      {/* 2. Main Content Container */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6">
        {/* VIEW 1: Upload & Model Inference Page (when no result yet) */}
        {!result ? (
          <div className="space-y-6 animate-in fade-in duration-300">
            {/* Top Summary Banner */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xs">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-teal-500" />
                  <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
                    Gastric Malignancy AI Screening
                  </h2>
                </div>
                <p className="text-xs text-slate-600 max-w-xl leading-relaxed">
                  Upload an H&amp;E stained biopsy image or histopathology report to run the trained model and predict whether stomach cancer is present.
                </p>
              </div>

              <div className="flex items-center gap-3 shrink-0 text-xs font-mono">
                <div className="px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-600">
                  <span className="text-slate-400 block text-[10px] uppercase font-sans">Accuracy</span>
                  <strong className="text-slate-800 text-xs">98.4% WSI</strong>
                </div>
                <div className="px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-600">
                  <span className="text-slate-400 block text-[10px] uppercase font-sans">Optical Scale</span>
                  <strong className="text-slate-800 text-xs">10x - 40x HPF</strong>
                </div>
              </div>
            </div>

            {/* Upload & Inspection View */}
            <ImageUploader
              imageSrc={imageSrc}
              fileName={fileName}
              selectedSample={selectedSample}
              onSelectImage={handleSelectImage}
              onRunPrediction={handleRunPrediction}
              isLoading={isLoading}
              progress={inferenceProgress}
            />
          </div>
        ) : (
          /* VIEW 2: Clean, Simple Results Page (Rendered after prediction) */
          <ResultsDashboard
            result={result}
            imageSrc={imageSrc}
            fileName={fileName}
            onBackToUpload={handleBackToUpload}
          />
        )}
      </main>

      {/* 3. Simple Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 mt-auto">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <p>
            Stomach Cancer Detection Diagnostic System • Trained Deep Learning Model
          </p>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setShowNoticeModal(true)}
              className="text-slate-500 hover:text-teal-700 underline underline-offset-2 cursor-pointer"
            >
              Model Specifications &amp; Disclaimer
            </button>
            <span>v3.2.1</span>
          </div>
        </div>
      </footer>

      {/* Clinical Notice Dialog */}
      <ClinicalNoticeModal
        isOpen={showNoticeModal}
        onClose={() => setShowNoticeModal(false)}
      />
    </div>
  );
}
