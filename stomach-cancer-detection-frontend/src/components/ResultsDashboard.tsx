import React, { useState } from 'react';
import { 
  AlertTriangle, 
  CheckCircle2, 
  FileText, 
  Printer, 
  Copy, 
  Check, 
  ArrowLeft, 
  RotateCcw,
  Microscope, 
  ShieldAlert,
  ChevronRight
} from 'lucide-react';
import { PredictionResult } from '../types/cancer-detection';

interface ResultsDashboardProps {
  result: PredictionResult;
  imageSrc: string | null;
  fileName: string | null;
  onBackToUpload: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  result,
  imageSrc,
  fileName,
  onBackToUpload,
}) => {
  const [copied, setCopied] = useState(false);
  const isMalignant = result.diagnosis === 'malignant';

  const handleCopySummary = () => {
    const text = `STOMACH CANCER HISTOPATHOLOGY PREDICTION REPORT
--------------------------------------------------
Specimen File: ${fileName || 'Digital Slide'}
Diagnosis: ${result.diagnosisTitle} (${result.diagnosis.toUpperCase()})
Confidence Score: ${result.confidence}%
Risk Classification: ${result.riskLevel}
Subtype: ${result.histologicalSubtype}
Lauren Classification: ${result.laurenClassification || 'N/A'}
Timestamp: ${result.timestamp}

SUMMARY:
${result.summary}

DIAGNOSTIC FINDINGS:
${result.pathologistNotes.map((n) => `• ${n}`).join('\n')}

RECOMMENDED NEXT STEPS:
${result.recommendations.map((r) => `• ${r}`).join('\n')}

Model: ${result.modelDetails.name} (${result.modelDetails.version})`;

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div id="results-dashboard" className="space-y-6 animate-in fade-in duration-300 print:m-0 print:p-0">
      {/* 1. Top Navigation Bar */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-200">
        <button
          type="button"
          id="btn-back-to-upload"
          onClick={onBackToUpload}
          className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:text-slate-950 bg-white hover:bg-slate-100 rounded-lg border border-slate-200 shadow-xs transition-colors cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4 text-slate-500" />
          <span>Upload Another Image</span>
        </button>

        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="hidden sm:inline">Prediction completed at</span>
          <span className="font-mono font-medium text-slate-700 bg-slate-100 px-2 py-0.5 rounded">
            {result.timestamp}
          </span>
        </div>
      </div>

      {/* 2. Primary Verdict Banner */}
      <div
        className={`rounded-2xl border p-6 sm:p-7 shadow-xs transition-all ${
          isMalignant
            ? 'bg-rose-50/80 border-rose-200 text-rose-950'
            : 'bg-emerald-50/80 border-emerald-200 text-emerald-950'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                  isMalignant
                    ? 'bg-rose-100 text-rose-800 border border-rose-300'
                    : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                }`}
              >
                {isMalignant ? (
                  <>
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
                    <span>Cancer Detected (Malignant)</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    <span>No Cancer Detected (Benign)</span>
                  </>
                )}
              </span>

              <span
                className={`text-xs px-2.5 py-0.5 rounded font-semibold ${
                  isMalignant ? 'bg-rose-200 text-rose-900' : 'bg-emerald-200 text-emerald-900'
                }`}
              >
                Risk: {result.riskLevel}
              </span>
            </div>

            <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900">
              {isMalignant ? 'Positive for Stomach Cancer' : 'Negative for Stomach Cancer'}
            </h2>

            <p className="text-sm font-medium text-slate-700 max-w-2xl leading-relaxed">
              {result.summary}
            </p>
          </div>

          {/* Model Confidence Box */}
          <div className="shrink-0 bg-white/90 backdrop-blur-xs rounded-xl border border-slate-200/80 p-4 sm:p-5 text-center sm:text-right min-w-[170px] shadow-xs">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block">
              Confidence Score
            </span>
            <div
              className={`text-4xl sm:text-5xl font-extrabold tracking-tight font-mono my-1 ${
                isMalignant ? 'text-rose-600' : 'text-emerald-600'
              }`}
            >
              {result.confidence}%
            </div>
            <span className="text-[11px] text-slate-500 block font-mono">
              Ensemble Model v3.2
            </span>
          </div>
        </div>
      </div>

      {/* 3. Specimen Image & Diagnostic Information */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Biopsy Slide Preview Thumbnail */}
        {imageSrc && (
          <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200 p-4 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                  <Microscope className="w-3.5 h-3.5 text-teal-600" />
                  <span>Analyzed Slide Specimen</span>
                </span>
                <span className="text-[11px] text-slate-500 font-mono">40x HPF</span>
              </div>
              <div className="relative rounded-xl border border-slate-200 overflow-hidden bg-slate-900/5 aspect-4/3 flex items-center justify-center">
                <img
                  src={imageSrc}
                  alt="Analyzed histopathology biopsy specimen"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <span className="truncate max-w-[200px]" title={fileName || 'Specimen'}>
                {fileName || 'Histopathology_Biopsy.png'}
              </span>
              <span className="text-slate-400 font-mono">H&amp;E Stained</span>
            </div>
          </div>
        )}

        {/* Classification Details Overview */}
        <div className={`${imageSrc ? 'lg:col-span-7' : 'lg:col-span-12'} bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs flex flex-col justify-between`}>
          <div>
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-4">
              <FileText className="w-4 h-4 text-teal-600" />
              <span>Histological Classification</span>
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 mb-4">
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-150">
                <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
                  Diagnosis Subtype
                </span>
                <span className="text-sm font-bold text-slate-900 mt-1 block">
                  {result.histologicalSubtype}
                </span>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-150">
                <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
                  Lauren Classification
                </span>
                <span className="text-sm font-bold text-slate-900 mt-1 block">
                  {result.laurenClassification || 'Non-neoplastic'}
                </span>
              </div>
            </div>

            <div className="text-xs text-slate-600 leading-relaxed bg-teal-50/50 rounded-xl p-3.5 border border-teal-100">
              <strong className="text-teal-950 font-semibold block mb-0.5">Summary Interpretation:</strong>
              {result.diagnosisTitle} - {result.summary}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 font-mono">
            <span>Model: {result.modelDetails.name}</span>
            <span>Latency: {result.modelDetails.latencyMs}ms</span>
          </div>
        </div>
      </div>

      {/* 4. Diagnostic Observations & Recommended Steps */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Diagnostic Findings */}
        <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-3">
            <Microscope className="w-4 h-4 text-teal-600" />
            <span>Pathological Findings</span>
          </h3>
          <ul className="space-y-2.5">
            {result.pathologistNotes.map((note, idx) => (
              <li key={idx} className="text-xs text-slate-700 flex items-start gap-2.5 leading-relaxed">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-600 mt-1.5 shrink-0" />
                <span>{note}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Recommended Clinical Workup */}
        <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-3">
            <ChevronRight className="w-4 h-4 text-teal-600" />
            <span>Recommended Clinical Actions</span>
          </h3>
          <ul className="space-y-2.5">
            {result.recommendations.map((rec, idx) => (
              <li key={idx} className="text-xs text-slate-700 flex items-start gap-2 leading-relaxed">
                <span className="text-teal-600 font-bold shrink-0">→</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* 5. Clean Action Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 bg-white rounded-xl border border-slate-200 shadow-xs print:hidden">
        <div className="flex items-center gap-2">
          <button
            type="button"
            id="btn-copy-summary"
            onClick={handleCopySummary}
            className="px-3.5 py-2 text-xs font-semibold text-slate-700 hover:text-slate-900 bg-slate-50 hover:bg-slate-100 rounded-lg border border-slate-200 transition-colors flex items-center gap-1.5 cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-teal-600" />
                <span className="text-teal-700">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-500" />
                <span>Copy Summary</span>
              </>
            )}
          </button>

          <button
            type="button"
            id="btn-print-report"
            onClick={handlePrint}
            className="px-3.5 py-2 text-xs font-semibold text-slate-700 hover:text-slate-900 bg-slate-50 hover:bg-slate-100 rounded-lg border border-slate-200 transition-colors flex items-center gap-1.5 cursor-pointer"
          >
            <Printer className="w-3.5 h-3.5 text-slate-500" />
            <span>Print Report</span>
          </button>
        </div>

        <button
          type="button"
          id="btn-analyze-another"
          onClick={onBackToUpload}
          className="px-4 py-2 text-xs font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-lg shadow-xs transition-colors flex items-center gap-1.5 cursor-pointer"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Analyze Another Slide</span>
        </button>
      </div>

      {/* 6. Medical Safety Disclaimer Banner */}
      <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200 text-amber-900 text-xs flex items-start gap-3">
        <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold block text-amber-950">
            Clinical Decision Support Disclaimer
          </span>
          <p className="mt-0.5 text-amber-800 leading-relaxed">
            This AI prediction is designed for diagnostic decision support and screening. It does not replace microscopic validation by a certified pathologist.
          </p>
        </div>
      </div>
    </div>
  );
};
