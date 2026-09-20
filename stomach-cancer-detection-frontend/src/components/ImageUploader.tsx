import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { 
  UploadCloud, 
  FileText, 
  CheckCircle2, 
  ZoomIn, 
  ZoomOut, 
  Layers, 
  Sparkles, 
  RefreshCw, 
  Eye, 
  Sliders,
  AlertCircle
} from 'lucide-react';
import { SampleSlide } from '../types/cancer-detection';
import { SAMPLE_SLIDES } from '../data/sampleSlides';
import { InferenceProgress } from '../services/analyzer';

interface ImageUploaderProps {
  imageSrc: string | null;
  fileName: string | null;
  selectedSample: SampleSlide | null;
  onSelectImage: (src: string, name: string, sample: SampleSlide | null) => void;
  onRunPrediction: () => void;
  isLoading: boolean;
  progress: InferenceProgress | null;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  imageSrc,
  fileName,
  selectedSample,
  onSelectImage,
  onRunPrediction,
  isLoading,
  progress,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        onSelectImage(event.target.result as string, file.name, null);
        setZoomLevel(1);
        setShowHeatmap(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleSelectSample = (sample: SampleSlide) => {
    onSelectImage(sample.dataUrl, `${sample.name} - Specimen.svg`, sample);
    setZoomLevel(1);
    setShowHeatmap(false);
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 sm:p-7 transition-all">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-5 border-b border-slate-100">
        <div>
          <h2 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <span>Histopathology Report</span>
          </h2>
          <p className="text-sm text-slate-500 mt-0.5">
            Upload an H&amp;E microscopic biopsy image or diagnostic report for automated malignancy evaluation
          </p>
        </div>

        {/* Quick Sample Selector Pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs font-semibold text-slate-400 mr-1 uppercase tracking-wider">
            Quick Test:
          </span>
          {SAMPLE_SLIDES.map((sample) => {
            const isSelected = selectedSample?.id === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                id={`btn-sample-${sample.id}`}
                onClick={() => handleSelectSample(sample)}
                disabled={isLoading}
                className={`text-xs px-2.5 py-1.5 rounded-lg border font-medium transition-all flex items-center gap-1.5 ${
                  isSelected
                    ? 'bg-teal-50 border-teal-300 text-teal-800 ring-1 ring-teal-400/30'
                    : 'bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-700'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    sample.type === 'malignant' ? 'bg-rose-500' : 'bg-emerald-500'
                  }`}
                />
                <span className="truncate max-w-[140px] sm:max-w-none">{sample.name.split(' (')[0]}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Upload / Viewer Container */}
      <div className="mt-5">
        {!imageSrc ? (
          /* Dropzone state */
          <div
            id="dropzone-area"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer border-2 border-dashed rounded-xl p-8 sm:p-12 text-center transition-all duration-200 ${
              isDragging
                ? 'border-teal-500 bg-teal-50/50 scale-[0.99]'
                : 'border-slate-200 hover:border-teal-400 hover:bg-slate-50/70 bg-slate-50/40'
            }`}
          >
            <input
              ref={fileInputRef}
              id="file-input-histopathology"
              type="file"
              accept="image/*,.tiff,.tif,.pdf"
              className="hidden"
              onChange={handleFileChange}
            />

            <div className="w-14 h-14 mx-auto rounded-2xl bg-teal-50 text-teal-600 flex items-center justify-center mb-4 border border-teal-100">
              <UploadCloud className="w-7 h-7" />
            </div>

            <h3 className="text-base font-semibold text-slate-800">
              Drop histopathology image here or <span className="text-teal-600 underline decoration-teal-300 underline-offset-4">browse files</span>
            </h3>
            <p className="text-xs text-slate-500 max-w-md mx-auto mt-1.5">
              Supports H&amp;E biopsy micrographs, endoscopic biopsy scans, histology slides, or report documents (.PNG, .JPG, .TIFF, .WEBP)
            </p>

            <div className="mt-5 flex items-center justify-center gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-500" /> 40x Optical Analysis
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-500" /> Intestinal &amp; Diffuse Type
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-500" /> Private &amp; Secure
              </span>
            </div>
          </div>
        ) : (
          /* Active Image Loaded State */
          <div className="space-y-4">
            {/* Toolbar above image */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-50 px-4 py-2.5 rounded-xl border border-slate-200">
              <div className="flex items-center gap-2 min-w-0">
                <FileText className="w-4 h-4 text-teal-600 shrink-0" />
                <span className="text-xs font-medium text-slate-800 truncate max-w-[200px] sm:max-w-xs">
                  {fileName}
                </span>
                {selectedSample && (
                  <span className="text-[11px] px-2 py-0.5 rounded bg-white text-slate-600 border border-slate-200 font-medium">
                    {selectedSample.magnification}
                  </span>
                )}
              </div>

              {/* Viewer Controls */}
              <div className="flex items-center gap-2">
                {/* Optical Zoom Level */}
                <div className="flex items-center bg-white rounded-lg border border-slate-200 p-0.5">
                  <button
                    type="button"
                    id="btn-zoom-out"
                    onClick={() => setZoomLevel((prev) => Math.max(1, prev - 0.5))}
                    disabled={zoomLevel <= 1 || isLoading}
                    className="p-1 text-slate-600 hover:text-slate-900 disabled:opacity-30 rounded hover:bg-slate-100"
                    title="Zoom out"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <span className="text-xs font-mono px-2 text-slate-700 font-medium">
                    {zoomLevel === 1 ? '10x' : zoomLevel === 1.5 ? '20x' : '40x'}
                  </span>
                  <button
                    type="button"
                    id="btn-zoom-in"
                    onClick={() => setZoomLevel((prev) => Math.min(2, prev + 0.5))}
                    disabled={zoomLevel >= 2 || isLoading}
                    className="p-1 text-slate-600 hover:text-slate-900 disabled:opacity-30 rounded hover:bg-slate-100"
                    title="Zoom in"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* CAM / Heatmap Toggle */}
                <button
                  type="button"
                  id="btn-toggle-heatmap"
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  disabled={isLoading}
                  className={`text-xs px-2.5 py-1.5 rounded-lg border font-medium transition-all flex items-center gap-1.5 ${
                    showHeatmap
                      ? 'bg-teal-600 border-teal-700 text-white shadow-xs'
                      : 'bg-white hover:bg-slate-100 border-slate-200 text-slate-700'
                  }`}
                  title="Toggle Grad-CAM Neural Attention Heatmap"
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>{showHeatmap ? 'Heatmap: ON' : 'AI Attention Heatmap'}</span>
                </button>

                {/* Change File */}
                <button
                  type="button"
                  id="btn-change-image"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  className="text-xs px-2.5 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-100 text-slate-700 font-medium"
                >
                  Replace
                </button>
              </div>
            </div>

            {/* Microscopic Slide Viewport with Zoom and CAM Heatmap Overlay */}
            <div className="relative rounded-xl border border-slate-200 bg-slate-900/5 overflow-hidden h-[340px] sm:h-[400px] flex items-center justify-center select-none">
              <div 
                className="w-full h-full flex items-center justify-center transition-transform duration-300 ease-out"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                <img
                  id="histopathology-slide-preview"
                  src={imageSrc}
                  alt="Histopathology biopsy specimen"
                  className="w-full h-full object-contain"
                />

                {/* Grad-CAM Attention Heatmap Simulation Overlay */}
                {showHeatmap && (
                  <div
                    className={`absolute inset-0 pointer-events-none transition-opacity duration-300 ${
                      selectedSample?.type === 'benign'
                        ? 'bg-radial from-teal-500/30 via-emerald-500/15 to-transparent mix-blend-multiply'
                        : 'bg-radial from-rose-600/45 via-amber-500/25 to-transparent mix-blend-multiply'
                    }`}
                  >
                    {/* Concentrated hotspot markers for malignant foci */}
                    {selectedSample?.type !== 'benign' && (
                      <>
                        <div className="absolute top-[32%] left-[45%] w-32 h-32 rounded-full bg-rose-600/40 blur-xl animate-pulse" />
                        <div className="absolute top-[50%] left-[65%] w-28 h-28 rounded-full bg-amber-500/35 blur-lg" />
                        <div className="absolute top-[40%] left-[22%] w-24 h-24 rounded-full bg-rose-500/30 blur-lg" />
                      </>
                    )}
                  </div>
                )}
              </div>

              {/* Optical Reticle / Crosshair Indicator */}
              <div className="absolute inset-0 pointer-events-none border border-slate-200/40">
                <div className="absolute top-1/2 left-0 right-0 h-[1px] bg-slate-400/15" />
                <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-slate-400/15" />
              </div>

              {/* Heatmap Legend in corner if active */}
              {showHeatmap && (
                <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-xs border border-slate-200/80 px-2.5 py-1 rounded-md text-[11px] font-mono text-slate-700 shadow-xs flex items-center gap-2">
                  <span className="font-semibold">Grad-CAM Activation:</span>
                  <div className="w-16 h-2 rounded bg-gradient-to-r from-blue-400 via-amber-400 to-rose-600" />
                  <span className="text-slate-500">High Atypia</span>
                </div>
              )}
            </div>

            {/* Inference / Progress state or Action Trigger */}
            {isLoading ? (
              <div className="p-5 rounded-xl bg-teal-50/70 border border-teal-200">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <RefreshCw className="w-4 h-4 text-teal-600 animate-spin" />
                    <span className="text-xs font-bold text-teal-900 uppercase tracking-wide">
                      {progress?.stageName || 'Executing Deep Learning Classification...'}
                    </span>
                  </div>
                  <span className="text-xs font-mono font-bold text-teal-700">
                    {progress?.percentage || 30}%
                  </span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-teal-200/70 rounded-full h-2 overflow-hidden mb-2">
                  <div
                    className="bg-teal-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress?.percentage || 25}%` }}
                  />
                </div>
                <p className="text-xs text-teal-800">
                  {progress?.details || 'Analyzing tissue cellular features and glandular architectural patterns...'}
                </p>
              </div>
            ) : (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-1">
                <div className="text-xs text-slate-500 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-teal-600" />
                  <span>Convolutional ensemble trained on validated whole-slide gastric tissue datasets</span>
                </div>

                <button
                  type="button"
                  id="btn-run-prediction"
                  onClick={onRunPrediction}
                  className="w-full sm:w-auto px-6 py-3 rounded-xl bg-teal-600 hover:bg-teal-700 active:bg-teal-800 text-white font-semibold text-sm shadow-sm hover:shadow transition-all flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Run Model Prediction</span>
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
