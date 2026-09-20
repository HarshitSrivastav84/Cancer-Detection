import React from 'react';
import { Microscope, Activity, RotateCcw, AlertTriangle, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  onReset: () => void;
  hasImage: boolean;
  onOpenDisclaimer: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onReset, hasImage, onOpenDisclaimer }) => {
  return (
    <header className="border-b border-slate-200 bg-white/95 sticky top-0 z-40 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Left: Brand / Title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-teal-600 flex items-center justify-center text-white shadow-sm shadow-teal-700/20">
            <Microscope className="w-5 h-5 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                MediPredict
              </h1>
            </div>
            <p className="text-xs text-slate-500 hidden sm:block">
              Stomach Cancer Detection
            </p>
          </div>
        </div>

        {/* Right: Status and Actions */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-slate-100 border border-slate-200 text-xs text-slate-600 font-medium">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
          </div>

          {hasImage && (
            <button
              type="button"
              id="btn-reset-header"
              onClick={onReset}
              className="p-2 sm:px-3 sm:py-1.5 text-xs font-medium text-slate-700 hover:text-rose-700 bg-white hover:bg-rose-50 rounded-lg border border-slate-200 hover:border-rose-200 transition-colors flex items-center space-x-1.5 shadow-sm"
              title="Reset current analysis"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">New Analysis</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
