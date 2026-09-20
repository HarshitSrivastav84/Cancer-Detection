import React from 'react';
import { X, ShieldCheck, CheckCircle2, AlertCircle, Database, Cpu } from 'lucide-react';

interface ClinicalNoticeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ClinicalNoticeModal: React.FC<ClinicalNoticeModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in duration-200">
      <div 
        className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-xl border border-slate-200 space-y-4"
        role="dialog"
        aria-modal="true"
      >
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center border border-teal-200">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <h3 className="text-base font-bold text-slate-900">
              Clinical Validation &amp; Safety Notice
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 p-1 rounded-lg hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-3.5 text-xs text-slate-600 leading-relaxed max-h-[70vh] overflow-y-auto pr-1">
          <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
            <span className="font-semibold text-slate-900 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-teal-600" />
              Architecture &amp; Training Details
            </span>
            <p>
              The detection model employs a deep convolutional ensemble (ResNet-50 backbone with patch-level Vision Transformer attention) trained on multi-center whole-slide imaging (WSI) histopathology datasets of gastric tissue biopsies.
            </p>
          </div>

          <div className="space-y-2">
            <span className="font-semibold text-slate-900 block">Screening Capabilities:</span>
            <ul className="space-y-1.5">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0 mt-0.5" />
                <span>Detection of Tubular, Papillary, and Mucinous Gastric Adenocarcinoma.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0 mt-0.5" />
                <span>Identification of Diffuse Signet-Ring Cell Carcinoma infiltrates.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0 mt-0.5" />
                <span>Discrimination against benign chronic superficial gastritis and reactive mucosal changes.</span>
              </li>
            </ul>
          </div>

          <div className="p-3 bg-amber-50 rounded-xl border border-amber-200 text-amber-900">
            <strong className="block text-amber-950 font-semibold mb-1">Diagnostic Disclaimer:</strong>
            Predictions provided by this tool are strictly intended as secondary diagnostic support and pre-screening triage for healthcare professionals. Final definitive diagnoses require microscopic correlation by a licensed pathologist and correlation with endoscopic, radiological, and clinical findings.
          </div>
        </div>

        <div className="pt-2 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-lg transition-colors"
          >
            Acknowledge &amp; Close
          </button>
        </div>
      </div>
    </div>
  );
};
