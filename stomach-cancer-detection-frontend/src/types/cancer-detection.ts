export type DiagnosisResult = 'malignant' | 'benign' | 'inconclusive';

export interface CellularFeature {
  name: string;
  score: number; // 0 to 100
  status: 'normal' | 'moderate' | 'abnormal';
  description: string;
}

export interface PredictionResult {
  id: string;
  timestamp: string;
  diagnosis: DiagnosisResult;
  diagnosisTitle: string;
  confidence: number; // 0 to 100
  riskLevel: 'Low' | 'Moderate' | 'High' | 'Critical';
  histologicalSubtype: string;
  laurenClassification?: 'Intestinal Type' | 'Diffuse Type' | 'Mixed Type' | 'Non-neoplastic';
  summary: string;
  pathologistNotes: string[];
  cellularFeatures: CellularFeature[];
  recommendations: string[];
  modelDetails: {
    name: string;
    version: string;
    latencyMs: number;
    patchesAnalyzed: number;
  };
}

export interface SampleSlide {
  id: string;
  name: string;
  type: DiagnosisResult;
  subtype: string;
  description: string;
  magnification: string;
  dataUrl: string;
  presetResult: PredictionResult;
}
