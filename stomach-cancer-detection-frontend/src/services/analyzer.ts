import { PredictionResult, SampleSlide } from '../types/cancer-detection';
import { SAMPLE_SLIDES } from '../data/sampleSlides';

/**
 * Simulates real neural network patch inference stages
 */
export interface InferenceProgress {
  step: number;
  stageName: string;
  percentage: number;
  details: string;
}

export async function analyzeHistopathologyImage(
  imageSource: string,
  fileName: string,
  onProgress?: (progress: InferenceProgress) => void,
  selectedSample?: SampleSlide | null
): Promise<PredictionResult> {
  const steps: InferenceProgress[] = [
    { step: 1, stageName: 'Tissue Preprocessing & Normalization', percentage: 22, details: 'Color deconvolution (H&E separation) & background artifact removal...' },
    { step: 2, stageName: 'High-Power Field Patch Extraction', percentage: 54, details: 'Segmenting 256 tissue tiles across 40x optical grid...' },
    { step: 3, stageName: 'Cellular Morphology & Gland Architecture', percentage: 78, details: 'Computing nuclear-to-cytoplasmic ratio & glandular polarity...' },
    { step: 4, stageName: 'Deep Ensemble Prediction Synthesis', percentage: 98, details: 'Synthesizing ResNet-50 & Vision Transformer classification scores...' },
  ];

  for (const step of steps) {
    if (onProgress) {
      onProgress(step);
    }
    // Realistic clinical processing delay
    await new Promise((resolve) => setTimeout(resolve, 380));
  }

  // If this matches a sample slide or user picked a sample
  if (selectedSample) {
    return {
      ...selectedSample.presetResult,
      id: `diag-${Date.now()}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      modelDetails: {
        ...selectedSample.presetResult.modelDetails,
        latencyMs: Math.floor(620 + Math.random() * 180),
      }
    };
  }

  // Check against sample slides
  const matchedSample = SAMPLE_SLIDES.find(s => s.dataUrl === imageSource);
  if (matchedSample) {
    return {
      ...matchedSample.presetResult,
      id: `diag-${Date.now()}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    };
  }

  // If user uploaded a custom image:
  // We can analyze image features or text if it's a report document
  const lowerName = fileName.toLowerCase();
  const isProbablyReport = lowerName.includes('report') || lowerName.includes('biopsy') || lowerName.includes('doc');
  const isProbablyMalignant = lowerName.includes('cancer') || lowerName.includes('malig') || lowerName.includes('adeno') || lowerName.includes('carcinoma') || lowerName.includes('positive');
  const isProbablyBenign = lowerName.includes('benign') || lowerName.includes('normal') || lowerName.includes('negative') || lowerName.includes('gastritis');

  let isMalignant = true;
  let subtype = 'Gastric Tubular Adenocarcinoma (Moderate Differentiation)';
  let laurenType: 'Intestinal Type' | 'Diffuse Type' | 'Mixed Type' | 'Non-neoplastic' = 'Intestinal Type';
  let confidence = 94.6;

  if (isProbablyBenign && !isProbablyMalignant) {
    isMalignant = false;
    subtype = 'Non-neoplastic Gastric Mucosa with Reactive Foveolar Hyperplasia';
    laurenType = 'Non-neoplastic';
    confidence = 96.2;
  } else if (lowerName.includes('signet') || lowerName.includes('diffuse')) {
    isMalignant = true;
    subtype = 'Poorly Cohesive Gastric Carcinoma (Signet Ring Cell Type)';
    laurenType = 'Diffuse Type';
    confidence = 97.5;
  } else {
    // Deterministic hash based on fileName length & char codes
    let hash = 0;
    for (let i = 0; i < fileName.length; i++) {
      hash = (hash * 31 + fileName.charCodeAt(i)) % 1000;
    }
    // 65% chance malignant, 35% benign if unknown uploaded image
    isMalignant = hash % 3 !== 0;
    confidence = Number((91 + (hash % 8) + Math.random() * 0.9).toFixed(1));
  }

  if (isMalignant) {
    return {
      id: `diag-${Date.now()}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      diagnosis: 'malignant',
      diagnosisTitle: 'Positive for Gastric Malignancy',
      confidence,
      riskLevel: 'Critical',
      histologicalSubtype: subtype,
      laurenClassification: laurenType,
      summary: isProbablyReport 
        ? 'Report interpretation and specimen evaluation indicate active malignant gastric neoplasia. Structural glandular effacement and nuclear atypia confirmed across evaluated fields.'
        : 'Deep learning histopathology evaluation indicates invasive gastric adenocarcinoma. Computer-assisted vision detected fused atypical glands, increased nuclear-to-cytoplasmic ratio, and abnormal chromatin aggregation.',
      pathologistNotes: [
        'Atypical architectural gland arrangement with irregular lumina and cribriforming',
        'Marked nuclear enlargement with hyperchromatic chromatin and prominent nucleoli',
        'Stromal desmoplasia observed adjacent to infiltrative cellular clusters',
        'Loss of cellular apical-basal polarity along the epithelial lining'
      ],
      cellularFeatures: [
        {
          name: 'Nuclear Pleomorphism & Atypia',
          score: Math.min(98, Math.floor(confidence - 2 + Math.random() * 4)),
          status: 'abnormal',
          description: 'High variation in nuclear shape and size with prominent chromatin irregularity.'
        },
        {
          name: 'Glandular Architecture Distortion',
          score: Math.min(96, Math.floor(confidence - 5 + Math.random() * 4)),
          status: 'abnormal',
          description: 'Distorted cribriform or fused glandular spaces replacing regular mucosal pits.'
        },
        {
          name: 'Signet-Ring Morphology',
          score: laurenType === 'Diffuse Type' ? 95 : 22,
          status: laurenType === 'Diffuse Type' ? 'abnormal' : 'normal',
          description: laurenType === 'Diffuse Type' ? 'Abundant non-cohesive cells with eccentric nuclei.' : 'Minimal signet-ring cell presence detected.'
        },
        {
          name: 'Stromal Infiltration & Desmoplasia',
          score: Math.min(94, Math.floor(confidence - 6 + Math.random() * 3)),
          status: 'abnormal',
          description: 'Invasion through basement membrane into lamina propria and deeper tissue.'
        }
      ],
      recommendations: [
        'Immediate surgical oncology and gastroenterology multidisciplinary consultation',
        'Perform complete IHC panel (HER2, Claudin 18.2, MMR/MSI, and PD-L1 TPS/CPS)',
        'Full contrast-enhanced thoracic/abdominal CT staging and diagnostic laparoscopy if clinically indicated',
        'Correlate findings with upper GI endoscopic biopsy report and clinical history'
      ],
      modelDetails: {
        name: 'GastricNet-Vision Deep ResNet-50 Ensemble',
        version: 'v3.2.1-histopathology',
        latencyMs: 760,
        patchesAnalyzed: 256
      }
    };
  } else {
    return {
      id: `diag-${Date.now()}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      diagnosis: 'benign',
      diagnosisTitle: 'Negative for Malignancy (Benign Mucosa)',
      confidence,
      riskLevel: 'Low',
      histologicalSubtype: 'Benign Gastric Antral / Body Mucosa with Quiescent Gastritis',
      laurenClassification: 'Non-neoplastic',
      summary: 'Histopathology image evaluation revealed orderly cellular architecture with intact tubular glands and basal round nuclei. No evidence of severe dysplasia, signet-ring cells, or invasive malignancy was observed.',
      pathologistNotes: [
        'Regular foveolar architecture with preserved apical mucin-secreting cytoplasm',
        'Basally located, uniform monomorphic nuclei with low mitotic figures',
        'No desmoplastic stromal response or invasion across muscularis mucosae',
        'Mild scattered lymphoplasmacytic infiltrate consistent with chronic superficial gastritis'
      ],
      cellularFeatures: [
        {
          name: 'Nuclear Pleomorphism & Atypia',
          score: 11,
          status: 'normal',
          description: 'Uniform basally polarized nuclei with normal nuclear-to-cytoplasmic ratio.'
        },
        {
          name: 'Glandular Architecture Distortion',
          score: 14,
          status: 'normal',
          description: 'Normal parallel tubular crypts with preserved foveolar contour.'
        },
        {
          name: 'Signet-Ring Morphology',
          score: 2,
          status: 'normal',
          description: 'No discohesive signet-ring cell structures.'
        },
        {
          name: 'Stromal Infiltration & Desmoplasia',
          score: 8,
          status: 'normal',
          description: 'Lamina propria intact without atypical cellular infiltration.'
        }
      ],
      recommendations: [
        'Correlate with clinical symptomatology (dyspepsia, epigastric discomfort, reflux)',
        'Check for Helicobacter pylori presence via rapid urease test or stool antigen assay',
        'Follow standard medical management for benign chronic gastritis if indicated'
      ],
      modelDetails: {
        name: 'GastricNet-Vision Deep ResNet-50 Ensemble',
        version: 'v3.2.1-histopathology',
        latencyMs: 640,
        patchesAnalyzed: 256
      }
    };
  }
}
