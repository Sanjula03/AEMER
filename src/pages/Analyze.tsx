import { useState, useCallback, useRef } from 'react';
import { Mic, Video, FileText, Upload, Loader2, CheckCircle, AlertCircle, Layers } from 'lucide-react';
import { ResultModal } from '../components/ResultModal';
import { saveResult } from '../lib/storage';

interface AnalyzeProps {
  onNavigate: (page: string) => void;
}

type InputType = 'audio' | 'video' | 'text' | 'multimodal' | null;
type Step = 1 | 2 | 3;

export function Analyze({ onNavigate }: AnalyzeProps) {
  const [step, setStep] = useState<Step>(1);
  const [inputType, setInputType] = useState<InputType>(null);
  const [textInput, setTextInput] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [showResultModal, setShowResultModal] = useState(false);
  // Multimodal state
  const [mmAudioFile, setMmAudioFile] = useState<File | null>(null);
  const [mmVideoFile, setMmVideoFile] = useState<File | null>(null);
  const [mmTextInput, setMmTextInput] = useState('');

  // Drag-and-drop state
  const [isDragging, setIsDragging] = useState(false);
  const [mmAudioDrag, setMmAudioDrag] = useState(false);
  const [mmVideoDrag, setMmVideoDrag] = useState(false);
  const dragCounter = useRef(0);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    dragCounter.current++;
    if (e.dataTransfer.items?.length) setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setIsDragging(false);
    dragCounter.current = 0;
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      setFile(droppedFile);
      setError(null);
    }
  }, []);

  const handleMmDrop = useCallback((e: React.DragEvent, type: 'audio' | 'video') => {
    e.preventDefault(); e.stopPropagation();
    if (type === 'audio') setMmAudioDrag(false); else setMmVideoDrag(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      if (type === 'audio') setMmAudioFile(droppedFile);
      else setMmVideoFile(droppedFile);
    }
  }, []);

  const handleInputTypeSelect = (type: InputType) => {
    setInputType(type);
    setStep(2);
    setError(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const validAudioTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp3'];
      const validVideoTypes = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/webm'];

      if (inputType === 'audio' && !validAudioTypes.includes(selectedFile.type)) {
        setError('Please upload a valid audio file (MP3, WAV, OGG)');
        return;
      }

      if (inputType === 'video' && !validVideoTypes.includes(selectedFile.type) &&
        !selectedFile.type.startsWith('image/')) {
        setError('Please upload a valid video (MP4, MOV, WEBM) or image (JPG, PNG) file');
        return;
      }

      setFile(selectedFile);
      setError(null);
    }
  };

  // Backend API URL - Hugging Face Space
  const API_URL = 'https://sanjulasunath-aemer.hf.space';

  /**
   * Send audio file to the Python backend for emotion analysis.
   * The backend runs your trained PyTorch model on the audio.
   */
  const analyzeAudio = async (audioFile: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', audioFile);

    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Analysis failed');
    }

    return response.json();
  };

  /**
   * Send text to the Python backend for emotion analysis.
   * Uses the DistilBERT text emotion model.
   */
  const analyzeText = async (text: string): Promise<any> => {
    const response = await fetch(`${API_URL}/predict-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Text analysis failed');
    }

    const result = await response.json();

    // Add weights for consistency with audio response
    return {
      ...result,
      detected_accent: null,
      audio_weight: 0.0,
      text_weight: 1.0,
      visual_weight: 0.0,
      audio_score: 0.0,
      text_score: result.confidence_score,
      visual_score: 0.0,
    };
  };

  /**
   * Send video/image file to the Python backend for facial emotion analysis.
   * Uses the ResNet-18 video emotion model with face detection.
   */
  const analyzeVideo = async (videoFile: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', videoFile);

    const response = await fetch(`${API_URL}/predict-video`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Video analysis failed');
    }

    const result = await response.json();
    // Map video response format to expected format
    return {
      emotion_label: result.emotion,
      confidence_score: result.confidence,
      all_probabilities: result.all_probabilities || {
        angry: 1 / 7, happy: 1 / 7, sad: 1 / 7, neutral: 1 / 7, fear: 1 / 7, surprise: 1 / 7, disgust: 1 / 7
      },
      faces_detected: result.faces_detected || 0,
      quality_warning: result.quality_warning,  // Pass through quality warning
      detected_accent: null,
      audio_weight: 0.0,
      text_weight: 0.0,
      visual_weight: 1.0,
      audio_score: 0.0,
      text_score: 0.0,
      visual_score: result.confidence,
    };
  };

  /**
   * Send multiple inputs to the multimodal fusion endpoint.
   */
  const analyzeMultimodal = async (): Promise<any> => {
    const formData = new FormData();
    if (mmAudioFile) formData.append('audio_file', mmAudioFile);
    if (mmVideoFile) formData.append('video_file', mmVideoFile);
    if (mmTextInput.trim()) formData.append('text', mmTextInput.trim());

    const response = await fetch(`${API_URL}/predict-multimodal`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Multimodal analysis failed');
    }

    const result = await response.json();
    return {
      emotion_label: result.emotion_label,
      confidence_score: result.confidence_score,
      all_probabilities: result.all_probabilities,
      quality_warning: result.quality_warning,
      detected_accent: result.audio_result?.detected_accent || null,
      fusion_method: result.fusion_method,
      modalities_used: result.modalities_used,
      audio_result: result.audio_result,
      text_result: result.text_result,
      video_result: result.video_result,
    };
  };

  const handleSubmit = async () => {
    if (!inputType) return;

    if (inputType === 'text' && !textInput.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    if ((inputType === 'audio' || inputType === 'video') && !file) {
      setError('Please upload a file');
      return;
    }

    // Multimodal validation: at least 2 inputs
    if (inputType === 'multimodal') {
      const count = [mmAudioFile, mmVideoFile, mmTextInput.trim()].filter(Boolean).length;
      if (count < 2) {
        setError('Please provide at least 2 types of input for multimodal analysis');
        return;
      }
    }

    setProcessing(true);
    setError(null);

    try {
      let modelResult;
      if (inputType === 'audio') {
        modelResult = await analyzeAudio(file!);
      } else if (inputType === 'video') {
        modelResult = await analyzeVideo(file!);
      } else if (inputType === 'multimodal') {
        modelResult = await analyzeMultimodal();
      } else {
        modelResult = await analyzeText(textInput);
      }

      console.log('✅ Analysis Result:', modelResult);

      saveResult({
        input_type: inputType as 'audio' | 'video' | 'text' | 'multimodal',
        filename: file?.name || mmAudioFile?.name || mmVideoFile?.name,
        emotion_label: modelResult.emotion_label,
        confidence_score: modelResult.confidence_score,
        all_probabilities: modelResult.all_probabilities,
        detected_accent: modelResult.detected_accent || null,
        quality_warning: modelResult.quality_warning,
        fusion_method: modelResult.fusion_method,
        modalities_used: modelResult.modalities_used,
        audio_result: modelResult.audio_result ? {
          emotion_label: modelResult.audio_result.emotion_label,
          confidence_score: modelResult.audio_result.confidence_score,
          weight: modelResult.audio_result.weight || 1.0,
          detected_accent: modelResult.audio_result.detected_accent,
        } : undefined,
        text_result: modelResult.text_result ? {
          emotion_label: modelResult.text_result.emotion_label,
          confidence_score: modelResult.text_result.confidence_score,
          weight: modelResult.text_result.weight || 1.0,
        } : undefined,
        video_result: modelResult.video_result ? {
          emotion_label: modelResult.video_result.emotion_label,
          confidence_score: modelResult.video_result.confidence_score,
          weight: modelResult.video_result.weight || 1.0,
        } : undefined,
      });

      setAnalysisResult(modelResult);
      setShowResultModal(true);
      setSuccess(true);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setProcessing(false);
    }
  };

  const handleReset = () => {
    setStep(1);
    setInputType(null);
    setTextInput('');
    setFile(null);
    setError(null);
    setSuccess(false);
    setMmAudioFile(null);
    setMmVideoFile(null);
    setMmTextInput('');
  };

  return (
    <>
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="animate-fade-in-up">
          <h2 className="text-xl sm:text-3xl font-bold text-white mb-2">Analyze Emotion</h2>
          <p className="text-xs sm:text-base" style={{ color: '#737373' }}>
            Submit audio, video, or text for multimodal emotion recognition
          </p>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center justify-center mb-8">
          <div className="flex items-center space-x-2">
            {[1, 2, 3].map((s) => (
              <div key={s} className="flex items-center">
                <div
                  className="flex items-center justify-center w-10 h-10 rounded-full font-semibold text-sm transition-all"
                  style={
                    step >= s
                      ? { background: 'linear-gradient(135deg, #06b6d4, #0891b2)', color: '#fff', boxShadow: '0 4px 15px rgba(6,182,212,0.3)' }
                      : { background: '#171717', color: '#525252', border: '1px solid rgba(255,255,255,0.06)' }
                  }
                >
                  {s}
                </div>
                {s < 3 && (
                  <div
                    className="h-0.5 w-8 sm:w-20 mx-1 sm:mx-2 rounded-full transition-all"
                    style={{ background: step > s ? '#06b6d4' : '#1f1f1f' }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {step === 1 && (
          <div className="animate-fade-in-up">
            <h3 className="text-xl font-bold text-white mb-6">Step 1: Select Input Type</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                { type: 'audio' as InputType, icon: Mic, label: 'Audio', desc: 'Upload audio for emotion analysis', color: '#06b6d4' },
                { type: 'video' as InputType, icon: Video, label: 'Video', desc: 'Upload video with audio & visual cues', color: '#3b82f6' },
                { type: 'text' as InputType, icon: FileText, label: 'Text', desc: 'Analyze text for emotional content', color: '#a855f7' },
                { type: 'multimodal' as InputType, icon: Layers, label: 'Multimodal', desc: 'Combine audio, text & video', color: '#10b981' },
              ].map(({ type, icon: Icon, label, desc, color }, idx) => (
                <button
                  key={type}
                  onClick={() => handleInputTypeSelect(type)}
                  className="relative overflow-hidden rounded-2xl p-6 text-center transition-all hover:scale-[1.03] group animate-fade-in-up"
                  style={{
                    background: '#111111',
                    border: `1px solid ${color}20`,
                    animationDelay: `${idx * 100}ms`,
                    animationFillMode: 'backwards',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = `${color}50`;
                    e.currentTarget.style.boxShadow = `0 8px 30px ${color}15`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = `${color}20`;
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div className="absolute top-0 left-0 right-0 h-px transition-opacity opacity-0 group-hover:opacity-100" style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)` }} />
                  <div className="p-3 rounded-xl mx-auto w-fit mb-4 transition-transform group-hover:scale-110" style={{ background: `${color}15` }}>
                    <Icon className="w-8 h-8" style={{ color }} />
                  </div>
                  <h4 className="font-bold text-lg text-white mb-1">{label}</h4>
                  <p className="text-xs" style={{ color: '#737373' }}>{desc}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {step === 2 && inputType && (
          <div className="animate-fade-in-up">
            <h3 className="text-xl font-semibold text-white mb-4">Step 2: Provide Input</h3>
            <div className="rounded-2xl p-4 sm:p-6" style={{ background: '#111111', border: '1px solid rgba(6,182,212,0.1)' }}>
              {inputType === 'multimodal' ? (
                <div className="space-y-4">
                  <p className="text-sm" style={{ color: '#737373' }}>Provide at least 2 types of input for fusion analysis</p>

                  {/* ── Audio Upload Card ── */}
                  <div
                    className="relative overflow-hidden rounded-2xl p-5 transition-all group animate-fade-in-up"
                    style={{
                      background: mmAudioDrag ? 'rgba(6,182,212,0.08)' : 'rgba(6,182,212,0.03)',
                      border: mmAudioDrag ? '2px dashed rgba(6,182,212,0.5)' : '1px solid rgba(6,182,212,0.1)',
                      animationDelay: '50ms', animationFillMode: 'backwards',
                      boxShadow: mmAudioDrag ? '0 0 30px rgba(6,182,212,0.12)' : 'none',
                    }}
                    onDragEnter={e => { e.preventDefault(); setMmAudioDrag(true); }}
                    onDragOver={e => e.preventDefault()}
                    onDragLeave={e => { e.preventDefault(); setMmAudioDrag(false); }}
                    onDrop={e => handleMmDrop(e, 'audio')}
                    onMouseEnter={e => { if (!mmAudioDrag) { e.currentTarget.style.borderColor = 'rgba(6,182,212,0.25)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(6,182,212,0.08)'; } }}
                    onMouseLeave={e => { if (!mmAudioDrag) { e.currentTarget.style.borderColor = 'rgba(6,182,212,0.1)'; e.currentTarget.style.boxShadow = 'none'; } }}
                  >
                    <div className="absolute top-0 left-0 right-0 h-0.5 opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: 'linear-gradient(90deg, transparent, #06b6d4, transparent)' }} />
                    <div className="flex items-center gap-4">
                      <div className="p-3 rounded-xl transition-transform group-hover:scale-110" style={{ background: 'rgba(6,182,212,0.1)' }}>
                        <Mic className="w-5 h-5" style={{ color: '#06b6d4' }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-white">Audio File</p>
                        <p className="text-xs" style={{ color: '#525252' }}>Drag & drop or browse · MP3, WAV, OGG</p>
                      </div>
                      <input type="file" aria-label="Upload Audio File" accept="audio/*" onChange={e => setMmAudioFile(e.target.files?.[0] || null)} className="hidden" id="mm-audio" />
                      <label
                        htmlFor="mm-audio"
                        className="px-4 py-2 rounded-xl text-xs font-semibold cursor-pointer transition-all"
                        style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.15), rgba(6,182,212,0.08))', border: '1px solid rgba(6,182,212,0.2)', color: '#22d3ee' }}
                        onMouseEnter={e => { (e.target as HTMLElement).style.background = 'linear-gradient(135deg, rgba(6,182,212,0.25), rgba(6,182,212,0.15))'; (e.target as HTMLElement).style.boxShadow = '0 4px 15px rgba(6,182,212,0.2)'; }}
                        onMouseLeave={e => { (e.target as HTMLElement).style.background = 'linear-gradient(135deg, rgba(6,182,212,0.15), rgba(6,182,212,0.08))'; (e.target as HTMLElement).style.boxShadow = 'none'; }}
                      >
                        {mmAudioFile ? '✓ Change' : 'Browse'}
                      </label>
                    </div>
                    {mmAudioFile && (
                      <div className="mt-3 flex items-center gap-2 pl-14">
                        <CheckCircle className="w-3.5 h-3.5" style={{ color: '#10b981' }} />
                        <span className="text-xs" style={{ color: '#10b981' }}>{mmAudioFile.name}</span>
                      </div>
                    )}
                  </div>

                  {/* ── Text Input Card ── */}
                  <div
                    className="relative overflow-hidden rounded-2xl p-5 transition-all group animate-fade-in-up"
                    style={{ background: 'rgba(168,85,247,0.03)', border: '1px solid rgba(168,85,247,0.1)', animationDelay: '150ms', animationFillMode: 'backwards' }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(168,85,247,0.25)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(168,85,247,0.08)'; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(168,85,247,0.1)'; e.currentTarget.style.boxShadow = 'none'; }}
                  >
                    <div className="absolute top-0 left-0 right-0 h-0.5 opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: 'linear-gradient(90deg, transparent, #a855f7, transparent)' }} />
                    <div className="flex items-center gap-4 mb-3">
                      <div className="p-3 rounded-xl transition-transform group-hover:scale-110" style={{ background: 'rgba(168,85,247,0.1)' }}>
                        <FileText className="w-5 h-5" style={{ color: '#a855f7' }} />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-white">Text Input</p>
                        <p className="text-xs" style={{ color: '#525252' }}>Type or paste · optional</p>
                      </div>
                    </div>
                    <textarea
                      aria-label="Enter textual input for analysis"
                      value={mmTextInput}
                      onChange={e => setMmTextInput(e.target.value)}
                      rows={2}
                      className="w-full px-4 py-3 rounded-xl text-sm focus:outline-none transition-all"
                      style={{ background: '#0a0a0a', border: '1px solid rgba(168,85,247,0.12)', color: '#e5e5e5', resize: 'none', overflow: 'hidden', minHeight: '64px', maxHeight: '200px' }}
                      placeholder="Type text here..."
                      onInput={e => { const t = e.currentTarget; t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 200) + 'px'; if (t.scrollHeight > 200) t.style.overflow = 'auto'; else t.style.overflow = 'hidden'; }}
                      onFocus={e => { e.currentTarget.style.borderColor = '#a855f7'; e.currentTarget.style.boxShadow = '0 0 0 3px rgba(168,85,247,0.1)'; }}
                      onBlur={e => { e.currentTarget.style.borderColor = 'rgba(168,85,247,0.12)'; e.currentTarget.style.boxShadow = 'none'; }}
                    />
                    {mmTextInput.length > 0 && (
                      <div className="flex justify-end mt-1.5">
                        <span className="text-[10px]" style={{ color: '#a855f7' }}>{mmTextInput.length} chars</span>
                      </div>
                    )}
                  </div>

                  {/* ── Video/Image Upload Card ── */}
                  <div
                    className="relative overflow-hidden rounded-2xl p-5 transition-all group animate-fade-in-up"
                    style={{
                      background: mmVideoDrag ? 'rgba(59,130,246,0.08)' : 'rgba(59,130,246,0.03)',
                      border: mmVideoDrag ? '2px dashed rgba(59,130,246,0.5)' : '1px solid rgba(59,130,246,0.1)',
                      animationDelay: '250ms', animationFillMode: 'backwards',
                      boxShadow: mmVideoDrag ? '0 0 30px rgba(59,130,246,0.12)' : 'none',
                    }}
                    onDragEnter={e => { e.preventDefault(); setMmVideoDrag(true); }}
                    onDragOver={e => e.preventDefault()}
                    onDragLeave={e => { e.preventDefault(); setMmVideoDrag(false); }}
                    onDrop={e => handleMmDrop(e, 'video')}
                    onMouseEnter={e => { if (!mmVideoDrag) { e.currentTarget.style.borderColor = 'rgba(59,130,246,0.25)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(59,130,246,0.08)'; } }}
                    onMouseLeave={e => { if (!mmVideoDrag) { e.currentTarget.style.borderColor = 'rgba(59,130,246,0.1)'; e.currentTarget.style.boxShadow = 'none'; } }}
                  >
                    <div className="absolute top-0 left-0 right-0 h-0.5 opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: 'linear-gradient(90deg, transparent, #3b82f6, transparent)' }} />
                    <div className="flex items-center gap-4">
                      <div className="p-3 rounded-xl transition-transform group-hover:scale-110" style={{ background: 'rgba(59,130,246,0.1)' }}>
                        <Video className="w-5 h-5" style={{ color: '#3b82f6' }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-white">Image / Video</p>
                        <p className="text-xs" style={{ color: '#525252' }}>Drag & drop or browse · MP4, JPG, PNG</p>
                      </div>
                      <input type="file" aria-label="Upload Video or Image File" accept="video/*,image/*" onChange={e => setMmVideoFile(e.target.files?.[0] || null)} className="hidden" id="mm-video" />
                      <label
                        htmlFor="mm-video"
                        className="px-4 py-2 rounded-xl text-xs font-semibold cursor-pointer transition-all"
                        style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.08))', border: '1px solid rgba(59,130,246,0.2)', color: '#60a5fa' }}
                        onMouseEnter={e => { (e.target as HTMLElement).style.background = 'linear-gradient(135deg, rgba(59,130,246,0.25), rgba(59,130,246,0.15))'; (e.target as HTMLElement).style.boxShadow = '0 4px 15px rgba(59,130,246,0.2)'; }}
                        onMouseLeave={e => { (e.target as HTMLElement).style.background = 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.08))'; (e.target as HTMLElement).style.boxShadow = 'none'; }}
                      >
                        {mmVideoFile ? '✓ Change' : 'Browse'}
                      </label>
                    </div>
                    {mmVideoFile && (
                      <div className="mt-3 flex items-center gap-2 pl-14">
                        <CheckCircle className="w-3.5 h-3.5" style={{ color: '#10b981' }} />
                        <span className="text-xs" style={{ color: '#10b981' }}>{mmVideoFile.name}</span>
                      </div>
                    )}
                  </div>
                </div>
              ) : inputType === 'text' ? (
                <div className="animate-fade-in-up">
                  <label className="block text-sm font-medium mb-3 flex items-center gap-2" style={{ color: '#a3a3a3' }}>
                    <FileText className="w-4 h-4" style={{ color: '#a855f7' }} />
                    Enter text to analyze
                  </label>
                  <textarea
                    aria-label="Enter text input for emotion analysis"
                    value={textInput}
                    onChange={e => setTextInput(e.target.value)}
                    rows={3}
                    className="w-full px-4 py-3 rounded-xl focus:outline-none transition-all text-sm"
                    style={{ background: '#0a0a0a', border: '1px solid rgba(168,85,247,0.15)', color: '#e5e5e5', lineHeight: '1.7', resize: 'none', overflow: 'hidden', minHeight: '100px', maxHeight: '300px' }}
                    placeholder="Type or paste your text here for emotion analysis..."
                    onInput={e => { const t = e.currentTarget; t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 300) + 'px'; if (t.scrollHeight > 300) t.style.overflow = 'auto'; else t.style.overflow = 'hidden'; }}
                    onFocus={e => { e.currentTarget.style.borderColor = '#a855f7'; e.currentTarget.style.boxShadow = '0 0 0 3px rgba(168,85,247,0.1)'; }}
                    onBlur={e => { e.currentTarget.style.borderColor = 'rgba(168,85,247,0.15)'; e.currentTarget.style.boxShadow = 'none'; }}
                  />
                  <div className="flex justify-end mt-2">
                    <span className="text-xs" style={{ color: textInput.length > 0 ? '#a855f7' : '#404040' }}>{textInput.length} characters</span>
                  </div>
                </div>
              ) : (
                /* ─── Audio / Video DRAG-AND-DROP Zone ─── */
                <label
                  htmlFor="file-upload"
                  className="relative block rounded-2xl text-center transition-all cursor-pointer group overflow-hidden animate-fade-in-up"
                  style={{
                    padding: isDragging ? '3rem 2rem' : '2.5rem 2rem',
                    background: isDragging ? 'rgba(6,182,212,0.08)' : 'rgba(6,182,212,0.02)',
                    border: isDragging ? '2px dashed #06b6d4' : '2px dashed rgba(6,182,212,0.15)',
                    boxShadow: isDragging ? '0 0 40px rgba(6,182,212,0.15), inset 0 0 40px rgba(6,182,212,0.04)' : 'none',
                  }}
                  onDragEnter={handleDragEnter}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onMouseEnter={e => { if (!isDragging) { e.currentTarget.style.borderColor = 'rgba(6,182,212,0.35)'; e.currentTarget.style.background = 'rgba(6,182,212,0.04)'; } }}
                  onMouseLeave={e => { if (!isDragging) { e.currentTarget.style.borderColor = 'rgba(6,182,212,0.15)'; e.currentTarget.style.background = 'rgba(6,182,212,0.02)'; } }}
                >
                  {/* Animated radial glow */}
                  <div
                    className="absolute inset-0 transition-opacity pointer-events-none"
                    style={{
                      opacity: isDragging ? 1 : 0,
                      background: 'radial-gradient(circle at center, rgba(6,182,212,0.1), transparent 70%)',
                      animation: isDragging ? 'pulse 1.5s ease-in-out infinite' : 'none',
                    }}
                  />
                  {/* Hover shimmer */}
                  <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" style={{ background: 'radial-gradient(circle at center, rgba(6,182,212,0.05), transparent 70%)' }} />

                  <div className="relative flex flex-col items-center">
                    <input
                      type="file"
                      aria-label="Upload general media file"
                      onChange={handleFileChange}
                      accept={inputType === 'audio' ? 'audio/*' : 'video/*,image/*'}
                      className="hidden"
                      id="file-upload"
                    />

                    {file ? (
                      /* ── File Selected State ── */
                      <div className="flex flex-col items-center animate-fade-in-up">
                        <div className="p-5 rounded-2xl mb-4" style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.2)' }}>
                          <CheckCircle className="w-8 h-8" style={{ color: '#10b981' }} />
                        </div>
                        <p className="font-semibold text-white mb-1">{file.name}</p>
                        <p className="text-xs mb-4" style={{ color: '#6b7280' }}>
                          {(file.size / (1024 * 1024)).toFixed(2)} MB · {file.type.split('/')[1]?.toUpperCase() || 'FILE'}
                        </p>
                        <span
                          className="inline-flex items-center gap-1.5 text-xs font-medium cursor-pointer transition-all px-3 py-1.5 rounded-lg"
                          style={{ color: '#06b6d4', background: 'rgba(6,182,212,0.08)', border: '1px solid rgba(6,182,212,0.15)' }}
                        >
                          <Upload className="w-3 h-3" />
                          Change File
                        </span>
                      </div>
                    ) : isDragging ? (
                      /* ── Dragging State ── */
                      <>
                        <div
                          className="p-5 rounded-2xl mb-5"
                          style={{ background: 'rgba(6,182,212,0.15)', border: '1px solid rgba(6,182,212,0.12)', animation: 'float 2s ease-in-out infinite' }}
                        >
                          <Upload className="w-8 h-8" style={{ color: '#06b6d4' }} />
                        </div>
                        <p className="font-semibold text-lg" style={{ color: '#22d3ee' }}>Drop it here!</p>
                      </>
                    ) : (
                      /* ── Default Upload State ── */
                      <>
                        <div
                          className="p-5 rounded-2xl mb-5 transition-all group-hover:scale-110"
                          style={{ background: 'rgba(6,182,212,0.07)', border: '1px solid rgba(6,182,212,0.12)' }}
                        >
                          <Upload className="w-8 h-8" style={{ color: '#06b6d4' }} />
                        </div>
                        <p className="font-semibold text-white mb-1">
                          Drag & drop your file here
                        </p>
                        <p className="text-sm mb-3" style={{ color: '#525252' }}>
                          or
                        </p>
                        <span
                          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all"
                          style={{
                            background: 'linear-gradient(135deg, rgba(6,182,212,0.15), rgba(6,182,212,0.08))',
                            border: '1px solid rgba(6,182,212,0.2)',
                            color: '#22d3ee',
                          }}
                        >
                          <Upload className="w-4 h-4" />
                          Browse Files
                        </span>
                        <p className="text-xs mt-3" style={{ color: '#404040' }}>
                          {inputType === 'audio' ? 'Supports MP3, WAV, OGG · Max 25MB' : 'Supports MP4, MOV, WEBM, JPG, PNG · Max 25MB'}
                        </p>
                      </>
                    )}
                  </div>
                </label>
              )}

              {error && (
                <div className="mt-4 flex items-center space-x-2 p-3 rounded-lg" style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)', color: '#fb7185' }}>
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span className="text-sm">{error}</span>
                </div>
              )}

              <div className="flex space-x-4 mt-6">
                <button
                  onClick={handleReset}
                  className="flex-1 px-6 py-3 rounded-xl font-medium text-sm transition-all"
                  style={{ border: '1px solid rgba(255,255,255,0.08)', color: '#a3a3a3', background: 'transparent' }}
                >
                  Back
                </button>
                <button
                  onClick={() => setStep(3)}
                  disabled={
                    (inputType === 'text' && !textInput.trim()) ||
                    ((inputType === 'audio' || inputType === 'video') && !file) ||
                    (inputType === 'multimodal' && [mmAudioFile, mmVideoFile, mmTextInput.trim()].filter(Boolean).length < 2)
                  }
                  className="flex-1 px-6 py-3 rounded-xl text-white font-medium text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{ background: 'linear-gradient(135deg, #06b6d4, #0891b2)', boxShadow: '0 4px 15px rgba(6,182,212,0.3)' }}
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="animate-fade-in-up">
            <h3 className="text-xl font-semibold text-white mb-4">Step 3: Review & Submit</h3>
            <div className="rounded-xl p-6" style={{ background: '#111111', border: '1px solid rgba(6,182,212,0.1)' }}>
              <div className="space-y-3 mb-6">
                <div className="flex justify-between items-center py-2" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                  <span style={{ color: '#737373' }}>Input Type</span>
                  <span className="font-medium text-white capitalize">{inputType}</span>
                </div>
                {inputType === 'text' && (
                  <div className="flex justify-between items-center py-2" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <span style={{ color: '#737373' }}>Text Length</span>
                    <span className="font-medium text-white">{textInput.length} characters</span>
                  </div>
                )}
                {file && (
                  <div className="flex justify-between items-center py-2" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <span style={{ color: '#737373' }}>File</span>
                    <span className="font-medium text-white">{file.name}</span>
                  </div>
                )}
                {inputType === 'multimodal' && (
                  <div className="flex justify-between items-center py-2">
                    <span style={{ color: '#737373' }}>Inputs</span>
                    <span className="font-medium text-white">
                      {[mmAudioFile && '🎤', mmTextInput.trim() && '📝', mmVideoFile && '📹'].filter(Boolean).join(' + ')}
                    </span>
                  </div>
                )}
              </div>

              {success ? (
                <div className="flex items-center justify-center space-x-3 py-4" style={{ color: '#10b981' }}>
                  <CheckCircle className="w-6 h-6" />
                  <span className="font-medium">Analysis complete!</span>
                </div>
              ) : (
                <div className="flex space-x-4">
                  <button
                    onClick={() => setStep(2)}
                    disabled={processing}
                    className="flex-1 px-6 py-3 rounded-xl font-medium text-sm transition-all disabled:opacity-40"
                    style={{ border: '1px solid rgba(255,255,255,0.08)', color: '#a3a3a3' }}
                  >
                    Back
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={processing}
                    className="flex-1 px-6 py-3 rounded-xl text-white font-medium text-sm transition-all disabled:opacity-40 flex items-center justify-center space-x-2"
                    style={{ background: 'linear-gradient(135deg, #06b6d4, #0891b2)', boxShadow: '0 4px 15px rgba(6,182,212,0.3)' }}
                  >
                    {processing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <span>Submit Analysis</span>
                    )}
                  </button>
                </div>
              )}

              {error && (
                <div className="mt-4 flex items-center space-x-2 p-3 rounded-lg" style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)', color: '#fb7185' }}>
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span className="text-sm">{error}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Result Modal */}
      {
        showResultModal && analysisResult && (
          <ResultModal
            result={analysisResult}
            onNavigate={onNavigate}
            onClose={() => {
              setShowResultModal(false);
              setSuccess(false);
              setStep(1);
              setInputType(null);
              setFile(null);
              setMmAudioFile(null);
              setMmVideoFile(null);
              setMmTextInput('');
            }}
          />
        )
      }
    </>
  );
}
