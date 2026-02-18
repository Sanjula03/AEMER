import { useState } from 'react';
import { Mic, Video, FileText, Upload, Loader2, CheckCircle, AlertCircle, Layers } from 'lucide-react';
import { ResultModal } from '../components/ResultModal';
import { saveResult } from '../lib/storage';

interface AnalyzeProps {
  onNavigate: (page: string) => void;
}

type InputType = 'audio' | 'video' | 'text' | 'multimodal' | null;
type Step = 1 | 2 | 3;

export function Analyze({ onNavigate: _onNavigate }: AnalyzeProps) {
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
        angry: 0.25, happy: 0.25, sad: 0.25, neutral: 0.25
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
        input_type: inputType as 'audio' | 'video' | 'text',
        filename: file?.name || mmAudioFile?.name || mmVideoFile?.name,
        emotion_label: modelResult.emotion_label,
        confidence_score: modelResult.confidence_score,
        all_probabilities: modelResult.all_probabilities,
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
        <div>
          <h2 className="text-3xl font-bold text-amber-100 mb-2">Analyze Emotion</h2>
          <p className="text-amber-200/70">
            Submit audio, video, or text for multimodal emotion recognition
          </p>
        </div>

        <div className="flex items-center justify-center mb-8">
          <div className="flex items-center space-x-4">
            <div className={`flex items-center justify-center w-10 h-10 rounded-full ${step >= 1 ? 'bg-teal-600 text-white' : 'bg-gray-200 text-gray-500'
              }`}>
              1
            </div>
            <div className={`h-1 w-24 ${step >= 2 ? 'bg-teal-600' : 'bg-gray-200'}`} />
            <div className={`flex items-center justify-center w-10 h-10 rounded-full ${step >= 2 ? 'bg-teal-600 text-white' : 'bg-gray-200 text-gray-500'
              }`}>
              2
            </div>
            <div className={`h-1 w-24 ${step >= 3 ? 'bg-teal-600' : 'bg-gray-200'}`} />
            <div className={`flex items-center justify-center w-10 h-10 rounded-full ${step >= 3 ? 'bg-teal-600 text-white' : 'bg-gray-200 text-gray-500'
              }`}>
              3
            </div>
          </div>
        </div>

        {step === 1 && (
          <div>
            <h3 className="text-2xl font-bold text-amber-100 mb-6">Step 1: Select Input Type</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <button
                onClick={() => handleInputTypeSelect('audio')}
                className="relative overflow-hidden bg-gradient-to-br from-teal-400 to-emerald-500 text-white rounded-2xl p-8 hover:scale-105 hover:shadow-xl shadow-lg shadow-teal-500/30 transition-all group"
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -translate-y-8 translate-x-8" />
                <Mic className="w-14 h-14 mx-auto mb-4 group-hover:scale-110 transition-transform" />
                <h4 className="font-bold text-xl mb-2">Audio</h4>
                <p className="text-white/80">
                  Upload audio for emotion analysis
                </p>
              </button>

              <button
                onClick={() => handleInputTypeSelect('video')}
                className="relative overflow-hidden bg-gradient-to-br from-blue-400 to-indigo-500 text-white rounded-2xl p-8 hover:scale-105 hover:shadow-xl shadow-lg shadow-blue-500/30 transition-all group"
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -translate-y-8 translate-x-8" />
                <Video className="w-14 h-14 mx-auto mb-4 group-hover:scale-110 transition-transform" />
                <h4 className="font-bold text-xl mb-2">Video</h4>
                <p className="text-white/80">
                  Upload video with audio & visual cues
                </p>
              </button>

              <button
                onClick={() => handleInputTypeSelect('text')}
                className="relative overflow-hidden bg-gradient-to-br from-purple-400 to-pink-500 text-white rounded-2xl p-8 hover:scale-105 hover:shadow-xl shadow-lg shadow-purple-500/30 transition-all group"
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -translate-y-8 translate-x-8" />
                <FileText className="w-14 h-14 mx-auto mb-4 group-hover:scale-110 transition-transform" />
                <h4 className="font-bold text-xl mb-2">Text</h4>
                <p className="text-white/80">
                  Analyze text for emotional content
                </p>
              </button>

              <button
                onClick={() => handleInputTypeSelect('multimodal')}
                className="relative overflow-hidden bg-gradient-to-br from-amber-400 to-orange-500 text-white rounded-2xl p-8 hover:scale-105 hover:shadow-xl shadow-lg shadow-amber-500/30 transition-all group"
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -translate-y-8 translate-x-8" />
                <Layers className="w-14 h-14 mx-auto mb-4 group-hover:scale-110 transition-transform" />
                <h4 className="font-bold text-xl mb-2">Multimodal</h4>
                <p className="text-white/80">
                  Combine audio, text & video
                </p>
              </button>
            </div>
          </div>
        )}

        {step === 2 && inputType && (
          <div>
            <h3 className="text-xl font-semibold text-amber-100 mb-4">Step 2: Provide Input</h3>
            <div className="bg-stone-800/50 border border-amber-900/30 rounded-xl p-8">
              {inputType === 'multimodal' ? (
                <div className="space-y-6">
                  <p className="text-amber-200/70 text-sm">Provide at least 2 types of input for fusion analysis</p>

                  {/* Audio Upload */}
                  <div className="border border-amber-900/30 rounded-lg p-4">
                    <label className="block text-sm font-medium text-teal-400 mb-2">🎤 Audio (optional)</label>
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => setMmAudioFile(e.target.files?.[0] || null)}
                      className="text-sm text-amber-200 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-teal-600 file:text-white hover:file:bg-teal-700"
                    />
                    {mmAudioFile && <p className="mt-1 text-xs text-amber-200/60">✅ {mmAudioFile.name}</p>}
                  </div>

                  {/* Text Input */}
                  <div className="border border-amber-900/30 rounded-lg p-4">
                    <label className="block text-sm font-medium text-purple-400 mb-2">📝 Text (optional)</label>
                    <textarea
                      value={mmTextInput}
                      onChange={(e) => setMmTextInput(e.target.value)}
                      rows={2}
                      className="w-full px-4 py-2 bg-stone-700/50 border border-amber-900/30 rounded-lg text-amber-100 focus:ring-2 focus:ring-teal-500 text-sm"
                      placeholder="Type text here..."
                    />
                  </div>

                  {/* Video/Image Upload */}
                  <div className="border border-amber-900/30 rounded-lg p-4">
                    <label className="block text-sm font-medium text-blue-400 mb-2">📹 Image/Video (optional)</label>
                    <input
                      type="file"
                      accept="video/*,image/*"
                      onChange={(e) => setMmVideoFile(e.target.files?.[0] || null)}
                      className="text-sm text-amber-200 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                    />
                    {mmVideoFile && <p className="mt-1 text-xs text-amber-200/60">✅ {mmVideoFile.name}</p>}
                  </div>
                </div>
              ) : inputType === 'text' ? (
                <div>
                  <label className="block text-sm font-medium text-amber-200 mb-2">
                    Enter text to analyze
                  </label>
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-3 bg-stone-700/50 border border-amber-900/30 rounded-lg text-amber-100 focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                    placeholder="Type or paste text here..."
                  />
                </div>
              ) : (
                <div className="border-2 border-dashed border-amber-900/30 rounded-xl p-8 text-center">
                  <div className="flex flex-col items-center">
                    <Upload className="w-12 h-12 text-amber-400 mx-auto mb-4" />
                    <input
                      type="file"
                      onChange={handleFileChange}
                      accept={inputType === 'audio' ? 'audio/*' : 'video/*,image/*'}
                      className="hidden"
                      id="file-upload"
                    />
                    <label
                      htmlFor="file-upload"
                      className="cursor-pointer text-teal-600 hover:text-teal-700 font-medium"
                    >
                      Choose file
                    </label>
                    {file && (
                      <p className="mt-2 text-sm text-gray-600">Selected: {file.name}</p>
                    )}
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-4 flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
                  <AlertCircle className="w-5 h-5" />
                  <span className="text-sm">{error}</span>
                </div>
              )}

              <div className="flex space-x-4 mt-6">
                <button
                  onClick={handleReset}
                  className="flex-1 px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
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
                  className="flex-1 px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div>
            <h3 className="text-xl font-semibold text-amber-100 mb-4">Step 3: Review & Submit</h3>
            <div className="bg-stone-800/50 border border-amber-900/30 rounded-xl p-8">
              <div className="space-y-4 mb-6">
                <div className="flex justify-between">
                  <span className="text-amber-200/70">Input Type:</span>
                  <span className="font-medium text-amber-100 capitalize">{inputType}</span>
                </div>
                {inputType === 'text' && (
                  <div className="flex justify-between">
                    <span className="text-amber-200/70">Text Length:</span>
                    <span className="font-medium text-amber-100">{textInput.length} characters</span>
                  </div>
                )}
                {file && (
                  <div className="flex justify-between">
                    <span className="text-amber-200/70">File:</span>
                    <span className="font-medium text-amber-100">{file.name}</span>
                  </div>
                )}
              </div>

              {success ? (
                <div className="flex items-center justify-center space-x-3 text-green-400 py-4">
                  <CheckCircle className="w-6 h-6" />
                  <span className="font-medium">Analysis complete! Redirecting...</span>
                </div>
              ) : (
                <div className="flex space-x-4">
                  <button
                    onClick={() => setStep(2)}
                    disabled={processing}
                    className="flex-1 px-6 py-3 border border-amber-700 text-amber-200 rounded-xl hover:bg-amber-900/30 transition-colors disabled:opacity-50">
                    Back
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={processing}
                    className="flex-1 px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors disabled:opacity-50 flex items-center justify-center space-x-2"
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
                <div className="mt-4 flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
                  <AlertCircle className="w-5 h-5" />
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
