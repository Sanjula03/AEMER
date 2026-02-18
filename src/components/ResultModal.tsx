import { X, AlertTriangle } from 'lucide-react';

interface ModalityResult {
    emotion_label: string;
    confidence_score: number;
    all_probabilities: Record<string, number>;
    weight: number;
}

interface EmotionResult {
    emotion_label: string;
    confidence_score: number;
    all_probabilities?: Record<string, number>;
    quality_warning?: string;
    fusion_method?: string;
    modalities_used?: string[];
    audio_result?: ModalityResult;
    text_result?: ModalityResult;
    video_result?: ModalityResult;
}

interface ResultModalProps {
    result: EmotionResult;
    onClose: () => void;
}

export function ResultModal({ result, onClose }: ResultModalProps) {
    const getEmotionEmoji = (emotion: string) => {
        const emojis: Record<string, string> = {
            happy: '😊',
            sad: '😢',
            angry: '😠',
            neutral: '😐',
            fear: '😨',
            surprise: '😲',
            disgust: '🤢',
        };
        return emojis[emotion.toLowerCase()] || '🎭';
    };

    const getEmotionColor = (emotion: string) => {
        const colors: Record<string, string> = {
            happy: 'from-yellow-400 to-orange-500',
            sad: 'from-blue-400 to-blue-600',
            angry: 'from-red-400 to-red-600',
            neutral: 'from-gray-400 to-gray-600',
            fear: 'from-purple-400 to-purple-600',
            surprise: 'from-pink-400 to-pink-600',
            disgust: 'from-green-400 to-green-600',
        };
        return colors[emotion.toLowerCase()] || 'from-teal-400 to-teal-600';
    };

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 0.8) return 'text-green-600';
        if (confidence >= 0.6) return 'text-yellow-600';
        return 'text-orange-600';
    };

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            {/* Backdrop - clicking closes modal */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal - stopPropagation prevents backdrop close when clicking modal */}
            <div
                className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md max-h-[85vh] flex flex-col overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header with gradient */}
                <div
                    className={`relative bg-gradient-to-r ${getEmotionColor(result.emotion_label)} p-6 text-white text-center shrink-0`}
                >
                    <button
                        onClick={onClose}
                        className="absolute top-3 right-3 p-2 rounded-full bg-white/20 hover:bg-white/40 transition-colors cursor-pointer z-10"
                    >
                        <X className="w-5 h-5" />
                    </button>

                    <div className="text-5xl mb-2">{getEmotionEmoji(result.emotion_label)}</div>
                    <h2 className="text-2xl font-bold capitalize">{result.emotion_label}</h2>
                    <p className="text-white/80 text-sm mt-1">Detected Emotion</p>
                </div>

                {/* Scrollable Content */}
                <div className="p-5 space-y-5 overflow-y-auto flex-1 min-h-0">
                    {/* Quality Warning */}
                    {result.quality_warning && (
                        <div className="flex items-start space-x-2 bg-amber-50 border border-amber-200 rounded-lg p-3">
                            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-amber-800">{result.quality_warning}</p>
                        </div>
                    )}

                    {/* Confidence Score */}
                    <div className="text-center">
                        <div className="text-sm text-gray-500 mb-1">Confidence Score</div>
                        <div className={`text-3xl font-bold ${getConfidenceColor(result.confidence_score)}`}>
                            {(result.confidence_score * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                            <div
                                className={`bg-gradient-to-r ${getEmotionColor(result.emotion_label)} h-2.5 rounded-full transition-all duration-500`}
                                style={{ width: `${result.confidence_score * 100}%` }}
                            />
                        </div>
                    </div>

                    {/* All Probabilities */}
                    {result.all_probabilities && Object.keys(result.all_probabilities).length > 0 && (
                        <div>
                            <div className="text-sm text-gray-500 mb-2 text-center">All Emotion Probabilities</div>
                            <div className="space-y-1.5">
                                {Object.entries(result.all_probabilities)
                                    .sort(([, a], [, b]) => b - a)
                                    .map(([emotion, probability]) => (
                                        <div key={emotion} className="flex items-center justify-between">
                                            <div className="flex items-center space-x-2">
                                                <span className="text-lg">{getEmotionEmoji(emotion)}</span>
                                                <span className="text-gray-700 capitalize font-medium text-sm">{emotion}</span>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <div className="w-20 bg-gray-200 rounded-full h-2">
                                                    <div
                                                        className={`bg-gradient-to-r ${getEmotionColor(emotion)} h-2 rounded-full`}
                                                        style={{ width: `${probability * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-sm font-medium text-gray-600 w-12 text-right">
                                                    {(probability * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                            </div>
                        </div>
                    )}

                    {/* Multimodal Breakdown */}
                    {result.modalities_used && result.modalities_used.length > 1 && (
                        <div>
                            <div className="text-sm text-gray-500 mb-2 text-center">
                                Modality Breakdown ({result.fusion_method?.replace('_', ' ')})
                            </div>
                            <div className="space-y-1.5">
                                {result.audio_result && (
                                    <div className="flex items-center justify-between bg-teal-50 rounded-lg p-2">
                                        <span className="text-sm">🎤 Audio</span>
                                        <span className="text-sm font-medium capitalize">{result.audio_result.emotion_label}</span>
                                        <span className="text-xs text-gray-500">{(result.audio_result.confidence_score * 100).toFixed(0)}%</span>
                                        <span className="text-xs text-gray-400">w: {(result.audio_result.weight * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                                {result.text_result && (
                                    <div className="flex items-center justify-between bg-purple-50 rounded-lg p-2">
                                        <span className="text-sm">📝 Text</span>
                                        <span className="text-sm font-medium capitalize">{result.text_result.emotion_label}</span>
                                        <span className="text-xs text-gray-500">{(result.text_result.confidence_score * 100).toFixed(0)}%</span>
                                        <span className="text-xs text-gray-400">w: {(result.text_result.weight * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                                {result.video_result && (
                                    <div className="flex items-center justify-between bg-blue-50 rounded-lg p-2">
                                        <span className="text-sm">📹 Video</span>
                                        <span className="text-sm font-medium capitalize">{result.video_result.emotion_label}</span>
                                        <span className="text-xs text-gray-500">{(result.video_result.confidence_score * 100).toFixed(0)}%</span>
                                        <span className="text-xs text-gray-400">w: {(result.video_result.weight * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Fixed Close Button at bottom */}
                <div className="p-4 border-t border-gray-100 shrink-0">
                    <button
                        onClick={onClose}
                        className="w-full py-3 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors cursor-pointer"
                    >
                        Analyze Another
                    </button>
                </div>
            </div>
        </div>
    );
}
