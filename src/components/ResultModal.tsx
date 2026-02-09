import { X, AlertTriangle } from 'lucide-react';

interface EmotionResult {
    emotion_label: string;
    confidence_score: number;
    all_probabilities?: Record<string, number>;
    quality_warning?: string;
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
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/50 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-in fade-in zoom-in duration-300">
                {/* Header with gradient */}
                <div
                    className={`bg-gradient-to-r ${getEmotionColor(result.emotion_label)} p-6 text-white text-center`}
                >
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 p-1 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>

                    <div className="text-6xl mb-3">{getEmotionEmoji(result.emotion_label)}</div>
                    <h2 className="text-3xl font-bold capitalize">{result.emotion_label}</h2>
                    <p className="text-white/80 mt-1">Detected Emotion</p>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                    {/* Quality Warning */}
                    {result.quality_warning && (
                        <div className="flex items-start space-x-2 bg-amber-50 border border-amber-200 rounded-lg p-3">
                            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-amber-800">{result.quality_warning}</p>
                        </div>
                    )}

                    {/* Confidence Score */}
                    <div className="text-center">
                        <div className="text-sm text-gray-500 mb-2">Confidence Score</div>
                        <div className={`text-4xl font-bold ${getConfidenceColor(result.confidence_score)}`}>
                            {(result.confidence_score * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 mt-3">
                            <div
                                className={`bg-gradient-to-r ${getEmotionColor(result.emotion_label)} h-3 rounded-full transition-all duration-500`}
                                style={{ width: `${result.confidence_score * 100}%` }}
                            />
                        </div>
                    </div>

                    {/* All Probabilities */}
                    {result.all_probabilities && Object.keys(result.all_probabilities).length > 0 && (
                        <div>
                            <div className="text-sm text-gray-500 mb-3 text-center">All Emotion Probabilities</div>
                            <div className="space-y-2">
                                {Object.entries(result.all_probabilities)
                                    .sort(([, a], [, b]) => b - a)
                                    .map(([emotion, probability]) => (
                                        <div key={emotion} className="flex items-center justify-between">
                                            <div className="flex items-center space-x-2">
                                                <span className="text-xl">{getEmotionEmoji(emotion)}</span>
                                                <span className="text-gray-700 capitalize font-medium">{emotion}</span>
                                            </div>
                                            <div className="flex items-center space-x-3">
                                                <div className="w-24 bg-gray-200 rounded-full h-2">
                                                    <div
                                                        className={`bg-gradient-to-r ${getEmotionColor(emotion)} h-2 rounded-full`}
                                                        style={{ width: `${probability * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-sm font-medium text-gray-600 w-14 text-right">
                                                    {(probability * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                            </div>
                        </div>
                    )}

                    {/* Close Button */}
                    <button
                        onClick={onClose}
                        className="w-full py-3 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors"
                    >
                        Analyze Another
                    </button>
                </div>
            </div>
        </div>
    );
}
