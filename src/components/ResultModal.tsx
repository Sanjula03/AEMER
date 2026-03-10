import { createPortal } from 'react-dom';
import { useState, useEffect } from 'react';
import { X, AlertTriangle, Globe, Shield, ArrowRight, Sparkles, Loader2 } from 'lucide-react';
import {
    getMentalStateSummary,
    getWellbeingIndicator,
    DISCLAIMER_TEXT,
} from '../lib/emotionInsights';
import { fetchAIInsights } from '../lib/aiService';

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
    detected_accent?: string | null;
    fusion_method?: string;
    modalities_used?: string[];
    audio_result?: ModalityResult & { detected_accent?: string | null };
    text_result?: ModalityResult;
    video_result?: ModalityResult;
}

interface ResultModalProps {
    result: EmotionResult;
    onClose: () => void;
    onNavigate?: (page: string) => void;
}

export function ResultModal({ result, onClose, onNavigate }: ResultModalProps) {
    const [aiInsights, setAiInsights] = useState<string | null>(null);
    const [insightsLoading, setInsightsLoading] = useState(true);
    const [insightsSource, setInsightsSource] = useState<string>('loading');

    // Fetch AI insights when modal opens
    useEffect(() => {
        let cancelled = false;
        const loadInsights = async () => {
            try {
                const response = await fetchAIInsights({
                    emotion_label: result.emotion_label,
                    confidence_score: result.confidence_score,
                    all_probabilities: result.all_probabilities,
                    detected_accent: result.detected_accent,
                    modalities_used: result.modalities_used,
                    audio_result: result.audio_result as any,
                    text_result: result.text_result as any,
                    video_result: result.video_result as any,
                });
                if (!cancelled) {
                    setAiInsights(response?.insights || null);
                    setInsightsSource(response?.source || 'error');
                }
            } catch {
                if (!cancelled) {
                    setAiInsights(null);
                    setInsightsSource('error');
                }
            } finally {
                if (!cancelled) setInsightsLoading(false);
            }
        };
        loadInsights();
        return () => { cancelled = true; };
    }, [result]);

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
            happy: 'from-yellow-400 to-cyan-400',
            sad: 'from-blue-400 to-blue-600',
            angry: 'from-red-400 to-red-600',
            neutral: 'from-gray-400 to-gray-600',
            fear: 'from-purple-400 to-purple-600',
            surprise: 'from-pink-400 to-pink-600',
            disgust: 'from-green-400 to-green-600',
        };
        return colors[emotion.toLowerCase()] || 'from-teal-400 to-teal-600';
    };

    const getAccentInfo = (accent: string) => {
        const info: Record<string, { flag: string; label: string }> = {
            american: { flag: '🇺🇸', label: 'American' },
            british: { flag: '🇬🇧', label: 'British' },
            australian: { flag: '🇦🇺', label: 'Australian' },
            indian: { flag: '🇮🇳', label: 'Indian' },
            canadian: { flag: '🇨🇦', label: 'Canadian' },
            scottish: { flag: '🏴󠁧󠁢󠁳󠁣󠁴󠁿', label: 'Scottish' },
            irish: { flag: '🇮🇪', label: 'Irish' },
            african: { flag: '🌍', label: 'African' },
            newzealand: { flag: '🇳🇿', label: 'New Zealand' },
            welsh: { flag: '🏴󠁧󠁢󠁷󠁬󠁳󠁿', label: 'Welsh' },
            malaysian: { flag: '🇲🇾', label: 'Malaysian' },
            filipino: { flag: '🇵🇭', label: 'Filipino' },
            singaporean: { flag: '🇸🇬', label: 'Singaporean' },
            hongkong: { flag: '🇭🇰', label: 'Hong Kong' },
            bermudian: { flag: '🇧🇲', label: 'Bermudian' },
            southatlantic: { flag: '🌊', label: 'South Atlantic' },
            south_asian: { flag: '🌏', label: 'South Asian' },
        };
        return info[accent.toLowerCase()] || { flag: '🌍', label: accent };
    };

    // Get accent from the result (direct for audio, from audio_result for multimodal)
    const detectedAccent = result.detected_accent || result.audio_result?.detected_accent || null;

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 0.8) return 'text-green-600';
        if (confidence >= 0.6) return 'text-yellow-600';
        return 'text-orange-600';
    };

    return createPortal(
        <div
            className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
        >
            {/* Modal — centered, fixed max height, internal scroll */}
            <div
                className="bg-white rounded-2xl shadow-2xl w-full max-w-md"
                style={{ maxHeight: '85vh', display: 'flex', flexDirection: 'column' }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header — fixed at top */}
                <div
                    className={`bg-gradient-to-r ${getEmotionColor(result.emotion_label)} p-5 text-white text-center rounded-t-2xl`}
                    style={{ flexShrink: 0, position: 'relative' }}
                >
                    <button
                        onClick={onClose}
                        aria-label="Close modal"
                        className="absolute top-3 right-3 p-2 rounded-full bg-white/20 hover:bg-white/40 transition-colors cursor-pointer"
                    >
                        <X className="w-5 h-5" />
                    </button>
                    <div className="text-5xl mb-2">{getEmotionEmoji(result.emotion_label)}</div>
                    <h2 className="text-2xl font-bold capitalize">{result.emotion_label}</h2>
                    <p className="text-white/80 text-sm mt-1">Detected Emotion</p>
                    {/* Well-being Badge */}
                    <div className="mt-2 inline-flex items-center space-x-1.5 bg-white/15 backdrop-blur-sm rounded-full px-3 py-1">
                        <span className="text-sm">{getWellbeingIndicator(result.emotion_label, result.confidence_score).emoji}</span>
                        <span className="text-sm font-medium">{getWellbeingIndicator(result.emotion_label, result.confidence_score).label}</span>
                    </div>
                </div>

                {/* Scrollable content area */}
                <div
                    className="p-5 space-y-4"
                    style={{ overflowY: 'auto', flex: '1 1 auto', minHeight: 0 }}
                >
                    {/* Mental State Summary */}
                    <div className="bg-stone-50 rounded-lg p-3 text-center">
                        <p className="text-xs text-gray-600 leading-relaxed">
                            {getMentalStateSummary(result.emotion_label, result.confidence_score, result.modalities_used)}
                        </p>
                    </div>

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

                    {/* Accent Detection Badge */}
                    {detectedAccent && (
                        <div className="text-center">
                            <div className="text-sm text-gray-500 mb-2">Detected Accent</div>
                            <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-xl px-4 py-2.5">
                                <Globe className="w-4 h-4 text-indigo-600" />
                                <span className="text-2xl">{getAccentInfo(detectedAccent).flag}</span>
                                <span className="text-base font-semibold text-indigo-800 capitalize">
                                    {getAccentInfo(detectedAccent).label}
                                </span>
                            </div>
                        </div>
                    )}

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

                    {/* ── AI Insights Section ── */}
                    <div className="rounded-lg overflow-hidden" style={{ border: '1px solid #e0f2fe' }}>
                        <div className="flex items-center gap-2 px-3 py-2" style={{ background: 'linear-gradient(135deg, #f0f9ff, #e0f2fe)' }}>
                            <Sparkles className="w-4 h-4" style={{ color: '#0891b2' }} />
                            <span className="text-sm font-semibold" style={{ color: '#0e7490' }}>AI Insights</span>
                            {insightsSource === 'gemini' && (
                                <span className="text-[9px] px-1.5 py-0.5 rounded-full" style={{ background: '#06b6d4', color: 'white' }}>Gemini</span>
                            )}
                        </div>
                        <div className="px-3 py-3" style={{ background: '#fafcff' }}>
                            {insightsLoading ? (
                                <div className="flex items-center justify-center gap-2 py-4">
                                    <Loader2 className="w-4 h-4 animate-spin" style={{ color: '#06b6d4' }} />
                                    <span className="text-xs" style={{ color: '#737373' }}>Generating AI insights...</span>
                                </div>
                            ) : aiInsights ? (
                                <p className="text-xs text-gray-700 leading-relaxed whitespace-pre-wrap">{aiInsights}</p>
                            ) : (
                                <p className="text-xs text-gray-400 text-center py-2">AI insights unavailable. Start the backend to enable.</p>
                            )}
                        </div>
                    </div>
                </div>

                {/* Disclaimer */}
                <div className="px-5 pb-2">
                    <div className="flex items-start space-x-2 bg-gray-50 rounded-lg p-2.5">
                        <Shield className="w-3.5 h-3.5 text-gray-400 flex-shrink-0 mt-0.5" />
                        <p className="text-[10px] text-gray-400 leading-relaxed">{DISCLAIMER_TEXT}</p>
                    </div>
                </div>

                {/* Buttons — fixed at bottom */}
                <div
                    className="p-4 rounded-b-2xl"
                    style={{ flexShrink: 0, borderTop: '1px solid #f3f4f6' }}
                >
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="flex-1 py-3 rounded-lg font-medium text-sm transition-all cursor-pointer"
                            style={{
                                background: 'transparent',
                                border: '1.5px solid #e5e7eb',
                                color: '#6b7280',
                            }}
                            onMouseEnter={e => { e.currentTarget.style.background = '#f9fafb'; e.currentTarget.style.borderColor = '#d1d5db'; }}
                            onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = '#e5e7eb'; }}
                        >
                            Analyze Another
                        </button>
                        {onNavigate && (
                            <button
                                onClick={() => {
                                    onClose();
                                    onNavigate('results');
                                }}
                                className="flex-1 py-3 rounded-lg font-medium text-sm text-white transition-all cursor-pointer flex items-center justify-center gap-2"
                                style={{
                                    background: 'linear-gradient(135deg, #06b6d4, #0891b2)',
                                    boxShadow: '0 4px 15px rgba(6,182,212,0.35)',
                                }}
                                onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 6px 25px rgba(6,182,212,0.5)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
                                onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 4px 15px rgba(6,182,212,0.35)'; e.currentTarget.style.transform = 'translateY(0)'; }}
                            >
                                View Full Report
                                <ArrowRight className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div >,
        document.body
    );
}
