import { useEffect, useState } from 'react';
import {
  Download, Calendar, Trash2, RefreshCw, Globe, ChevronDown, ChevronUp,
  AlertTriangle, Shield, Phone, Heart, Brain, Activity, Sparkles
} from 'lucide-react';
import { getStoredResults, deleteResult, type AnalysisResult } from '../lib/storage';
import { RadarChart } from '../components/RadarChart';
import { EmotionGauge } from '../components/EmotionGauge';
import {
  getMentalStateSummary,
  getEmotionalValence,
  getValenceInfo,
  getArousalLevel,
  getArousalInfo,
  getWellbeingIndicator,
  getConfidenceLabel,
  getCopingStrategies,
  getContextualTips,
  getHelplineInfo,
  getDominantAndSecondary,
  getModalityAgreement,
  DISCLAIMER_TEXT,
} from '../lib/emotionInsights';
import { generateSingleReportHTML, downloadPDF } from '../lib/reportGenerator';



/* ── Typing Text Component ── */
function TypingText({ text, speed = 20 }: { text: string; speed?: number }) {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    setDisplayed('');
    let i = 0;
    const timer = setInterval(() => {
      if (i < text.length) { setDisplayed(text.slice(0, i + 1)); i++; }
      else clearInterval(timer);
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed]);
  return (
    <span>
      {displayed}
      <span className="inline-block w-0.5 h-3.5 ml-0.5 bg-cyan-400 align-middle" style={{ animation: 'typing-cursor 0.8s step-end infinite' }} />
    </span>
  );
}

export function Results() {
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [showMetadata, setShowMetadata] = useState(false);
  const [showHelplines, setShowHelplines] = useState(false);
  const [barsVisible, setBarsVisible] = useState(false);

  useEffect(() => {
    loadAnalyses();
  }, []);

  const loadAnalyses = () => {
    setLoading(true);
    const results = getStoredResults();
    setAnalyses(results);
    if (results.length > 0) {
      setSelectedAnalysis(results[0]);
    }
    setLoading(false);
    setTimeout(() => setBarsVisible(true), 300);
  };

  const handleDelete = (id: string) => {
    deleteResult(id);
    loadAnalyses();
  };

  const handleClearSelected = () => {
    if (!selectedAnalysis) return;
    deleteResult(selectedAnalysis.id);
    const remaining = analyses.filter(a => a.id !== selectedAnalysis.id);
    setAnalyses(remaining);
    setSelectedAnalysis(remaining.length > 0 ? remaining[0] : null);
  };

  const handleDownloadReport = () => {
    if (!selectedAnalysis) return;
    const html = generateSingleReportHTML(selectedAnalysis);
    downloadPDF(html, `AEMER-Report-${selectedAnalysis.id.slice(0, 8)}.pdf`);
  };

  const getEmotionEmoji = (emotion: string) => {
    const emojis: Record<string, string> = {
      happy: '😊', sad: '😢', angry: '😠', neutral: '😐',
      fear: '😨', surprise: '😲', disgust: '🤢',
    };
    return emojis[emotion.toLowerCase()] || '🎭';
  };

  const getEmotionGradient = (emotion: string) => {
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
    };
    return info[accent.toLowerCase()] || { flag: '🌍', label: accent };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center space-y-3">
          <div className="relative">
            <div className="w-12 h-12 border-2 rounded-full animate-spin" style={{ borderColor: 'rgba(6,182,212,0.15)', borderTopColor: '#06b6d4' }} />
            <div className="absolute inset-0 w-12 h-12 border-2 rounded-full animate-spin" style={{ borderColor: 'transparent', borderBottomColor: 'rgba(6,182,212,0.3)', animationDirection: 'reverse', animationDuration: '1.5s' }} />
          </div>
          <span style={{ color: '#525252', fontSize: '13px' }}>Loading results...</span>
        </div>
      </div>
    );
  }

  if (analyses.length === 0) {
    return (
      <div className="text-center py-16 animate-fade-in-up" style={{ background: '#0a0a0a', borderRadius: '16px', border: '1px solid rgba(6,182,212,0.08)' }}>
        <div className="text-6xl mb-4">📊</div>
        <h3 className="text-xl font-semibold text-white mb-2">No Results Yet</h3>
        <p style={{ color: '#525252', fontSize: '14px' }}>Upload an audio, video, or text file in the Analyze tab to see your detailed report here.</p>
      </div>
    );
  }

  // Compute insights for selected analysis
  const sel = selectedAnalysis;
  const wellbeing = sel ? getWellbeingIndicator(sel.emotion_label, sel.confidence_score) : null;
  const mentalSummary = sel ? getMentalStateSummary(sel.emotion_label, sel.confidence_score, sel.modalities_used) : '';
  const valence = sel ? getEmotionalValence(sel.emotion_label) : 'neutral';
  const valenceInfo = getValenceInfo(valence);
  const arousal = sel ? getArousalLevel(sel.emotion_label) : 'low';
  const arousalInfo = getArousalInfo(arousal);
  const confidenceLabel = sel ? getConfidenceLabel(sel.confidence_score) : 'Low';
  const dominantSecondary = sel?.all_probabilities ? getDominantAndSecondary(sel.all_probabilities) : null;
  const copingStrategies = sel ? getCopingStrategies(sel.emotion_label) : [];
  const contextualTip = sel ? getContextualTips(sel.emotion_label) : '';
  const helplines = getHelplineInfo();
  const modalityAgreement = sel ? getModalityAgreement(
    sel.audio_result?.emotion_label,
    sel.text_result?.emotion_label,
    sel.video_result?.emotion_label,
  ) : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 animate-fade-in-up">
        <div>
          <h2 className="text-xl sm:text-3xl font-bold text-white mb-1 flex items-center space-x-2 sm:space-x-3">
            <Sparkles className="w-5 h-5 sm:w-7 sm:h-7 text-cyan-400" />
            <span>Analysis Report</span>
          </h2>
          <p className="text-xs sm:text-sm" style={{ color: '#525252' }}>
            <TypingText text={`Detailed emotion insights & mental health recommendations (${analyses.length} total)`} />
          </p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={(e) => {
              const icon = e.currentTarget.querySelector('svg');
              if (icon) { icon.style.animation = 'spin 0.6s ease'; setTimeout(() => { icon.style.animation = ''; }, 600); }
              loadAnalyses();
            }}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all"
            style={{
              background: 'linear-gradient(135deg, rgba(6,182,212,0.12), rgba(6,182,212,0.06))',
              border: '1px solid rgba(6,182,212,0.15)',
              color: '#22d3ee',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'linear-gradient(135deg, rgba(6,182,212,0.2), rgba(6,182,212,0.1))'; e.currentTarget.style.borderColor = 'rgba(6,182,212,0.3)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(6,182,212,0.15)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'linear-gradient(135deg, rgba(6,182,212,0.12), rgba(6,182,212,0.06))'; e.currentTarget.style.borderColor = 'rgba(6,182,212,0.15)'; e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'translateY(0)'; }}
            title="Refresh data"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          <button
            onClick={handleClearSelected}
            disabled={!selectedAnalysis}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-sm font-medium transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              background: 'linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05))',
              border: '1px solid rgba(239,68,68,0.15)',
              color: '#f87171',
            }}
            onMouseEnter={e => { if (selectedAnalysis) { e.currentTarget.style.background = 'linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.1))'; e.currentTarget.style.borderColor = 'rgba(239,68,68,0.3)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(239,68,68,0.12)'; e.currentTarget.style.transform = 'translateY(-1px)'; } }}
            onMouseLeave={e => { e.currentTarget.style.background = 'linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05))'; e.currentTarget.style.borderColor = 'rgba(239,68,68,0.15)'; e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'translateY(0)'; }}
            title="Delete current analysis"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear This</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ═══ LEFT SIDEBAR — Analysis list ═══ */}
        <div className="lg:col-span-1 rounded-2xl p-4 max-h-[700px] overflow-y-auto animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '100ms', animationFillMode: 'backwards' }}>
          <h3 className="font-semibold text-white mb-3">Recent Analyses</h3>
          <div className="space-y-2">
            {analyses.map((analysis) => (
              <div
                key={analysis.id}
                className={`relative group p-3 rounded-xl transition-all cursor-pointer`}
                style={selectedAnalysis?.id === analysis.id
                  ? { background: 'rgba(6,182,212,0.08)', border: '1px solid rgba(6,182,212,0.2)', boxShadow: '0 0 15px rgba(6,182,212,0.08)' }
                  : { background: 'transparent', border: '1px solid transparent' }
                }
                onClick={() => setSelectedAnalysis(analysis)}
                onMouseEnter={e => { if (selectedAnalysis?.id !== analysis.id) { e.currentTarget.style.background = 'rgba(255,255,255,0.02)'; e.currentTarget.style.borderColor = 'rgba(6,182,212,0.1)'; } }}
                onMouseLeave={e => { if (selectedAnalysis?.id !== analysis.id) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'transparent'; } }}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">{getEmotionEmoji(analysis.emotion_label)}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-white capitalize">{analysis.emotion_label}</div>
                    <div className="text-xs text-neutral-500">
                      {new Date(analysis.timestamp).toLocaleString()}
                    </div>
                    {analysis.filename && (
                      <div className="text-xs text-cyan-400 truncate">{analysis.filename}</div>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <div className="text-sm font-medium text-cyan-400">
                      {(analysis.confidence_score * 100).toFixed(0)}%
                    </div>
                    <span className="text-xs">{getWellbeingIndicator(analysis.emotion_label, analysis.confidence_score).emoji}</span>
                  </div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(analysis.id); }}
                  className="absolute top-2 right-2 p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* ═══ RIGHT SIDE — Full Report ═══ */}
        <div className="lg:col-span-2 space-y-5">
          {sel && (
            <>
              {/* ─── 1. EMOTION HEADER ─── */}
              <div className={`bg-gradient-to-r ${getEmotionGradient(sel.emotion_label)} rounded-2xl p-6 text-white shadow-lg animate-fade-in-up relative overflow-hidden`} style={{ animationDelay: '150ms', animationFillMode: 'backwards' }}>
                {/* Shimmer effect */}
                <div className="absolute inset-0 opacity-20" style={{ background: 'linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%)', animation: 'shimmer 3s infinite' }} />
                <div className="flex items-center space-x-4">
                  <div className="text-6xl">{getEmotionEmoji(sel.emotion_label)}</div>
                  <div className="flex-1">
                    <h3 className="text-3xl font-bold capitalize">{sel.emotion_label}</h3>
                    <p className="text-white/80 text-sm mt-1">Detected Emotion</p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <span className="px-2 py-0.5 bg-white/20 rounded-full text-xs font-medium capitalize">{sel.input_type}</span>
                      {sel.detected_accent && (
                        <span className="px-2 py-0.5 bg-white/20 rounded-full text-xs font-medium">
                          {getAccentInfo(sel.detected_accent).flag} {getAccentInfo(sel.detected_accent).label}
                        </span>
                      )}
                    </div>
                  </div>
                  {/* Well-being badge */}
                  {wellbeing && (
                    <div className="bg-white/15 backdrop-blur-sm rounded-xl px-4 py-3 text-center">
                      <div className="text-2xl mb-1">{wellbeing.emoji}</div>
                      <div className="text-sm font-semibold">{wellbeing.label}</div>
                    </div>
                  )}
                </div>
              </div>

              {/* ─── Quality Warning ─── */}
              {sel.quality_warning && (
                <div className="flex items-start space-x-3 bg-cyan-900/20 border border-cyan-500/40 rounded-xl p-4">
                  <AlertTriangle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-neutral-300">{sel.quality_warning}</p>
                </div>
              )}

              {/* ─── 2. MENTAL STATE SUMMARY ─── */}
              <div className="rounded-2xl p-6 animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '200ms', animationFillMode: 'backwards' }}>
                <h4 className="font-semibold text-white mb-3 flex items-center space-x-2">
                  <Brain className="w-5 h-5 text-cyan-400" />
                  <span>Mental State Summary</span>
                </h4>
                <p className="text-neutral-300 text-sm leading-relaxed">{mentalSummary}</p>

                {/* Contextual Tip */}
                <div className="mt-4 bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                  <p className="text-sm text-cyan-300">{contextualTip}</p>
                </div>
              </div>

              {/* ─── 3. CONFIDENCE GAUGE + VALENCE/AROUSAL ─── */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5 animate-fade-in-up" style={{ animationDelay: '250ms', animationFillMode: 'backwards' }}>
                {/* Confidence Gauge */}
                <div className="rounded-2xl p-5" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)' }}>
                  <h4 className="font-semibold text-white mb-2 text-center flex items-center justify-center space-x-2">
                    <Activity className="w-4 h-4 text-cyan-400" />
                    <span>Confidence Score</span>
                  </h4>
                  <EmotionGauge value={sel.confidence_score} label={confidenceLabel} />
                </div>

                {/* Valence + Arousal */}
                <div className="rounded-2xl p-5 space-y-4" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)' }}>
                  <h4 className="font-semibold text-white flex items-center space-x-2">
                    <Heart className="w-4 h-4 text-cyan-400" />
                    <span>Emotional Profile</span>
                  </h4>

                  {/* Valence Badge */}
                  <div className={`flex items-center space-x-3 rounded-lg p-3 border ${valenceInfo.bg}`}>
                    <span className="text-2xl">{valenceInfo.emoji}</span>
                    <div>
                      <div className={`font-semibold text-sm ${valenceInfo.color}`}>
                        {valenceInfo.label} Valence
                      </div>
                      <p className="text-xs text-neutral-500">
                        Emotional polarity of the detected state
                      </p>
                    </div>
                  </div>

                  {/* Arousal Badge */}
                  <div className={`flex items-center space-x-3 rounded-lg p-3 border ${arousalInfo.bg}`}>
                    <span className="text-2xl">{arousalInfo.emoji}</span>
                    <div>
                      <div className={`font-semibold text-sm ${arousalInfo.color}`}>
                        {arousalInfo.label}
                      </div>
                      <p className="text-xs text-neutral-500">
                        {arousalInfo.description}
                      </p>
                    </div>
                  </div>

                  {/* Well-being full description */}
                  {wellbeing && (
                    <div className="bg-neutral-800/30 rounded-lg p-3 border border-neutral-700/30">
                      <div className="flex items-center space-x-2 mb-1">
                        <span>{wellbeing.emoji}</span>
                        <span className={`font-semibold text-sm ${wellbeing.color}`}>{wellbeing.label}</span>
                      </div>
                      <p className="text-xs text-neutral-500">{wellbeing.description}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* ─── 4. DOMINANT vs SECONDARY EMOTION ─── */}
              {dominantSecondary && (
                <div className="rounded-2xl p-5 animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '300ms', animationFillMode: 'backwards' }}>
                  <h4 className="font-semibold text-white mb-3">Dominant vs. Secondary Emotion</h4>
                  <div className="grid grid-cols-2 gap-4 mb-3">
                    <div
                      className="rounded-xl p-4 text-center transition-all"
                      style={{ background: 'rgba(6,182,212,0.06)', border: '1px solid rgba(6,182,212,0.1)' }}
                      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 8px 20px rgba(6,182,212,0.12)'; }}
                      onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
                    >
                      <div className="text-3xl mb-1">{getEmotionEmoji(dominantSecondary.dominant.emotion)}</div>
                      <div className="text-sm font-semibold text-cyan-300 capitalize">{dominantSecondary.dominant.emotion}</div>
                      <div className="text-xs text-neutral-500">{(dominantSecondary.dominant.probability * 100).toFixed(1)}%</div>
                    </div>
                    <div
                      className="rounded-xl p-4 text-center transition-all"
                      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
                      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 8px 20px rgba(255,255,255,0.04)'; }}
                      onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
                    >
                      <div className="text-3xl mb-1">{getEmotionEmoji(dominantSecondary.secondary.emotion)}</div>
                      <div className="text-sm font-semibold text-neutral-300 capitalize">{dominantSecondary.secondary.emotion}</div>
                      <div className="text-xs text-neutral-500">{(dominantSecondary.secondary.probability * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                  <p className="text-xs text-neutral-500 italic">{dominantSecondary.interpretation}</p>
                </div>
              )}

              {/* ─── 5. RADAR CHART ─── */}
              {sel.all_probabilities && Object.keys(sel.all_probabilities).length > 2 && (
                <div className="rounded-2xl p-5 animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '350ms', animationFillMode: 'backwards' }}>
                  <h4 className="font-semibold text-white mb-3 text-center">Emotion Distribution</h4>
                  <RadarChart data={sel.all_probabilities} size={280} />
                  {/* Probability list below radar */}
                  <div className="mt-4 space-y-1.5">
                    {Object.entries(sel.all_probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([emotion, probability]) => (
                        <div key={emotion} className="flex items-center space-x-3 group">
                          <span className="text-lg w-7 group-hover:scale-125 transition-transform inline-block">{getEmotionEmoji(emotion)}</span>
                          <span className="w-20 text-neutral-300 capitalize text-sm font-medium">{emotion}</span>
                          <div className="flex-1 rounded-full h-2.5 overflow-hidden" style={{ background: '#171717' }}>
                            <div
                              className={`bg-gradient-to-r ${getEmotionGradient(emotion)} h-2.5 rounded-full`}
                              style={{
                                width: barsVisible ? `${probability * 100}%` : '0%',
                                transition: 'width 1s cubic-bezier(0.22, 1, 0.36, 1)',
                              }}
                            />
                          </div>
                          <span className="w-14 text-right text-sm font-semibold" style={{ color: '#06b6d4' }}>
                            {(probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* ─── 6. MODALITY BREAKDOWN ─── */}
              {sel.modalities_used && sel.modalities_used.length > 1 && (
                <div className="rounded-2xl p-5 animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '400ms', animationFillMode: 'backwards' }}>
                  <h4 className="font-semibold text-white mb-1 flex items-center space-x-2">
                    <Globe className="w-4 h-4 text-cyan-400" />
                    <span>Modality Breakdown</span>
                  </h4>
                  {modalityAgreement && (
                    <p className="text-xs text-neutral-500 mb-4">
                      {modalityAgreement.emoji} {modalityAgreement.description}
                      {sel.fusion_method && ` • Fusion: ${sel.fusion_method.replace('_', ' ')}`}
                    </p>
                  )}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {sel.audio_result && (
                      <div
                        className="rounded-xl p-4 transition-all"
                        style={{ background: 'rgba(6,182,212,0.04)', border: '1px solid rgba(6,182,212,0.1)' }}
                        onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 8px 20px rgba(6,182,212,0.1)'; }}
                        onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
                      >
                        <div className="text-lg mb-1">🎤</div>
                        <div className="text-sm font-semibold text-cyan-300">Audio</div>
                        <div className="text-xl font-bold text-white capitalize mt-1">{sel.audio_result.emotion_label}</div>
                        <div className="text-xs text-neutral-500 mt-1">
                          Confidence: {(sel.audio_result.confidence_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-neutral-600">
                          Weight: {(sel.audio_result.weight * 100).toFixed(0)}%
                        </div>
                        {sel.audio_result.detected_accent && (
                          <div className="mt-2 text-xs text-cyan-300">
                            {getAccentInfo(sel.audio_result.detected_accent).flag} {getAccentInfo(sel.audio_result.detected_accent).label}
                          </div>
                        )}
                      </div>
                    )}
                    {sel.text_result && (
                      <div
                        className="rounded-xl p-4 transition-all"
                        style={{ background: 'rgba(168,85,247,0.04)', border: '1px solid rgba(168,85,247,0.1)' }}
                        onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 8px 20px rgba(168,85,247,0.1)'; }}
                        onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
                      >
                        <div className="text-lg mb-1">📝</div>
                        <div className="text-sm font-semibold text-purple-300">Text</div>
                        <div className="text-xl font-bold text-white capitalize mt-1">{sel.text_result.emotion_label}</div>
                        <div className="text-xs text-neutral-500 mt-1">
                          Confidence: {(sel.text_result.confidence_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-neutral-600">
                          Weight: {(sel.text_result.weight * 100).toFixed(0)}%
                        </div>
                      </div>
                    )}
                    {sel.video_result && (
                      <div
                        className="rounded-xl p-4 transition-all"
                        style={{ background: 'rgba(56,189,248,0.04)', border: '1px solid rgba(56,189,248,0.1)' }}
                        onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 8px 20px rgba(56,189,248,0.1)'; }}
                        onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
                      >
                        <div className="text-lg mb-1">📹</div>
                        <div className="text-sm font-semibold text-blue-300">Video</div>
                        <div className="text-xl font-bold text-white capitalize mt-1">{sel.video_result.emotion_label}</div>
                        <div className="text-xs text-neutral-500 mt-1">
                          Confidence: {(sel.video_result.confidence_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-neutral-600">
                          Weight: {(sel.video_result.weight * 100).toFixed(0)}%
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ─── 7. MENTAL HEALTH CONTEXT & COPING STRATEGIES ─── */}
              <div className="rounded-2xl p-5 animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '450ms', animationFillMode: 'backwards' }}>
                <h4 className="font-semibold text-white mb-4 flex items-center space-x-2">
                  <Heart className="w-5 h-5 text-pink-400" />
                  <span>Mental Health Context &amp; Recommendations</span>
                </h4>

                {/* Coping Strategies */}
                <div className="space-y-3 mb-5">
                  {copingStrategies.map((strategy, i) => (
                    <div
                      key={i}
                      className="flex items-start space-x-3 rounded-xl p-3 transition-all"
                      style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid rgba(255,255,255,0.04)' }}
                      onMouseEnter={e => { e.currentTarget.style.background = 'rgba(6,182,212,0.04)'; e.currentTarget.style.borderColor = 'rgba(6,182,212,0.1)'; e.currentTarget.style.transform = 'translateX(4px)'; }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.015)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.04)'; e.currentTarget.style.transform = 'translateX(0)'; }}
                    >
                      <span className="text-2xl flex-shrink-0">{strategy.icon}</span>
                      <div>
                        <div className="text-sm font-semibold text-white">{strategy.title}</div>
                        <p className="text-xs text-neutral-500 mt-0.5 leading-relaxed">{strategy.description}</p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Helpline Toggle */}
                {(wellbeing?.level === 'elevated_concern' || wellbeing?.level === 'mild_concern') && (
                  <div className="border-t border-cyan-900/15 pt-4">
                    <button
                      onClick={() => setShowHelplines(!showHelplines)}
                      className="flex items-center space-x-2 text-sm text-neutral-400 hover:text-white transition-colors w-full"
                    >
                      <Phone className="w-4 h-4" />
                      <span>Need to Talk? Mental Health Helplines</span>
                      {showHelplines ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
                    </button>
                    {showHelplines && (
                      <div className="mt-3 space-y-2">
                        {helplines.map((line, i) => (
                          <div key={i} className="flex items-center justify-between bg-neutral-800/30 rounded-lg p-3 border border-neutral-700/20">
                            <div>
                              <div className="text-sm font-medium text-white">{line.name}</div>
                              <div className="text-xs text-neutral-500">{line.region} • {line.available}</div>
                            </div>
                            <div className="text-sm font-mono text-cyan-400">{line.number}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ─── 8. DISCLAIMER ─── */}
              <div className="flex items-start space-x-3 bg-neutral-900/30 border border-neutral-800/30 rounded-xl p-4 animate-fade-in-up" style={{ animationDelay: '500ms', animationFillMode: 'backwards' }}>
                <Shield className="w-5 h-5 text-neutral-600 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-neutral-600 leading-relaxed">{DISCLAIMER_TEXT}</p>
              </div>

              {/* ─── 9. METADATA (Collapsible) ─── */}
              <div className="rounded-2xl p-5 animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '550ms', animationFillMode: 'backwards' }}>
                <button
                  onClick={() => setShowMetadata(!showMetadata)}
                  className="w-full flex items-center justify-between p-4 hover:bg-neutral-800/20 transition-colors"
                >
                  <h4 className="font-semibold text-white flex items-center space-x-2">
                    <Calendar className="w-4 h-4 text-cyan-400" />
                    <span>Analysis Details &amp; Metadata</span>
                  </h4>
                  {showMetadata ? <ChevronUp className="w-4 h-4 text-neutral-500" /> : <ChevronDown className="w-4 h-4 text-neutral-500" />}
                </button>
                {showMetadata && (
                  <div className="px-4 pb-4 space-y-2 text-sm border-t border-cyan-900/15 pt-3">
                    <div className="flex justify-between">
                      <span className="text-neutral-400">Analysis ID:</span>
                      <span className="font-mono text-white text-xs">{sel.id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neutral-400">Timestamp:</span>
                      <span className="text-white">{new Date(sel.timestamp).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neutral-400">Input Type:</span>
                      <span className="text-white capitalize">{sel.input_type}</span>
                    </div>
                    {sel.filename && (
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Filename:</span>
                        <span className="text-white truncate ml-4">{sel.filename}</span>
                      </div>
                    )}
                    {sel.detected_accent && (
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Detected Accent:</span>
                        <span className="text-white">
                          {getAccentInfo(sel.detected_accent).flag} {getAccentInfo(sel.detected_accent).label}
                        </span>
                      </div>
                    )}
                    {sel.fusion_method && (
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Fusion Method:</span>
                        <span className="text-white capitalize">{sel.fusion_method.replace('_', ' ')}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ─── 10. DOWNLOAD REPORT ─── */}
              <div className="rounded-2xl p-5 animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '600ms', animationFillMode: 'backwards' }}>
                <h4 className="font-semibold text-white mb-4 flex items-center space-x-2">
                  <Download className="w-4 h-4 text-cyan-400" />
                  <span>Export Report</span>
                </h4>
                <button
                  onClick={handleDownloadReport}
                  className="flex items-center space-x-2 px-6 py-3 text-white rounded-xl text-sm font-medium transition-all"
                  style={{ background: 'linear-gradient(135deg, #06b6d4, #0891b2)', boxShadow: '0 4px 15px rgba(6,182,212,0.3)' }}
                  onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 8px 30px rgba(6,182,212,0.4)'; }}
                  onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 15px rgba(6,182,212,0.3)'; }}
                >
                  <Download className="w-4 h-4" />
                  <span>Download Full Report</span>
                </button>
                <p className="text-xs text-neutral-600 mt-2">Downloads a comprehensive PDF report with full analysis details.</p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
