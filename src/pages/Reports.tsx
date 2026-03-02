import { useState, useEffect, useRef } from 'react';
import { Download, BarChart3, TrendingUp, RefreshCw, Sparkles, PieChart, Award } from 'lucide-react';
import { getStoredResults, type AnalysisResult } from '../lib/storage';
import { generateSummaryReportHTML, downloadHTML } from '../lib/reportGenerator';

interface EmotionStats {
  emotion: string;
  count: number;
  avgConfidence: number;
  emoji: string;
}

/* ── Animated Counter Hook ── */
function useAnimatedCounter(target: number, duration = 1200, decimals = 0) {
  const [value, setValue] = useState(0);
  const ref = useRef<number>(0);
  useEffect(() => {
    const start = ref.current;
    const diff = target - start;
    if (diff === 0) return;
    const startTime = performance.now();
    const step = (now: number) => {
      const t = Math.min((now - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      const current = start + diff * ease;
      setValue(current);
      ref.current = current;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration]);
  return decimals > 0 ? value.toFixed(decimals) : Math.round(value);
}

/* ── Typing Text Component ── */
function TypingText({ text, speed = 30 }: { text: string; speed?: number }) {
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
      <span className="inline-block w-0.5 h-4 ml-0.5 bg-cyan-400 align-middle" style={{ animation: 'typing-cursor 0.8s step-end infinite' }} />
    </span>
  );
}

export function Reports() {
  const [emotionStats, setEmotionStats] = useState<EmotionStats[]>([]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [totalAnalyses, setTotalAnalyses] = useState(0);
  const [loading, setLoading] = useState(true);
  const [barsVisible, setBarsVisible] = useState(false);

  useEffect(() => { loadReportData(); }, []);
  useEffect(() => {
    if (!loading && emotionStats.length > 0) {
      const t = setTimeout(() => setBarsVisible(true), 400);
      return () => clearTimeout(t);
    }
  }, [loading, emotionStats]);

  const loadReportData = () => {
    setLoading(true);
    setBarsVisible(false);
    const analyses = getStoredResults();
    setResults(analyses);
    setTotalAnalyses(analyses.length);

    if (analyses.length > 0) {
      const emotionMap = new Map<string, { count: number; totalConf: number }>();
      analyses.forEach((a: AnalysisResult) => {
        const emo = emotionMap.get(a.emotion_label) || { count: 0, totalConf: 0 };
        emo.count++;
        emo.totalConf += a.confidence_score || 0;
        emotionMap.set(a.emotion_label, emo);
      });
      const emoStats: EmotionStats[] = Array.from(emotionMap.entries()).map(
        ([emotion, data]) => ({
          emotion,
          count: data.count,
          avgConfidence: data.totalConf / data.count,
          emoji: getEmotionEmoji(emotion),
        })
      );
      setEmotionStats(emoStats.sort((a, b) => b.count - a.count));
    } else {
      setEmotionStats([]);
    }
    setLoading(false);
  };

  const getEmotionEmoji = (emotion: string) => {
    const emojis: Record<string, string> = {
      happy: '😊', sad: '😢', angry: '😠', neutral: '😐',
      fear: '😨', surprise: '😲', disgust: '🤢',
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

  const getEmotionSolid = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: '#facc15', sad: '#60a5fa', angry: '#f87171', neutral: '#9ca3af',
      fear: '#c084fc', surprise: '#f472b6', disgust: '#4ade80',
    };
    return colors[emotion.toLowerCase()] || '#06b6d4';
  };

  const handleExportReport = () => {
    const html = generateSummaryReportHTML(results, emotionStats);
    downloadHTML(html, `AEMER-Analytics-${new Date().toISOString().split('T')[0]}.html`);
  };

  const avgConf = emotionStats.length > 0
    ? (emotionStats.reduce((sum, e) => sum + e.avgConfidence, 0) / emotionStats.length) * 100
    : 0;

  const animatedTotal = useAnimatedCounter(totalAnalyses);
  const animatedUnique = useAnimatedCounter(emotionStats.length);
  const animatedAvg = useAnimatedCounter(avgConf, 1400, 1);
  const maxCount = Math.max(...emotionStats.map(e => e.count), 1);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center space-y-3">
          <div className="relative">
            <div className="w-12 h-12 border-2 rounded-full animate-spin" style={{ borderColor: 'rgba(6,182,212,0.15)', borderTopColor: '#06b6d4' }} />
            <div className="absolute inset-0 w-12 h-12 border-2 rounded-full animate-spin" style={{ borderColor: 'transparent', borderBottomColor: 'rgba(6,182,212,0.3)', animationDirection: 'reverse', animationDuration: '1.5s' }} />
          </div>
          <span style={{ color: '#525252', fontSize: '13px' }}>Loading analytics...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between animate-fade-in-up">
        <div>
          <h2 className="text-3xl font-bold text-white mb-1 flex items-center space-x-3">
            <Sparkles className="w-7 h-7 text-cyan-400" />
            <span>Reports</span>
          </h2>
          <p style={{ color: '#525252', fontSize: '14px' }}>
            <TypingText text="Analytics and trends from your emotion analysis history" speed={25} />
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={loadReportData}
            className="p-2.5 rounded-xl transition-all group"
            style={{ border: '1px solid rgba(6,182,212,0.1)', background: 'rgba(6,182,212,0.03)' }}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(6,182,212,0.08)'; e.currentTarget.style.borderColor = 'rgba(6,182,212,0.25)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'rgba(6,182,212,0.03)'; e.currentTarget.style.borderColor = 'rgba(6,182,212,0.1)'; }}
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-neutral-400 group-hover:text-cyan-400 transition-colors" />
          </button>
          <button
            onClick={handleExportReport}
            disabled={emotionStats.length === 0}
            className="flex items-center space-x-2 px-5 py-2.5 text-white rounded-xl text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ background: 'linear-gradient(135deg, #06b6d4, #0891b2)', boxShadow: '0 4px 15px rgba(6,182,212,0.3)' }}
            onMouseEnter={e => { if (!e.currentTarget.disabled) e.currentTarget.style.boxShadow = '0 8px 25px rgba(6,182,212,0.4)'; }}
            onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 4px 15px rgba(6,182,212,0.3)'; }}
          >
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {emotionStats.length === 0 ? (
        <div className="text-center py-16 rounded-2xl animate-fade-in-up" style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '100ms', animationFillMode: 'backwards' }}>
          <div className="text-6xl mb-4">📈</div>
          <h3 className="text-xl font-semibold text-white mb-2">No Data Yet</h3>
          <p style={{ color: '#525252', fontSize: '14px' }}>Analyze some audio files to see your emotion trends here.</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* ── Stat Cards with Animated Counters ── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 animate-fade-in-up" style={{ animationDelay: '100ms', animationFillMode: 'backwards' }}>
            {[
              { icon: BarChart3, label: 'Total Analyses', value: animatedTotal, color: '#06b6d4', gradient: 'linear-gradient(135deg, #06b6d4, #0e7490)' },
              { icon: PieChart, label: 'Unique Emotions', value: animatedUnique, color: '#3b82f6', gradient: 'linear-gradient(135deg, #3b82f6, #1d4ed8)' },
              { icon: Award, label: 'Avg Confidence', value: `${animatedAvg}%`, color: '#a855f7', gradient: 'linear-gradient(135deg, #a855f7, #7c3aed)' },
            ].map(({ icon: Icon, label, value, color, gradient }, idx) => (
              <div
                key={label}
                className="relative overflow-hidden rounded-2xl p-6 transition-all group animate-fade-in-up"
                style={{
                  background: gradient,
                  boxShadow: `0 8px 30px ${color}25`,
                  animationDelay: `${100 + idx * 80}ms`,
                  animationFillMode: 'backwards',
                }}
                onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = `0 16px 40px ${color}35`; }}
                onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = `0 8px 30px ${color}25`; }}
              >
                {/* Decorative circle */}
                <div className="absolute -top-6 -right-6 w-24 h-24 rounded-full opacity-15" style={{ background: 'white' }} />
                <div className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: `linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)` }} />
                <Icon className="w-7 h-7 text-white/70 mb-3 group-hover:scale-110 transition-transform" />
                <div className="text-3xl font-bold text-white tracking-tight">{value}</div>
                <div className="text-sm text-white/60 mt-1">{label}</div>
              </div>
            ))}
          </div>

          {/* ── Emotion Distribution ── */}
          <div
            className="rounded-2xl p-6 animate-fade-in-up"
            style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '350ms', animationFillMode: 'backwards' }}
          >
            <h3 className="text-lg font-semibold text-white mb-5 flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-cyan-400" />
              <span>Emotion Distribution</span>
            </h3>
            <div className="space-y-5">
              {emotionStats.map((stat, idx) => (
                <div
                  key={stat.emotion}
                  className="group rounded-xl p-3 -mx-3 transition-all"
                  style={{ animationDelay: `${400 + idx * 80}ms` }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(6,182,212,0.03)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl group-hover:scale-125 transition-transform inline-block">{stat.emoji}</span>
                      <span className="font-medium text-white capitalize">{stat.emotion}</span>
                    </div>
                    <div className="flex items-center space-x-4 text-sm">
                      <span style={{ color: '#525252' }}>
                        {stat.count} {stat.count === 1 ? 'analysis' : 'analyses'}
                      </span>
                      <span className="font-semibold" style={{ color: getEmotionSolid(stat.emotion) }}>
                        {(stat.avgConfidence * 100).toFixed(1)}% avg
                      </span>
                    </div>
                  </div>
                  <div className="relative">
                    <div className="w-full h-3 rounded-full overflow-hidden" style={{ background: '#171717' }}>
                      <div
                        className={`bg-gradient-to-r ${getEmotionColor(stat.emotion)} h-3 rounded-full transition-all`}
                        style={{
                          width: barsVisible ? `${(stat.count / maxCount) * 100}%` : '0%',
                          transition: 'width 1s cubic-bezier(0.22, 1, 0.36, 1)',
                          transitionDelay: `${400 + idx * 150}ms`,
                          boxShadow: `0 0 10px ${getEmotionSolid(stat.emotion)}40`,
                        }}
                      />
                    </div>
                    <span className="absolute right-1 -top-0.5 text-xs font-medium text-white/90 leading-3" style={{ textShadow: '0 1px 3px rgba(0,0,0,0.8)' }}>
                      {((stat.count / totalAnalyses) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── Key Insights ── */}
          <div
            className="rounded-2xl p-5 animate-fade-in-up"
            style={{ background: '#0a0a0a', border: '1px solid rgba(6,182,212,0.08)', animationDelay: '500ms', animationFillMode: 'backwards' }}
          >
            <h4 className="font-semibold text-white mb-3 flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span>Key Insights</span>
            </h4>
            <div className="space-y-2">
              {[
                <>Most detected emotion: <strong className="capitalize text-cyan-400">{emotionStats[0]?.emotion}</strong> ({emotionStats[0]?.count} times)</>,
                <>Highest confidence: <strong className="capitalize text-cyan-400">{emotionStats.reduce((max, e) => e.avgConfidence > max.avgConfidence ? e : max).emotion}</strong> ({(emotionStats.reduce((max, e) => e.avgConfidence > max.avgConfidence ? e : max).avgConfidence * 100).toFixed(1)}%)</>,
                <>Analysis diversity: <strong className="text-white">{emotionStats.length}</strong> different emotions detected</>,
              ].map((content, i) => (
                <div
                  key={i}
                  className="flex items-center space-x-3 rounded-lg p-2.5 transition-all"
                  style={{ background: 'rgba(6,182,212,0.02)' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(6,182,212,0.06)'; e.currentTarget.style.transform = 'translateX(4px)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'rgba(6,182,212,0.02)'; e.currentTarget.style.transform = 'translateX(0)'; }}
                >
                  <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: '#06b6d4' }} />
                  <span className="text-sm" style={{ color: '#a3a3a3' }}>{content}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
