import { Activity, TrendingUp, Sparkles, ArrowRight, RefreshCw, Zap } from 'lucide-react';
import { useEffect, useState, useRef } from 'react';
import { getStoredResults, type AnalysisResult } from '../lib/storage';

interface DashboardStats {
  totalAnalyses: number;
  avgConfidence: number;
  topEmotion: string;
}

interface DashboardProps {
  onNavigate: (page: string) => void;
}

/** Animated counter component — counts up from 0 to target */
function AnimatedCounter({ target, suffix = '', decimals = 0 }: { target: number; suffix?: string; decimals?: number }) {
  const [count, setCount] = useState(0);
  const ref = useRef<number>(0);

  useEffect(() => {
    const duration = 1200;
    const startTime = performance.now();
    const startVal = ref.current;

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = startVal + (target - startVal) * eased;
      setCount(val);
      ref.current = val;
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [target]);

  return <>{decimals > 0 ? count.toFixed(decimals) : Math.round(count)}{suffix}</>;
}

/** Typing animation component */
function TypingText({ text, speed = 40 }: { text: string; speed?: number }) {
  const [displayed, setDisplayed] = useState('');
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    let i = 0;
    setDisplayed('');
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
        setTimeout(() => setShowCursor(false), 2000);
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);

  return (
    <span>
      {displayed}
      {showCursor && (
        <span
          className="inline-block w-0.5 h-5 ml-0.5 align-middle"
          style={{
            background: '#06b6d4',
            animation: 'typing-cursor 0.8s step-end infinite',
          }}
        />
      )}
    </span>
  );
}

export function Dashboard({ onNavigate }: DashboardProps) {
  const [stats, setStats] = useState<DashboardStats>({
    totalAnalyses: 0,
    avgConfidence: 0,
    topEmotion: 'N/A',
  });
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = () => {
    setLoading(true);
    try {
      const analyses = getStoredResults();
      if (analyses && analyses.length > 0) {
        const total = analyses.length;
        const avgConf = analyses.reduce((sum, a) => sum + (a.confidence_score || 0), 0) / total;
        const emotionCounts: Record<string, number> = {};
        analyses.forEach((a) => {
          emotionCounts[a.emotion_label] = (emotionCounts[a.emotion_label] || 0) + 1;
        });
        const topEmo = Object.keys(emotionCounts).reduce((a, b) =>
          emotionCounts[a] > emotionCounts[b] ? a : b
        );
        setStats({ totalAnalyses: total, avgConfidence: avgConf, topEmotion: topEmo });
        setRecentAnalyses(analyses.slice(0, 5));
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
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
      happy: '#facc15', sad: '#60a5fa', angry: '#f87171',
      neutral: '#a3a3a3', fear: '#a78bfa', surprise: '#f472b6', disgust: '#4ade80',
    };
    return colors[emotion.toLowerCase()] || '#22d3ee';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-3">
          <div
            className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin"
            style={{ borderColor: '#06b6d4', borderTopColor: 'transparent' }}
          />
          <span style={{ color: '#737373' }}>Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* ── Hero Section ── */}
      <div
        className="relative overflow-hidden rounded-2xl p-8 animate-fade-in-up"
        style={{
          background: 'linear-gradient(135deg, rgba(6,182,212,0.15) 0%, rgba(6,182,212,0.05) 100%)',
          border: '1px solid rgba(6,182,212,0.15)',
        }}
      >
        {/* Decorative elements */}
        <div className="absolute top-0 right-0 w-72 h-72 rounded-full blur-3xl" style={{ background: 'rgba(6,182,212,0.08)' }} />
        <div className="absolute bottom-0 left-0 w-48 h-48 rounded-full blur-3xl" style={{ background: 'rgba(6,182,212,0.05)' }} />

        <div className="relative flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-2xl sm:text-4xl font-bold text-white mb-2 sm:mb-3 flex items-center gap-2 sm:gap-3">
              <Sparkles className="w-6 h-6 sm:w-9 sm:h-9" style={{ color: '#22d3ee' }} />
              Welcome to AEMER
            </h2>
            <p className="text-sm sm:text-lg" style={{ color: '#a3a3a3' }}>
              <TypingText text="AI-Powered Emotion Recognition at Your Fingertips" />
            </p>
          </div>
          <button
            onClick={(e) => {
              const icon = e.currentTarget.querySelector('svg');
              if (icon) { icon.style.animation = 'spin 0.6s ease'; setTimeout(() => { icon.style.animation = ''; }, 600); }
              loadDashboardData();
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
        </div>
      </div>

      {/* ── Stats Grid ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Total Analyses */}
        <div
          className="relative overflow-hidden rounded-2xl p-6 animate-fade-in-up"
          style={{
            background: '#111111',
            border: '1px solid rgba(6,182,212,0.12)',
            animationDelay: '100ms',
            animationFillMode: 'backwards',
          }}
        >
          <div className="absolute top-0 left-0 right-0 h-px" style={{ background: 'linear-gradient(90deg, transparent, #06b6d4, transparent)' }} />
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-lg" style={{ background: 'rgba(6,182,212,0.1)' }}>
              <Activity className="w-5 h-5" style={{ color: '#06b6d4' }} />
            </div>
            <span className="text-sm font-medium" style={{ color: '#737373' }}>Total Analyses</span>
          </div>
          <div className="text-4xl font-extrabold text-white">
            <AnimatedCounter target={stats.totalAnalyses} />
          </div>
        </div>

        {/* Avg Confidence */}
        <div
          className="relative overflow-hidden rounded-2xl p-6 animate-fade-in-up"
          style={{
            background: '#111111',
            border: '1px solid rgba(16,185,129,0.12)',
            animationDelay: '200ms',
            animationFillMode: 'backwards',
          }}
        >
          <div className="absolute top-0 left-0 right-0 h-px" style={{ background: 'linear-gradient(90deg, transparent, #10b981, transparent)' }} />
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-lg" style={{ background: 'rgba(16,185,129,0.1)' }}>
              <TrendingUp className="w-5 h-5" style={{ color: '#10b981' }} />
            </div>
            <span className="text-sm font-medium" style={{ color: '#737373' }}>Avg Confidence</span>
          </div>
          <div className="text-4xl font-extrabold text-white">
            <AnimatedCounter target={stats.avgConfidence * 100} suffix="%" decimals={1} />
          </div>
        </div>

        {/* Top Emotion */}
        <div
          className="relative overflow-hidden rounded-2xl p-6 animate-fade-in-up"
          style={{
            background: '#111111',
            border: `1px solid ${getEmotionColor(stats.topEmotion)}20`,
            animationDelay: '300ms',
            animationFillMode: 'backwards',
          }}
        >
          <div className="absolute top-0 left-0 right-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${getEmotionColor(stats.topEmotion)}, transparent)` }} />
          <div className="flex items-center gap-3 mb-4">
            <span className="text-3xl">{getEmotionEmoji(stats.topEmotion)}</span>
            <span className="text-sm font-medium" style={{ color: '#737373' }}>Most Detected</span>
          </div>
          <div className="text-4xl font-extrabold text-white capitalize">
            {stats.topEmotion}
          </div>
        </div>
      </div>

      {/* ── Quick Actions + Recent ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quick Actions */}
        <div
          className="rounded-2xl p-6 animate-fade-in-up"
          style={{
            background: '#111111',
            border: '1px solid rgba(6,182,212,0.08)',
            animationDelay: '400ms',
            animationFillMode: 'backwards',
          }}
        >
          <h3 className="text-xl font-bold text-white mb-5 flex items-center gap-2">
            <Zap className="w-5 h-5" style={{ color: '#22d3ee' }} />
            Quick Actions
          </h3>
          <div className="space-y-3">
            {[
              {
                page: 'analyze', label: 'New Analysis',
                desc: 'Upload audio, video or text & detect emotions',
                icon: Activity, color: '#06b6d4',
              },
              {
                page: 'results', label: 'View Results',
                desc: 'Browse analysis history & detailed reports',
                icon: TrendingUp, color: '#10b981',
              },
              {
                page: 'reports', label: 'View Reports',
                desc: 'Analytics, trends & insights',
                icon: Sparkles, color: '#f59e0b',
              },
            ].map(({ page, label, desc, icon: Icon, color }) => (
              <button
                key={page}
                onClick={() => onNavigate(page)}
                className="w-full flex items-center justify-between p-4 rounded-xl transition-all hover:scale-[1.01] group"
                style={{
                  background: `${color}08`,
                  border: `1px solid ${color}15`,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = `${color}40`;
                  e.currentTarget.style.boxShadow = `0 4px 20px ${color}15`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = `${color}15`;
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div className="flex items-center space-x-4">
                  <div className="p-2.5 rounded-lg" style={{ background: `${color}15` }}>
                    <Icon className="w-5 h-5" style={{ color }} />
                  </div>
                  <div className="text-left">
                    <div className="font-semibold text-white">{label}</div>
                    <div className="text-xs" style={{ color: '#737373' }}>{desc}</div>
                  </div>
                </div>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" style={{ color: '#525252' }} />
              </button>
            ))}
          </div>
        </div>

        {/* Recent Analyses */}
        <div
          className="rounded-2xl p-6 animate-fade-in-up"
          style={{
            background: '#111111',
            border: '1px solid rgba(6,182,212,0.08)',
            animationDelay: '500ms',
            animationFillMode: 'backwards',
          }}
        >
          <h3 className="text-xl font-bold text-white mb-5">Recent Analyses</h3>
          {recentAnalyses.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-5xl mb-4 animate-float">🎤</div>
              <p style={{ color: '#525252' }}>No analyses yet. Start by uploading a file!</p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {recentAnalyses.map((analysis, idx) => (
                <div
                  key={analysis.id}
                  className="flex items-center justify-between p-4 rounded-xl cursor-pointer transition-all hover:scale-[1.01] animate-fade-in-up"
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.04)',
                    animationDelay: `${600 + idx * 80}ms`,
                    animationFillMode: 'backwards',
                  }}
                  onClick={() => onNavigate('results')}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = `${getEmotionColor(analysis.emotion_label)}30`;
                    e.currentTarget.style.background = `${getEmotionColor(analysis.emotion_label)}08`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.04)';
                    e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                  }}
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{getEmotionEmoji(analysis.emotion_label)}</span>
                    <div>
                      <div className="font-semibold text-white capitalize text-sm">{analysis.emotion_label}</div>
                      <div className="text-xs" style={{ color: '#525252' }}>
                        {new Date(analysis.timestamp).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                  <div className="text-lg font-bold" style={{ color: getEmotionColor(analysis.emotion_label) }}>
                    {(analysis.confidence_score * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
