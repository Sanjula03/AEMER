import { Activity, TrendingUp, Sparkles, ArrowRight, RefreshCw, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getStoredResults, type AnalysisResult } from '../lib/storage';

interface DashboardStats {
  totalAnalyses: number;
  avgConfidence: number;
  topEmotion: string;
}

interface DashboardProps {
  onNavigate: (page: string) => void;
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

        setStats({
          totalAnalyses: total,
          avgConfidence: avgConf,
          topEmotion: topEmo,
        });

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

  const getEmotionGradient = (emotion: string) => {
    const gradients: Record<string, string> = {
      happy: 'from-yellow-400 via-orange-400 to-pink-500',
      sad: 'from-blue-400 via-indigo-500 to-purple-600',
      angry: 'from-red-400 via-rose-500 to-pink-600',
      neutral: 'from-gray-400 via-slate-500 to-gray-600',
      fear: 'from-purple-400 via-violet-500 to-indigo-600',
      surprise: 'from-pink-400 via-fuchsia-500 to-rose-600',
      disgust: 'from-green-400 via-emerald-500 to-teal-600',
    };
    return gradients[emotion.toLowerCase()] || 'from-teal-400 to-blue-500';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-600 font-medium">Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header with gradient */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-purple-600 via-pink-500 to-orange-400 p-8 text-white">
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-32 translate-x-32" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/10 rounded-full translate-y-24 -translate-x-24" />
        <div className="relative">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-4xl font-bold mb-2 flex items-center gap-3">
                <Sparkles className="w-10 h-10" />
                Welcome to AEMER
              </h2>
              <p className="text-white/80 text-lg">
                AI-Powered Emotion Recognition at Your Fingertips
              </p>
            </div>
            <button
              onClick={loadDashboardData}
              className="p-3 bg-white/20 hover:bg-white/30 rounded-xl transition-all hover:scale-105"
            >
              <RefreshCw className="w-6 h-6" />
            </button>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Total Analyses */}
        <div className="relative overflow-hidden bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl p-6 text-white shadow-xl shadow-blue-500/25">
          <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -translate-y-8 translate-x-8" />
          <Activity className="w-10 h-10 mb-4" />
          <div className="text-5xl font-bold mb-2">{stats.totalAnalyses}</div>
          <div className="text-white/80 font-medium">Total Analyses</div>
        </div>

        {/* Avg Confidence */}
        <div className="relative overflow-hidden bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl p-6 text-white shadow-xl shadow-emerald-500/25">
          <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -translate-y-8 translate-x-8" />
          <TrendingUp className="w-10 h-10 mb-4" />
          <div className="text-5xl font-bold mb-2">{(stats.avgConfidence * 100).toFixed(1)}%</div>
          <div className="text-white/80 font-medium">Avg Confidence</div>
        </div>

        {/* Top Emotion */}
        <div className={`relative overflow-hidden bg-gradient-to-br ${getEmotionGradient(stats.topEmotion)} rounded-2xl p-6 text-white shadow-xl`}>
          <div className="absolute top-0 right-0 w-24 h-24 bg-white/10 rounded-full -translate-y-8 translate-x-8" />
          <div className="text-4xl mb-4">{getEmotionEmoji(stats.topEmotion)}</div>
          <div className="text-4xl font-bold mb-2 capitalize">{stats.topEmotion}</div>
          <div className="text-white/80 font-medium">Most Detected</div>
        </div>
      </div>

      {/* Quick Actions & Recent */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quick Actions */}
        <div className="bg-stone-800/50 rounded-2xl shadow-xl p-6 border border-amber-900/30">
          <h3 className="text-xl font-bold text-amber-100 mb-5 flex items-center gap-2">
            <Zap className="w-6 h-6 text-yellow-400" />
            Quick Actions
          </h3>
          <div className="space-y-4">
            <button
              onClick={() => onNavigate('analyze')}
              className="w-full flex items-center justify-between p-5 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl transition-all hover:scale-[1.02] hover:shadow-lg group"
            >
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-white/20 rounded-lg">
                  <Activity className="w-6 h-6" />
                </div>
                <div className="text-left">
                  <div className="font-bold text-lg">New Analysis</div>
                  <div className="text-white/80 text-sm">Upload audio & detect emotions</div>
                </div>
              </div>
              <ArrowRight className="w-6 h-6 group-hover:translate-x-2 transition-transform" />
            </button>

            <button
              onClick={() => onNavigate('results')}
              className="w-full flex items-center justify-between p-5 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl transition-all hover:scale-[1.02] hover:shadow-lg group"
            >
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-white/20 rounded-lg">
                  <TrendingUp className="w-6 h-6" />
                </div>
                <div className="text-left">
                  <div className="font-bold text-lg">View Results</div>
                  <div className="text-white/80 text-sm">Browse analysis history</div>
                </div>
              </div>
              <ArrowRight className="w-6 h-6 group-hover:translate-x-2 transition-transform" />
            </button>

            <button
              onClick={() => onNavigate('reports')}
              className="w-full flex items-center justify-between p-5 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-xl transition-all hover:scale-[1.02] hover:shadow-lg group"
            >
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-white/20 rounded-lg">
                  <Sparkles className="w-6 h-6" />
                </div>
                <div className="text-left">
                  <div className="font-bold text-lg">View Reports</div>
                  <div className="text-white/80 text-sm">Analytics & trends</div>
                </div>
              </div>
              <ArrowRight className="w-6 h-6 group-hover:translate-x-2 transition-transform" />
            </button>
          </div>
        </div>

        {/* Recent Analyses */}
        <div className="bg-stone-800/50 rounded-2xl shadow-xl p-6 border border-amber-900/30">
          <h3 className="text-xl font-bold text-amber-100 mb-5">Recent Analyses</h3>
          {recentAnalyses.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4 float-animation">🎤</div>
              <p className="text-amber-200/60">No analyses yet. Start by uploading an audio file!</p>
            </div>
          ) : (
            <div className="space-y-3">
              {recentAnalyses.map((analysis) => (
                <div
                  key={analysis.id}
                  className={`flex items-center justify-between p-4 rounded-xl bg-gradient-to-r ${getEmotionGradient(analysis.emotion_label)} text-white cursor-pointer hover:scale-[1.02] transition-all`}
                  onClick={() => onNavigate('results')}
                >
                  <div className="flex items-center space-x-4">
                    <span className="text-3xl">{getEmotionEmoji(analysis.emotion_label)}</span>
                    <div>
                      <div className="font-bold capitalize">{analysis.emotion_label}</div>
                      <div className="text-sm text-white/80">
                        {new Date(analysis.timestamp).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold">
                      {(analysis.confidence_score * 100).toFixed(0)}%
                    </div>
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
