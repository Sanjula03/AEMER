import { useState, useEffect } from 'react';
import { Download, BarChart3, TrendingUp, RefreshCw } from 'lucide-react';
import { getStoredResults, type AnalysisResult } from '../lib/storage';

interface EmotionStats {
  emotion: string;
  count: number;
  avgConfidence: number;
  emoji: string;
}

export function Reports() {
  const [emotionStats, setEmotionStats] = useState<EmotionStats[]>([]);
  const [totalAnalyses, setTotalAnalyses] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReportData();
  }, []);

  const loadReportData = () => {
    setLoading(true);
    const analyses = getStoredResults();
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
      happy: '😊',
      sad: '😢',
      angry: '😠',
      neutral: '😐',
    };
    return emojis[emotion.toLowerCase()] || '🎭';
  };

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: 'from-yellow-400 to-orange-500',
      sad: 'from-blue-400 to-blue-600',
      angry: 'from-red-400 to-red-600',
      neutral: 'from-gray-400 to-gray-600',
    };
    return colors[emotion.toLowerCase()] || 'from-teal-400 to-teal-600';
  };

  const handleExportReport = () => {
    const report = {
      generated_at: new Date().toISOString(),
      total_analyses: totalAnalyses,
      emotion_breakdown: emotionStats,
    };
    const dataStr = JSON.stringify(report, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `emotion-report-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-amber-200/70">Loading reports...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-amber-100 mb-2">Reports</h2>
          <p className="text-amber-200/70">
            Analytics and trends from your emotion analysis history
          </p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={loadReportData}
            className="p-2 text-amber-200 hover:text-amber-100 hover:bg-amber-900/30 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={handleExportReport}
            disabled={emotionStats.length === 0}
            className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-lg hover:from-amber-600 hover:to-orange-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-amber-500/30"
          >
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {emotionStats.length === 0 ? (
        <div className="text-center py-12 bg-stone-800/50 border border-amber-900/30 rounded-xl">
          <div className="text-6xl mb-4">📈</div>
          <h3 className="text-xl font-semibold text-amber-100 mb-2">No Data Yet</h3>
          <p className="text-amber-200/60">Analyze some audio files to see your emotion trends here.</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-br from-teal-600 to-emerald-700 rounded-xl p-6 shadow-lg shadow-teal-800/30">
              <BarChart3 className="w-8 h-8 text-white/80 mb-2" />
              <div className="text-3xl font-bold text-white">{totalAnalyses}</div>
              <div className="text-sm text-white/70">Total Analyses</div>
            </div>

            <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl p-6 shadow-lg shadow-blue-800/30">
              <TrendingUp className="w-8 h-8 text-white/80 mb-2" />
              <div className="text-3xl font-bold text-white">{emotionStats.length}</div>
              <div className="text-sm text-white/70">Unique Emotions</div>
            </div>

            <div className="bg-gradient-to-br from-purple-600 to-pink-700 rounded-xl p-6 shadow-lg shadow-purple-800/30">
              <TrendingUp className="w-8 h-8 text-white/80 mb-2" />
              <div className="text-3xl font-bold text-white">
                {emotionStats.length > 0
                  ? (
                    (emotionStats.reduce((sum, e) => sum + e.avgConfidence, 0) /
                      emotionStats.length) *
                    100
                  ).toFixed(1)
                  : 0}%
              </div>
              <div className="text-sm text-white/70">Avg Confidence</div>
            </div>
          </div>

          {/* Emotion Distribution */}
          <div className="bg-stone-800/50 border border-amber-900/30 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-amber-100 mb-4">
              Emotion Distribution
            </h3>
            <div className="space-y-4">
              {emotionStats.map((stat) => (
                <div key={stat.emotion} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-2xl">{stat.emoji}</span>
                      <span className="font-medium text-amber-100 capitalize">
                        {stat.emotion}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-sm">
                      <span className="text-amber-200/70">
                        {stat.count} {stat.count === 1 ? 'analysis' : 'analyses'}
                      </span>
                      <span className="text-amber-400 font-medium">
                        {(stat.avgConfidence * 100).toFixed(1)}% avg
                      </span>
                    </div>
                  </div>
                  <div className="relative">
                    <div className="w-full bg-stone-700 rounded-full h-4">
                      <div
                        className={`bg-gradient-to-r ${getEmotionColor(stat.emotion)} h-4 rounded-full transition-all`}
                        style={{
                          width: `${(stat.count / Math.max(...emotionStats.map((e) => e.count))) * 100
                            }%`,
                        }}
                      />
                    </div>
                    <span className="absolute right-2 top-0 text-xs font-medium text-white leading-4">
                      {((stat.count / totalAnalyses) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Key Insights */}
          <div className="bg-stone-800/50 border border-amber-900/30 rounded-xl p-4">
            <h4 className="font-medium text-amber-100 mb-2">📊 Key Insights</h4>
            <ul className="text-sm text-amber-200/80 space-y-1">
              <li>
                Most detected emotion: <strong className="capitalize text-amber-100">{emotionStats[0]?.emotion}</strong> ({emotionStats[0]?.count} times)
              </li>
              <li>
                Highest confidence: <strong className="capitalize text-amber-100">
                  {emotionStats.reduce((max, e) => e.avgConfidence > max.avgConfidence ? e : max).emotion}
                </strong> ({(emotionStats.reduce((max, e) => e.avgConfidence > max.avgConfidence ? e : max).avgConfidence * 100).toFixed(1)}%)
              </li>
              <li>
                Analysis diversity: {emotionStats.length} different emotions detected
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
