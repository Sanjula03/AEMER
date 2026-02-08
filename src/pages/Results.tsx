import { useEffect, useState } from 'react';
import { Download, Calendar, Trash2, RefreshCw } from 'lucide-react';
import { getStoredResults, deleteResult, clearResults, type AnalysisResult } from '../lib/storage';

export function Results() {
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);

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
  };

  const handleDelete = (id: string) => {
    deleteResult(id);
    loadAnalyses();
  };

  const handleClearAll = () => {
    if (confirm('Are you sure you want to delete all results?')) {
      clearResults();
      loadAnalyses();
      setSelectedAnalysis(null);
    }
  };

  const handleExport = (format: 'json' | 'csv') => {
    if (!selectedAnalysis) return;

    if (format === 'json') {
      const dataStr = JSON.stringify(selectedAnalysis, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `analysis-${selectedAnalysis.id.slice(0, 8)}.json`;
      link.click();
    } else if (format === 'csv') {
      const headers = Object.keys(selectedAnalysis).join(',');
      const values = Object.values(selectedAnalysis).map(v =>
        typeof v === 'string' ? `"${v}"` : v
      ).join(',');
      const csv = `${headers}\n${values}`;
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `analysis-${selectedAnalysis.id.slice(0, 8)}.csv`;
      link.click();
    }
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
      happy: 'text-yellow-600 bg-yellow-50 border-yellow-200',
      sad: 'text-blue-600 bg-blue-50 border-blue-200',
      angry: 'text-red-600 bg-red-50 border-red-200',
      neutral: 'text-gray-600 bg-gray-50 border-gray-200',
    };
    return colors[emotion.toLowerCase()] || colors.neutral;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-amber-200/70">Loading results...</div>
      </div>
    );
  }

  if (analyses.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">📊</div>
        <h3 className="text-xl font-semibold text-amber-100 mb-2">No Results Yet</h3>
        <p className="text-amber-200/60">Upload an audio file in the Analyze tab to see your results here.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-amber-100 mb-2">Results</h2>
          <p className="text-amber-200/70">
            Your emotion analysis history ({analyses.length} total)
          </p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={loadAnalyses}
            className="p-2 text-amber-200 hover:text-amber-100 hover:bg-amber-900/30 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={handleClearAll}
            className="px-3 py-2 text-red-400 hover:bg-red-900/30 rounded-lg transition-colors text-sm"
          >
            Clear All
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left sidebar - Analysis list */}
        <div className="lg:col-span-1 bg-stone-800/50 border border-amber-900/30 rounded-xl p-4 max-h-[600px] overflow-y-auto">
          <h3 className="font-semibold text-amber-100 mb-3">Recent Analyses</h3>
          <div className="space-y-2">
            {analyses.map((analysis) => (
              <div
                key={analysis.id}
                className={`relative group p-3 rounded-lg transition-colors cursor-pointer ${selectedAnalysis?.id === analysis.id
                  ? 'bg-amber-900/40 border border-amber-600/50'
                  : 'bg-stone-700/50 hover:bg-stone-700 border border-transparent'
                  }`}
                onClick={() => setSelectedAnalysis(analysis)}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">{getEmotionEmoji(analysis.emotion_label)}</span>
                  <div className="flex-1">
                    <div className="font-medium text-amber-100 capitalize">
                      {analysis.emotion_label}
                    </div>
                    <div className="text-xs text-amber-200/60">
                      {new Date(analysis.timestamp).toLocaleString()}
                    </div>
                    {analysis.filename && (
                      <div className="text-xs text-amber-400 truncate">
                        {analysis.filename}
                      </div>
                    )}
                  </div>
                  <div className="text-sm font-medium text-amber-400">
                    {(analysis.confidence_score * 100).toFixed(0)}%
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(analysis.id);
                  }}
                  className="absolute top-2 right-2 p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Right side - Selected analysis details */}
        <div className="lg:col-span-2 space-y-6">
          {selectedAnalysis && (
            <>
              {/* Main result card */}
              <div className="bg-stone-800/50 border border-amber-900/30 rounded-xl p-6">
                <div className="flex items-start justify-between mb-6">
                  <div className="flex items-center space-x-4">
                    <div className="text-5xl">{getEmotionEmoji(selectedAnalysis.emotion_label)}</div>
                    <div>
                      <div className={`inline-block px-4 py-2 rounded-lg border-2 ${getEmotionColor(selectedAnalysis.emotion_label)}`}>
                        <span className="text-3xl font-bold capitalize">
                          {selectedAnalysis.emotion_label}
                        </span>
                      </div>
                      {selectedAnalysis.filename && (
                        <div className="text-sm text-amber-200/60 mt-2">
                          File: {selectedAnalysis.filename}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleExport('json')}
                      className="p-2 text-amber-200 hover:text-amber-100 hover:bg-amber-900/30 rounded-lg transition-colors"
                      title="Export as JSON"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Confidence Score */}
                <div className="bg-stone-700/50 rounded-xl p-4 mb-6">
                  <div className="text-sm text-amber-200/70 mb-2">Confidence Score</div>
                  <div className="flex items-center space-x-4">
                    <div className="text-3xl font-bold text-amber-100">
                      {(selectedAnalysis.confidence_score * 100).toFixed(1)}%
                    </div>
                    <div className="flex-1">
                      <div className="w-full bg-stone-600 rounded-full h-3">
                        <div
                          className="bg-teal-600 h-3 rounded-full transition-all"
                          style={{ width: `${selectedAnalysis.confidence_score * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* All Probabilities */}
                {selectedAnalysis.all_probabilities && Object.keys(selectedAnalysis.all_probabilities).length > 0 && (
                  <div>
                    <h4 className="font-semibold text-amber-100 mb-3">All Emotion Probabilities</h4>
                    <div className="space-y-3">
                      {Object.entries(selectedAnalysis.all_probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .map(([emotion, probability]) => (
                          <div key={emotion} className="flex items-center space-x-3">
                            <span className="text-xl w-8">{getEmotionEmoji(emotion)}</span>
                            <span className="w-20 text-amber-200 capitalize font-medium">{emotion}</span>
                            <div className="flex-1 bg-stone-700 rounded-full h-2">
                              <div
                                className="bg-teal-500 h-2 rounded-full"
                                style={{ width: `${probability * 100}%` }}
                              />
                            </div>
                            <span className="w-16 text-right text-sm font-medium text-amber-200/70">
                              {(probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Metadata card */}
              <div className="bg-stone-800/50 border border-amber-900/30 rounded-xl p-6">
                <h4 className="font-semibold text-amber-100 mb-3 flex items-center space-x-2">
                  <Calendar className="w-5 h-5" />
                  <span>Analysis Details</span>
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-amber-200/70">Analysis ID:</span>
                    <span className="font-mono text-amber-100">{selectedAnalysis.id.slice(0, 8)}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-amber-200/70">Timestamp:</span>
                    <span className="text-amber-100">
                      {new Date(selectedAnalysis.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-amber-200/70">Input Type:</span>
                    <span className="text-amber-100 capitalize">{selectedAnalysis.input_type}</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
