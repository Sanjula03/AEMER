import { Activity, Sparkles } from 'lucide-react';
import { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
  currentPage: string;
  onNavigate: (page: string) => void;
}

export function Layout({ children, currentPage, onNavigate }: LayoutProps) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', emoji: '📊' },
    { id: 'analyze', label: 'Analyze', emoji: '🎤' },
    { id: 'results', label: 'Results', emoji: '📋' },
    { id: 'reports', label: 'Reports', emoji: '📈' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-800 via-amber-900 to-stone-900 relative overflow-hidden">
      {/* Subtle background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-20 -right-20 w-72 h-72 bg-gradient-to-br from-amber-600 to-orange-700 rounded-full opacity-20 blur-3xl" />
        <div className="absolute -bottom-20 -left-20 w-72 h-72 bg-gradient-to-br from-rose-700 to-amber-800 rounded-full opacity-20 blur-3xl" />
        <div className="absolute top-1/3 right-1/4 w-64 h-64 bg-gradient-to-br from-yellow-600 to-amber-700 rounded-full opacity-15 blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative bg-stone-900/80 backdrop-blur-lg border-b border-amber-900/30 sticky top-0 z-50 pt-safe">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl shadow-lg shadow-amber-500/30">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-amber-100 flex items-center gap-2">
                  AEMER
                  <Sparkles className="w-4 h-4 text-yellow-400" />
                </h1>
                <p className="text-xs text-amber-200/60">Emotion Recognition AI</p>
              </div>
            </div>

            <div className="flex items-center space-x-2 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-full px-4 py-2 shadow-lg shadow-emerald-500/30">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="text-xs text-white font-medium">Live on Cloud</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="relative bg-stone-900/50 backdrop-blur-sm border-b border-amber-900/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-2 py-3">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`px-5 py-2.5 rounded-xl font-medium text-sm transition-all flex items-center gap-2 ${currentPage === item.id
                  ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg shadow-amber-500/30'
                  : 'text-amber-200/70 hover:bg-amber-900/30 hover:text-amber-100'
                  }`}
              >
                <span>{item.emoji}</span>
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-stone-900/40 backdrop-blur-xl rounded-3xl border border-amber-900/20 p-6 shadow-2xl">
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-8 mt-8 text-amber-200/40 text-sm">
        <p>Built with 🎭 AI-Powered Emotion Recognition</p>
      </footer>
    </div>
  );
}
