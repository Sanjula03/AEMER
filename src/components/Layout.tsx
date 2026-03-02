import { Activity, Sparkles, LayoutDashboard, Mic, FileBarChart, PieChart } from 'lucide-react';
import { ReactNode, useEffect, useRef } from 'react';

interface LayoutProps {
  children: ReactNode;
  currentPage: string;
  onNavigate: (page: string) => void;
}

/**
 * Animated particle canvas background
 */
function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    const particles: Array<{
      x: number; y: number;
      vx: number; vy: number;
      size: number; opacity: number;
    }> = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 2 + 0.5,
        opacity: Math.random() * 0.3 + 0.05,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(6, 182, 212, ${p.opacity})`;
        ctx.fill();
      });

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(6, 182, 212, ${0.06 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      animId = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}

export function Layout({ children, currentPage, onNavigate }: LayoutProps) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'analyze', label: 'Analyze', icon: Mic },
    { id: 'results', label: 'Results', icon: FileBarChart },
    { id: 'reports', label: 'Reports', icon: PieChart },
  ];

  return (
    <div className="min-h-screen relative" style={{ background: '#050505' }}>
      <ParticleBackground />

      {/* Ambient glow orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
        <div
          className="absolute -top-32 -right-32 w-96 h-96 rounded-full blur-3xl"
          style={{ background: 'rgba(6, 182, 212, 0.06)' }}
        />
        <div
          className="absolute -bottom-32 -left-32 w-96 h-96 rounded-full blur-3xl"
          style={{ background: 'rgba(6, 182, 212, 0.04)' }}
        />
      </div>

      {/* ═══ TOP HEADER (visible on all sizes) ═══ */}
      <header
        className="relative sticky top-0 z-50"
        style={{
          background: 'rgba(5, 5, 5, 0.75)',
          backdropFilter: 'blur(24px) saturate(200%)',
          borderBottom: '1px solid rgba(6, 182, 212, 0.06)',
        }}
      >
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-14 sm:h-16">
            {/* Logo */}
            <div className="flex items-center space-x-2.5">
              <div
                className="p-1.5 sm:p-2 rounded-lg sm:rounded-xl"
                style={{
                  background: 'linear-gradient(135deg, #06b6d4, #0891b2)',
                  boxShadow: '0 4px 15px rgba(6, 182, 212, 0.25)',
                }}
              >
                <Activity className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
              </div>
              <div>
                <h1 className="text-base sm:text-lg font-bold text-white flex items-center gap-1.5" style={{ letterSpacing: '-0.02em' }}>
                  AEMER
                  <Sparkles className="w-3 h-3 sm:w-3.5 sm:h-3.5" style={{ color: '#22d3ee' }} />
                </h1>
                <p className="hidden sm:block" style={{ color: '#404040', fontSize: '11px', letterSpacing: '0.04em' }}>
                  Emotion Recognition AI
                </p>
              </div>
            </div>

            {/* ── Desktop Nav Pills (hidden on mobile) ── */}
            <nav className="hidden md:flex items-center">
              <div
                className="flex items-center rounded-2xl p-1"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.04)' }}
              >
                {navItems.map((item) => {
                  const isActive = currentPage === item.id;
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.id}
                      onClick={() => onNavigate(item.id)}
                      className="relative px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2"
                      style={
                        isActive
                          ? {
                            background: 'linear-gradient(135deg, rgba(6,182,212,0.15), rgba(6,182,212,0.08))',
                            color: '#22d3ee',
                            boxShadow: '0 0 20px rgba(6,182,212,0.1)',
                            border: '1px solid rgba(6,182,212,0.15)',
                          }
                          : {
                            color: '#525252',
                            background: 'transparent',
                            border: '1px solid transparent',
                          }
                      }
                      onMouseEnter={(e) => {
                        if (!isActive) {
                          e.currentTarget.style.color = '#a3a3a3';
                          e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) {
                          e.currentTarget.style.color = '#525252';
                          e.currentTarget.style.background = 'transparent';
                        }
                      }}
                    >
                      <Icon className="w-4 h-4" />
                      <span style={{ letterSpacing: '-0.01em' }}>{item.label}</span>
                      {isActive && (
                        <div
                          className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full"
                          style={{ background: '#06b6d4', boxShadow: '0 0 6px #06b6d4' }}
                        />
                      )}
                    </button>
                  );
                })}
              </div>
            </nav>

            {/* Live Badge */}
            <div
              className="flex items-center space-x-1.5 sm:space-x-2 rounded-full px-2.5 sm:px-3.5 py-1 sm:py-1.5"
              style={{
                background: 'rgba(16, 185, 129, 0.08)',
                border: '1px solid rgba(16, 185, 129, 0.15)',
              }}
            >
              <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: '#10b981', boxShadow: '0 0 6px #10b981' }} />
              <span className="text-[10px] sm:text-xs" style={{ color: '#10b981', fontWeight: 500 }}>Live</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content — add bottom padding on mobile for tab bar */}
      <main className="relative z-10 max-w-7xl mx-auto px-2 sm:px-6 lg:px-8 py-4 sm:py-8 pb-24 md:pb-8">
        <div
          className="rounded-xl sm:rounded-2xl p-3 sm:p-6"
          style={{
            background: 'rgba(10, 10, 10, 0.7)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(6, 182, 212, 0.08)',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.4)',
          }}
        >
          {children}
        </div>
      </main>

      {/* Footer (hidden on mobile to save space) */}
      <footer className="hidden md:block relative z-10 text-center py-8 mt-4 text-sm" style={{ color: '#525252' }}>
        <p>Built with 🎭 AI-Powered Emotion Recognition</p>
      </footer>

      {/* ═══ MOBILE BOTTOM TAB BAR ═══ */}
      <nav
        className="md:hidden fixed bottom-0 left-0 right-0 z-50"
        style={{
          background: 'rgba(5, 5, 5, 0.92)',
          backdropFilter: 'blur(24px) saturate(200%)',
          borderTop: '1px solid rgba(6, 182, 212, 0.08)',
          paddingBottom: 'env(safe-area-inset-bottom)',
        }}
      >
        <div className="flex justify-around items-center h-16">
          {navItems.map((item) => {
            const isActive = currentPage === item.id;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className="flex flex-col items-center justify-center gap-0.5 w-full h-full transition-all"
                style={{ color: isActive ? '#22d3ee' : '#525252' }}
              >
                <div
                  className="relative p-1.5 rounded-xl transition-all"
                  style={isActive ? {
                    background: 'rgba(6,182,212,0.1)',
                    boxShadow: '0 0 12px rgba(6,182,212,0.15)',
                  } : {}}
                >
                  <Icon className="w-5 h-5" />
                  {isActive && (
                    <div
                      className="absolute -top-0.5 left-1/2 -translate-x-1/2 w-4 h-0.5 rounded-full"
                      style={{ background: '#06b6d4', boxShadow: '0 0 6px #06b6d4' }}
                    />
                  )}
                </div>
                <span className="text-[10px] font-medium">{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
