import { Activity, Sparkles } from 'lucide-react';
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

    // Spawn particles
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

        // Wrap around edges
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(6, 182, 212, ${p.opacity})`;
        ctx.fill();
      });

      // Draw connection lines between nearby particles
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
    { id: 'dashboard', label: 'Dashboard', emoji: '📊' },
    { id: 'analyze', label: 'Analyze', emoji: '🎤' },
    { id: 'results', label: 'Results', emoji: '📋' },
    { id: 'reports', label: 'Reports', emoji: '📈' },
  ];

  return (
    <div className="min-h-screen relative" style={{ background: '#050505' }}>
      {/* Animated particle background */}
      <ParticleBackground />

      {/* Subtle ambient glow orbs */}
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

      {/* Header */}
      <header
        className="relative sticky top-0 z-50 pt-safe"
        style={{
          background: 'rgba(5, 5, 5, 0.8)',
          backdropFilter: 'blur(20px) saturate(180%)',
          borderBottom: '1px solid rgba(6, 182, 212, 0.08)',
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div
                className="p-2.5 rounded-xl shadow-lg"
                style={{
                  background: 'linear-gradient(135deg, #06b6d4, #0891b2)',
                  boxShadow: '0 4px 20px rgba(6, 182, 212, 0.3)',
                }}
              >
                <Activity className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white flex items-center gap-2">
                  AEMER
                  <Sparkles className="w-4 h-4" style={{ color: '#22d3ee' }} />
                </h1>
                <p className="text-xs" style={{ color: '#525252' }}>
                  Emotion Recognition AI
                </p>
              </div>
            </div>

            <div
              className="flex items-center space-x-2 rounded-full px-4 py-2"
              style={{
                background: 'linear-gradient(135deg, #059669, #10b981)',
                boxShadow: '0 4px 15px rgba(16, 185, 129, 0.3)',
              }}
            >
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="text-xs text-white font-medium">Live on Cloud</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav
        className="relative z-40"
        style={{
          background: 'rgba(5, 5, 5, 0.6)',
          backdropFilter: 'blur(12px)',
          borderBottom: '1px solid rgba(6, 182, 212, 0.06)',
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-1.5 py-3">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className="px-5 py-2.5 rounded-xl font-medium text-sm transition-all flex items-center gap-2"
                style={
                  currentPage === item.id
                    ? {
                      background: 'linear-gradient(135deg, #06b6d4, #0891b2)',
                      color: '#fff',
                      boxShadow: '0 4px 20px rgba(6, 182, 212, 0.3)',
                    }
                    : {
                      color: '#737373',
                      background: 'transparent',
                    }
                }
                onMouseEnter={(e) => {
                  if (currentPage !== item.id) {
                    e.currentTarget.style.color = '#d4d4d4';
                    e.currentTarget.style.background = 'rgba(6, 182, 212, 0.06)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (currentPage !== item.id) {
                    e.currentTarget.style.color = '#737373';
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                <span>{item.emoji}</span>
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div
          className="rounded-2xl p-6"
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

      {/* Footer */}
      <footer className="relative z-10 text-center py-8 mt-4 text-sm" style={{ color: '#525252' }}>
        <p>Built with 🎭 AI-Powered Emotion Recognition</p>
      </footer>
    </div>
  );
}
