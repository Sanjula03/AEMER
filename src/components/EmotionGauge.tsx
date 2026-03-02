/**
 * EmotionGauge — Modern SVG semi-circular gauge with gradient arc,
 * glow effects, tick marks, animated needle sweep, and HTML text overlay.
 */

import { useEffect, useRef, useState } from 'react';

interface EmotionGaugeProps {
    value: number; // 0 to 1
    label?: string;
    size?: number;
}

export function EmotionGauge({ value, label, size = 220 }: EmotionGaugeProps) {
    const targetValue = Math.min(Math.max(value, 0), 1);

    // ── Animated value (sweeps from 0 → target) ──
    const [animatedValue, setAnimatedValue] = useState(0);
    const animRef = useRef<number | null>(null);
    const prevTarget = useRef(0);

    useEffect(() => {
        const from = prevTarget.current;
        const to = targetValue;
        prevTarget.current = to;

        if (from === to) {
            setAnimatedValue(to);
            return;
        }

        const duration = 1200; // ms
        const startTime = performance.now();

        const animate = (now: number) => {
            const elapsed = now - startTime;
            const t = Math.min(elapsed / duration, 1);
            // easeOutExpo for a satisfying sweep
            const ease = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
            setAnimatedValue(from + (to - from) * ease);
            if (t < 1) {
                animRef.current = requestAnimationFrame(animate);
            }
        };

        animRef.current = requestAnimationFrame(animate);
        return () => {
            if (animRef.current) cancelAnimationFrame(animRef.current);
        };
    }, [targetValue]);

    const clampedValue = animatedValue;
    const percentage = Math.round(clampedValue * 100);

    const cx = size / 2;
    const cy = size * 0.52;
    const radius = size * 0.38;
    const strokeWidth = 12;

    // Arc calculations (semicircle from left to right)
    const startAngle = Math.PI;
    const endAngle = 0;

    const describeArc = (startA: number, endA: number) => {
        const x1 = cx + radius * Math.cos(startA);
        const y1 = cy - radius * Math.sin(startA);
        const x2 = cx + radius * Math.cos(endA);
        const y2 = cy - radius * Math.sin(endA);
        const sweep = startA - endA > Math.PI ? 1 : 0;
        return `M ${x1} ${y1} A ${radius} ${radius} 0 ${sweep} 1 ${x2} ${y2}`;
    };

    const sweepAngle = startAngle - (startAngle - endAngle) * clampedValue;
    const bgPath = describeArc(startAngle, endAngle);
    const valuePath = clampedValue > 0.01 ? describeArc(startAngle, sweepAngle) : '';

    // Color stops for gradient
    const getColor = () => {
        if (clampedValue >= 0.8) return { main: '#34d399', glow: '#10b981' };
        if (clampedValue >= 0.6) return { main: '#fbbf24', glow: '#f59e0b' };
        if (clampedValue >= 0.4) return { main: '#fb923c', glow: '#f97316' };
        return { main: '#f87171', glow: '#ef4444' };
    };
    const colors = getColor();

    // Generate tick marks around the semicircle
    const ticks = [];
    const numTicks = 20;
    for (let i = 0; i <= numTicks; i++) {
        const t = i / numTicks;
        const angle = startAngle - t * Math.PI;
        const isMajor = i % 5 === 0;
        const outerR = radius + (isMajor ? 16 : 10);
        const innerR = radius + 6;
        ticks.push({
            x1: cx + innerR * Math.cos(angle),
            y1: cy - innerR * Math.sin(angle),
            x2: cx + outerR * Math.cos(angle),
            y2: cy - outerR * Math.sin(angle),
            isMajor,
        });
    }

    // Needle position
    const needleAngle = startAngle - clampedValue * Math.PI;
    const needleLength = radius - 18;
    const needleX = cx + needleLength * Math.cos(needleAngle);
    const needleY = cy - needleLength * Math.sin(needleAngle);

    const svgHeight = size * 0.62;
    const gradientId = `gauge-grad-${Math.round(targetValue * 100)}`;
    const glowId = `gauge-glow-${Math.round(targetValue * 100)}`;

    return (
        <div style={{ position: 'relative', width: size, margin: '0 auto' }}>
            {/* SVG Gauge Arc */}
            <svg
                width={size}
                height={svgHeight}
                viewBox={`0 0 ${size} ${svgHeight}`}
                style={{ display: 'block' }}
            >
                <defs>
                    {/* Arc gradient */}
                    <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#ef4444" />
                        <stop offset="35%" stopColor="#f97316" />
                        <stop offset="60%" stopColor="#eab308" />
                        <stop offset="100%" stopColor="#22c55e" />
                    </linearGradient>
                    {/* Glow filter */}
                    <filter id={glowId} x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="4" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Tick marks */}
                {ticks.map((tick, i) => (
                    <line
                        key={i}
                        x1={tick.x1}
                        y1={tick.y1}
                        x2={tick.x2}
                        y2={tick.y2}
                        stroke={tick.isMajor ? 'rgba(6,182,212,0.3)' : 'rgba(6,182,212,0.1)'}
                        strokeWidth={tick.isMajor ? 2 : 1}
                        strokeLinecap="round"
                    />
                ))}

                {/* Background arc */}
                <path
                    d={bgPath}
                    fill="none"
                    stroke="rgba(6, 182, 212, 0.1)"
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                />

                {/* Value arc (with gradient + glow) */}
                {valuePath && (
                    <path
                        d={valuePath}
                        fill="none"
                        stroke={`url(#${gradientId})`}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        filter={`url(#${glowId})`}
                    />
                )}

                {/* Center dot */}
                <circle cx={cx} cy={cy} r={5} fill={colors.main} opacity={0.6} />

                {/* Needle */}
                <line
                    x1={cx}
                    y1={cy}
                    x2={needleX}
                    y2={needleY}
                    stroke={colors.main}
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    style={{
                        filter: `drop-shadow(0 0 4px ${colors.glow})`,
                    }}
                />

                {/* Needle end dot */}
                <circle
                    cx={needleX}
                    cy={needleY}
                    r={3.5}
                    fill={colors.main}
                    style={{
                        filter: `drop-shadow(0 0 6px ${colors.glow})`,
                    }}
                />
            </svg>

            {/* HTML text overlay — always crisp */}
            <div
                style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    textAlign: 'center',
                    paddingBottom: 4,
                }}
            >
                <div
                    style={{
                        fontSize: size * 0.19,
                        fontWeight: 800,
                        color: '#ffffff',
                        lineHeight: 1,
                        letterSpacing: '-0.02em',
                        textShadow: `0 0 12px ${colors.glow}60, 0 2px 4px rgba(0,0,0,0.5)`,
                    }}
                >
                    {percentage}%
                </div>
                {label && (
                    <div
                        style={{
                            fontSize: size * 0.065,
                            fontWeight: 600,
                            color: colors.main,
                            marginTop: 2,
                            letterSpacing: '0.05em',
                            textTransform: 'uppercase',
                            textShadow: `0 0 8px ${colors.glow}40`,
                        }}
                    >
                        {label}
                    </div>
                )}
            </div>
        </div>
    );
}
