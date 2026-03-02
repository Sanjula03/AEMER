/**
 * RadarChart — Pure SVG radar/spider chart for emotion probabilities
 * No external dependencies required
 */

interface RadarChartProps {
    data: Record<string, number>;
    size?: number;
}

const EMOTION_COLORS: Record<string, string> = {
    happy: '#facc15',
    sad: '#60a5fa',
    angry: '#f87171',
    neutral: '#9ca3af',
    fear: '#c084fc',
    surprise: '#f472b6',
    disgust: '#4ade80',
};

const EMOTION_EMOJIS: Record<string, string> = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    neutral: '😐',
    fear: '😨',
    surprise: '😲',
    disgust: '🤢',
};

export function RadarChart({ data, size = 280 }: RadarChartProps) {
    const entries = Object.entries(data).sort(([a], [b]) => a.localeCompare(b));
    const numAxes = entries.length;
    if (numAxes < 3) return null;

    const cx = size / 2;
    const cy = size / 2;
    const radius = size * 0.35;
    const angleStep = (2 * Math.PI) / numAxes;

    // Generate axis endpoints
    const axes = entries.map(([label], i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        return {
            label,
            x: cx + radius * Math.cos(angle),
            y: cy + radius * Math.sin(angle),
            labelX: cx + (radius + 28) * Math.cos(angle),
            labelY: cy + (radius + 28) * Math.sin(angle),
            angle,
        };
    });

    // Generate concentric rings (20%, 40%, 60%, 80%, 100%)
    const rings = [0.2, 0.4, 0.6, 0.8, 1.0];

    // Generate data polygon points
    const dataPoints = entries.map(([, value], i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        const r = radius * Math.min(value, 1);
        return {
            x: cx + r * Math.cos(angle),
            y: cy + r * Math.sin(angle),
        };
    });

    const dataPolygon = dataPoints.map(p => `${p.x},${p.y}`).join(' ');

    return (
        <div className="flex justify-center">
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="drop-shadow-lg">
                {/* Concentric ring backgrounds */}
                {rings.reverse().map((ring, i) => {
                    const ringPoints = axes.map(a => {
                        const r = radius * ring;
                        return `${cx + r * Math.cos(a.angle)},${cy + r * Math.sin(a.angle)}`;
                    }).join(' ');
                    return (
                        <polygon
                            key={i}
                            points={ringPoints}
                            fill={i % 2 === 0 ? 'rgba(180, 130, 60, 0.08)' : 'rgba(180, 130, 60, 0.04)'}
                            stroke="rgba(180, 130, 60, 0.2)"
                            strokeWidth="0.5"
                        />
                    );
                })}

                {/* Axis lines */}
                {axes.map((a, i) => (
                    <line
                        key={`axis-${i}`}
                        x1={cx}
                        y1={cy}
                        x2={a.x}
                        y2={a.y}
                        stroke="rgba(180, 130, 60, 0.25)"
                        strokeWidth="0.8"
                    />
                ))}

                {/* Data polygon fill */}
                <polygon
                    points={dataPolygon}
                    fill="rgba(20, 184, 166, 0.25)"
                    stroke="rgba(20, 184, 166, 0.8)"
                    strokeWidth="2"
                />

                {/* Data points */}
                {dataPoints.map((p, i) => (
                    <circle
                        key={`point-${i}`}
                        cx={p.x}
                        cy={p.y}
                        r="4"
                        fill={EMOTION_COLORS[entries[i][0].toLowerCase()] || '#14b8a6'}
                        stroke="white"
                        strokeWidth="1.5"
                    />
                ))}

                {/* Axis labels */}
                {axes.map((a, i) => (
                    <text
                        key={`label-${i}`}
                        x={a.labelX}
                        y={a.labelY}
                        textAnchor="middle"
                        dominantBaseline="central"
                        className="fill-amber-200/80"
                        fontSize="11"
                        fontWeight="500"
                    >
                        {EMOTION_EMOJIS[a.label.toLowerCase()] || '🎭'}{' '}
                        {a.label.charAt(0).toUpperCase() + a.label.slice(1)}
                    </text>
                ))}

                {/* Percentage labels on rings */}
                {[0.2, 0.4, 0.6, 0.8, 1.0].map((ring) => (
                    <text
                        key={`ring-label-${ring}`}
                        x={cx + 4}
                        y={cy - radius * ring + 2}
                        className="fill-amber-200/40"
                        fontSize="8"
                        textAnchor="start"
                    >
                        {Math.round(ring * 100)}%
                    </text>
                ))}
            </svg>
        </div>
    );
}
