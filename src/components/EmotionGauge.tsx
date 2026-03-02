/**
 * EmotionGauge — Semi-circular confidence gauge with color gradient
 */

interface EmotionGaugeProps {
    value: number; // 0 to 1
    label?: string;
    size?: number;
}

export function EmotionGauge({ value, label, size = 200 }: EmotionGaugeProps) {
    const clampedValue = Math.min(Math.max(value, 0), 1);
    const percentage = Math.round(clampedValue * 100);

    // SVG arc calculations
    const cx = size / 2;
    const cy = size * 0.6;
    const radius = size * 0.4;
    const strokeWidth = 14;

    // Arc from 180° to 0° (left to right semicircle)
    const startAngle = Math.PI;
    const endAngle = 0;
    const sweepAngle = startAngle - (startAngle - endAngle) * clampedValue;

    const bgArcStart = {
        x: cx + radius * Math.cos(startAngle),
        y: cy - radius * Math.sin(startAngle),
    };
    const bgArcEnd = {
        x: cx + radius * Math.cos(endAngle),
        y: cy - radius * Math.sin(endAngle),
    };

    const valueArcEnd = {
        x: cx + radius * Math.cos(sweepAngle),
        y: cy - radius * Math.sin(sweepAngle),
    };

    const largeArcFlag = clampedValue > 0.5 ? 1 : 0;

    const bgPath = `M ${bgArcStart.x} ${bgArcStart.y} A ${radius} ${radius} 0 1 1 ${bgArcEnd.x} ${bgArcEnd.y}`;
    const valuePath = `M ${bgArcStart.x} ${bgArcStart.y} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${valueArcEnd.x} ${valueArcEnd.y}`;

    // Determine color based on value
    const getColor = () => {
        if (clampedValue >= 0.8) return '#4ade80'; // Green
        if (clampedValue >= 0.6) return '#facc15'; // Yellow
        if (clampedValue >= 0.4) return '#fb923c'; // Orange
        return '#f87171'; // Red
    };

    const color = getColor();

    return (
        <div className="flex flex-col items-center">
            <svg width={size} height={size * 0.7} viewBox={`0 0 ${size} ${size * 0.7}`}>
                {/* Gradient definition */}
                <defs>
                    <linearGradient id={`gauge-gradient-${percentage}`} x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#f87171" />
                        <stop offset="33%" stopColor="#fb923c" />
                        <stop offset="66%" stopColor="#facc15" />
                        <stop offset="100%" stopColor="#4ade80" />
                    </linearGradient>
                </defs>

                {/* Background arc */}
                <path
                    d={bgPath}
                    fill="none"
                    stroke="rgba(120, 90, 40, 0.2)"
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                />

                {/* Value arc */}
                {clampedValue > 0.01 && (
                    <path
                        d={valuePath}
                        fill="none"
                        stroke={color}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        style={{
                            filter: `drop-shadow(0 0 6px ${color}40)`,
                            transition: 'all 0.5s ease-in-out',
                        }}
                    />
                )}

                <text
                    x={cx}
                    y={cy - 8}
                    textAnchor="middle"
                    fill="#fef3c7"
                    fontSize={size * 0.16}
                    fontWeight="bold"
                    fontFamily="system-ui, sans-serif"
                >
                    {percentage}%
                </text>

                {/* Label */}
                {label && (
                    <text
                        x={cx}
                        y={cy + size * 0.1}
                        textAnchor="middle"
                        fill="rgba(253, 230, 138, 0.6)"
                        fontSize={size * 0.065}
                        fontFamily="system-ui, sans-serif"
                    >
                        {label}
                    </text>
                )}
            </svg>
        </div>
    );
}
