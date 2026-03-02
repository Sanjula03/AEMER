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

    // SVG dimensions — extra height for text below the arc
    const svgWidth = size;
    const svgHeight = size * 0.78;
    const cx = svgWidth / 2;
    const cy = svgHeight * 0.55;
    const radius = size * 0.36;
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
            <svg width={svgWidth} height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`}>
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

                {/* Dark backing circle behind text for contrast */}
                <circle
                    cx={cx}
                    cy={cy - 2}
                    r={radius * 0.55}
                    fill="rgba(28, 25, 23, 0.6)"
                />

                {/* Percentage text */}
                <text
                    x={cx}
                    y={cy - 2}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="#ffffff"
                    fontSize={size * 0.18}
                    fontWeight="800"
                    fontFamily="system-ui, -apple-system, sans-serif"
                >
                    {percentage}%
                </text>

                {/* Label below the arc */}
                {label && (
                    <text
                        x={cx}
                        y={cy + radius * 0.42}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fill="#e5e5e5"
                        fontSize={size * 0.07}
                        fontWeight="600"
                        fontFamily="system-ui, -apple-system, sans-serif"
                    >
                        {label}
                    </text>
                )}
            </svg>
        </div>
    );
}
