/**
 * EmotionGauge — Semi-circular confidence gauge built with HTML/CSS
 * Uses conic-gradient for the arc — no SVG text rendering issues
 */

interface EmotionGaugeProps {
    value: number; // 0 to 1
    label?: string;
    size?: number;
}

export function EmotionGauge({ value, label, size = 200 }: EmotionGaugeProps) {
    const clampedValue = Math.min(Math.max(value, 0), 1);
    const percentage = Math.round(clampedValue * 100);

    // Color based on value
    const getColor = () => {
        if (clampedValue >= 0.8) return '#4ade80';
        if (clampedValue >= 0.6) return '#facc15';
        if (clampedValue >= 0.4) return '#fb923c';
        return '#f87171';
    };

    const color = getColor();

    // Convert value (0-1) to degrees for the semicircle (0° to 180°)
    const degrees = clampedValue * 180;

    const gaugeSize = size;
    const halfSize = gaugeSize / 2;

    return (
        <div className="flex flex-col items-center" style={{ paddingTop: 8 }}>
            {/* Gauge wrapper — clips to semicircle */}
            <div
                style={{
                    width: gaugeSize,
                    height: halfSize,
                    position: 'relative',
                    overflow: 'hidden',
                }}
            >
                {/* Background ring (full semicircle, dark) */}
                <div
                    style={{
                        width: gaugeSize,
                        height: gaugeSize,
                        borderRadius: '50%',
                        background: `conic-gradient(
                            from 180deg,
                            rgba(120, 90, 40, 0.2) 0deg,
                            rgba(120, 90, 40, 0.2) 180deg,
                            transparent 180deg
                        )`,
                        position: 'absolute',
                        top: 0,
                        left: 0,
                    }}
                />

                {/* Value ring */}
                <div
                    style={{
                        width: gaugeSize,
                        height: gaugeSize,
                        borderRadius: '50%',
                        background: `conic-gradient(
                            from 180deg,
                            ${color} 0deg,
                            ${color} ${degrees}deg,
                            transparent ${degrees}deg,
                            transparent 180deg
                        )`,
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        filter: `drop-shadow(0 0 8px ${color}50)`,
                        transition: 'all 0.5s ease-in-out',
                    }}
                />

                {/* Inner cutout (donut hole) */}
                <div
                    style={{
                        width: gaugeSize - 28,
                        height: gaugeSize - 28,
                        borderRadius: '50%',
                        background: 'rgb(41, 37, 36)', // stone-800 equivalent
                        position: 'absolute',
                        top: 14,
                        left: 14,
                    }}
                />
            </div>

            {/* Text overlay — positioned over the gauge */}
            <div
                style={{
                    marginTop: -(halfSize * 0.55),
                    textAlign: 'center',
                    position: 'relative',
                    zIndex: 1,
                }}
            >
                <div
                    style={{
                        fontSize: size * 0.2,
                        fontWeight: 800,
                        color: '#ffffff',
                        lineHeight: 1.1,
                        letterSpacing: '-0.02em',
                        textShadow: '0 2px 8px rgba(0,0,0,0.5)',
                    }}
                >
                    {percentage}%
                </div>
                {label && (
                    <div
                        style={{
                            fontSize: size * 0.075,
                            fontWeight: 600,
                            color: '#d4d4d4',
                            marginTop: 2,
                            textShadow: '0 1px 4px rgba(0,0,0,0.4)',
                        }}
                    >
                        {label}
                    </div>
                )}
            </div>
        </div>
    );
}
