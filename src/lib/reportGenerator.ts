/**
 * reportGenerator.ts — Generates self-contained styled HTML reports
 * for AEMER emotion analysis results.
 */
import type { AnalysisResult } from './storage';
import {
    getMentalStateSummary,
    getEmotionalValence,
    getArousalLevel,
    getWellbeingIndicator,
    getConfidenceLabel,
    getCopingStrategies,
    getContextualTips,
    getHelplineInfo,
    getDominantAndSecondary,
    DISCLAIMER_TEXT,
} from './emotionInsights';

const EMOTION_COLORS: Record<string, string> = {
    happy: '#facc15',
    sad: '#60a5fa',
    angry: '#f87171',
    neutral: '#9ca3af',
    fear: '#a78bfa',
    surprise: '#f472b6',
    disgust: '#4ade80',
};

const EMOTION_EMOJIS: Record<string, string> = {
    happy: '😊', sad: '😢', angry: '😠', neutral: '😐',
    fear: '😨', surprise: '😲', disgust: '🤢',
};

function getColor(emotion: string): string {
    return EMOTION_COLORS[emotion.toLowerCase()] || '#14b8a6';
}

function getEmoji(emotion: string): string {
    return EMOTION_EMOJIS[emotion.toLowerCase()] || '🎭';
}

/** Generate a styled HTML report for a single analysis */
export function generateSingleReportHTML(result: AnalysisResult): string {
    const wellbeing = getWellbeingIndicator(result.emotion_label, result.confidence_score);
    const summary = getMentalStateSummary(result.emotion_label, result.confidence_score, result.modalities_used);
    const valence = getEmotionalValence(result.emotion_label);
    const arousal = getArousalLevel(result.emotion_label);
    const confidenceLabel = getConfidenceLabel(result.confidence_score);
    const strategies = getCopingStrategies(result.emotion_label);
    const tip = getContextualTips(result.emotion_label);
    const helplines = getHelplineInfo();
    const dominant = result.all_probabilities ? getDominantAndSecondary(result.all_probabilities) : null;
    const color = getColor(result.emotion_label);
    const emoji = getEmoji(result.emotion_label);
    const pct = (result.confidence_score * 100).toFixed(1);
    const dateStr = new Date(result.timestamp).toLocaleString();

    // Generate probability bars
    const probBars = result.all_probabilities
        ? Object.entries(result.all_probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([emo, prob]) => `
                <div style="display:flex;align-items:center;gap:10px;margin:4px 0">
                    <span style="width:22px;text-align:center">${getEmoji(emo)}</span>
                    <span style="width:80px;font-weight:500;text-transform:capitalize;color:#e5e7eb">${emo}</span>
                    <div style="flex:1;background:rgba(255,255,255,0.08);border-radius:8px;height:10px;overflow:hidden">
                        <div style="width:${prob * 100}%;height:100%;background:${getColor(emo)};border-radius:8px"></div>
                    </div>
                    <span style="width:50px;text-align:right;font-size:13px;color:#9ca3af">${(prob * 100).toFixed(1)}%</span>
                </div>
            `).join('')
        : '';

    // Coping strategies HTML
    const strategiesHTML = strategies.map(s => `
        <div style="display:flex;align-items:flex-start;gap:12px;background:rgba(255,255,255,0.04);border-radius:10px;padding:14px;border:1px solid rgba(255,255,255,0.06)">
            <span style="font-size:24px;flex-shrink:0">${s.icon}</span>
            <div>
                <div style="font-weight:600;color:#e5e7eb;font-size:14px">${s.title}</div>
                <div style="font-size:12px;color:#9ca3af;margin-top:2px;line-height:1.5">${s.description}</div>
            </div>
        </div>
    `).join('');

    // Helplines HTML
    const helplinesHTML = helplines.map(h => `
        <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:10px 14px;border:1px solid rgba(255,255,255,0.06)">
            <div>
                <div style="font-weight:500;color:#e5e7eb;font-size:13px">${h.name}</div>
                <div style="font-size:11px;color:#6b7280">${h.region} • ${h.available}</div>
            </div>
            <div style="font-family:monospace;color:#06b6d4;font-size:13px;font-weight:600">${h.number}</div>
        </div>
    `).join('');

    // Modality breakdown
    let modalityHTML = '';
    if (result.modalities_used && result.modalities_used.length > 1) {
        const modCards = [];
        if (result.audio_result) {
            modCards.push(`
                <div style="flex:1;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:20px">🎤</div>
                    <div style="font-weight:600;color:#06b6d4;font-size:13px;margin-top:4px">Audio</div>
                    <div style="font-size:18px;font-weight:700;color:#e5e7eb;text-transform:capitalize;margin-top:6px">${result.audio_result.emotion_label}</div>
                    <div style="font-size:11px;color:#9ca3af;margin-top:4px">Confidence: ${(result.audio_result.confidence_score * 100).toFixed(0)}%</div>
                </div>
            `);
        }
        if (result.text_result) {
            modCards.push(`
                <div style="flex:1;background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2);border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:20px">📝</div>
                    <div style="font-weight:600;color:#a855f7;font-size:13px;margin-top:4px">Text</div>
                    <div style="font-size:18px;font-weight:700;color:#e5e7eb;text-transform:capitalize;margin-top:6px">${result.text_result.emotion_label}</div>
                    <div style="font-size:11px;color:#9ca3af;margin-top:4px">Confidence: ${(result.text_result.confidence_score * 100).toFixed(0)}%</div>
                </div>
            `);
        }
        if (result.video_result) {
            modCards.push(`
                <div style="flex:1;background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.2);border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:20px">📹</div>
                    <div style="font-weight:600;color:#60a5fa;font-size:13px;margin-top:4px">Video</div>
                    <div style="font-size:18px;font-weight:700;color:#e5e7eb;text-transform:capitalize;margin-top:6px">${result.video_result.emotion_label}</div>
                    <div style="font-size:11px;color:#9ca3af;margin-top:4px">Confidence: ${(result.video_result.confidence_score * 100).toFixed(0)}%</div>
                </div>
            `);
        }
        modalityHTML = `
            <div style="margin-top:24px">
                <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin:0 0 12px 0">🔀 Modality Breakdown</h3>
                <div style="display:flex;gap:12px">${modCards.join('')}</div>
            </div>
        `;
    }

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEMER Emotion Report — ${result.emotion_label}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #050505 0%, #0a0a0a 50%, #050505 100%);
            color: #d4d4d8;
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 700px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            padding: 32px 24px;
            background: linear-gradient(135deg, ${color}22, ${color}08);
            border: 1px solid ${color}33;
            border-radius: 16px;
            margin-bottom: 24px;
        }
        .section {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }
        @media print {
            body { background: #050505; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div style="text-align:center;margin-bottom:24px">
            <div style="font-size:28px;font-weight:800;color:#ffffff;letter-spacing:-0.02em">AEMER</div>
            <div style="font-size:12px;color:#525252;margin-top:2px">Accent-Aware Emotion Recognition</div>
        </div>

        <!-- Emotion Result -->
        <div class="header">
            <div style="font-size:60px;margin-bottom:8px">${emoji}</div>
            <div style="font-size:32px;font-weight:800;color:${color};text-transform:capitalize">${result.emotion_label}</div>
            <div style="font-size:14px;color:#9ca3af;margin-top:4px">Detected Emotion</div>
            <div style="display:inline-flex;align-items:center;gap:6px;margin-top:12px;background:rgba(255,255,255,0.08);padding:6px 14px;border-radius:20px">
                <span>${wellbeing.emoji}</span>
                <span style="font-weight:600;font-size:13px;color:#e5e7eb">${wellbeing.label}</span>
            </div>
            <div style="margin-top:16px">
                <span style="font-size:36px;font-weight:800;color:#fff">${pct}%</span>
                <span style="font-size:13px;color:#9ca3af;margin-left:8px">${confidenceLabel}</span>
            </div>
            ${result.detected_accent ? `<div style="margin-top:8px;font-size:12px;color:#9ca3af">Detected Accent: <strong style="color:#e5e7eb">${result.detected_accent}</strong></div>` : ''}
        </div>

        <!-- Mental State Summary -->
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:10px">🧠 Mental State Summary</h3>
            <p style="font-size:14px;color:#a1a1aa;line-height:1.7">${summary}</p>
            <div style="margin-top:12px;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.15);border-radius:8px;padding:10px 14px">
                <p style="font-size:12px;color:#06b6d4;line-height:1.5">💡 ${tip}</p>
            </div>
        </div>

        <!-- Emotional Profile -->
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:12px">💠 Emotional Profile</h3>
            <div style="display:flex;gap:12px;flex-wrap:wrap">
                <div style="flex:1;min-width:140px;background:rgba(255,255,255,0.04);border-radius:10px;padding:12px;text-align:center;border:1px solid rgba(255,255,255,0.06)">
                    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em">Valence</div>
                    <div style="font-size:16px;font-weight:700;color:#e5e7eb;text-transform:capitalize;margin-top:4px">${valence}</div>
                </div>
                <div style="flex:1;min-width:140px;background:rgba(255,255,255,0.04);border-radius:10px;padding:12px;text-align:center;border:1px solid rgba(255,255,255,0.06)">
                    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em">Arousal</div>
                    <div style="font-size:16px;font-weight:700;color:#e5e7eb;text-transform:capitalize;margin-top:4px">${arousal}</div>
                </div>
                <div style="flex:1;min-width:140px;background:rgba(255,255,255,0.04);border-radius:10px;padding:12px;text-align:center;border:1px solid rgba(255,255,255,0.06)">
                    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em">Well-being</div>
                    <div style="font-size:16px;font-weight:700;color:#e5e7eb;margin-top:4px">${wellbeing.emoji} ${wellbeing.label}</div>
                </div>
            </div>
        </div>

        ${dominant ? `
        <!-- Dominant vs Secondary -->
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:12px">🎯 Dominant vs. Secondary Emotion</h3>
            <div style="display:flex;gap:12px;margin-bottom:10px">
                <div style="flex:1;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.15);border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:28px">${getEmoji(dominant.dominant.emotion)}</div>
                    <div style="font-weight:600;color:#06b6d4;text-transform:capitalize;margin-top:4px">${dominant.dominant.emotion}</div>
                    <div style="font-size:13px;color:#9ca3af">${(dominant.dominant.probability * 100).toFixed(1)}%</div>
                </div>
                <div style="flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:28px">${getEmoji(dominant.secondary.emotion)}</div>
                    <div style="font-weight:600;color:#9ca3af;text-transform:capitalize;margin-top:4px">${dominant.secondary.emotion}</div>
                    <div style="font-size:13px;color:#9ca3af">${(dominant.secondary.probability * 100).toFixed(1)}%</div>
                </div>
            </div>
            <p style="font-size:12px;color:#6b7280;font-style:italic">${dominant.interpretation}</p>
        </div>
        ` : ''}

        <!-- Emotion Probabilities -->
        ${probBars ? `
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:12px">📊 Emotion Distribution</h3>
            ${probBars}
        </div>
        ` : ''}

        ${modalityHTML ? `<div class="section">${modalityHTML}</div>` : ''}

        <!-- Coping Strategies -->
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:12px">💚 Recommendations</h3>
            <div style="display:flex;flex-direction:column;gap:10px">
                ${strategiesHTML}
            </div>
        </div>

        <!-- Helplines -->
        ${wellbeing.level !== 'stable' ? `
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:12px">📞 Mental Health Helplines</h3>
            <div style="display:flex;flex-direction:column;gap:8px">
                ${helplinesHTML}
            </div>
        </div>
        ` : ''}

        <!-- Metadata -->
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:12px">📋 Analysis Details</h3>
            <table style="width:100%;font-size:13px">
                <tr><td style="color:#6b7280;padding:4px 0">Report ID</td><td style="color:#e5e7eb;text-align:right;font-family:monospace;font-size:11px">${result.id}</td></tr>
                <tr><td style="color:#6b7280;padding:4px 0">Date & Time</td><td style="color:#e5e7eb;text-align:right">${dateStr}</td></tr>
                <tr><td style="color:#6b7280;padding:4px 0">Input Type</td><td style="color:#e5e7eb;text-align:right;text-transform:capitalize">${result.input_type}</td></tr>
                ${result.filename ? `<tr><td style="color:#6b7280;padding:4px 0">File</td><td style="color:#e5e7eb;text-align:right">${result.filename}</td></tr>` : ''}
                ${result.fusion_method ? `<tr><td style="color:#6b7280;padding:4px 0">Fusion Method</td><td style="color:#e5e7eb;text-align:right;text-transform:capitalize">${result.fusion_method.replace('_', ' ')}</td></tr>` : ''}
            </table>
        </div>

        <!-- Disclaimer -->
        <div style="text-align:center;padding:20px;margin-top:8px">
            <p style="font-size:11px;color:#404040;line-height:1.6;max-width:500px;margin:0 auto">⚠️ ${DISCLAIMER_TEXT}</p>
        </div>

        <!-- Footer -->
        <div style="text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,0.05);margin-top:8px">
            <div style="font-size:13px;font-weight:700;color:#525252">AEMER</div>
            <div style="font-size:10px;color:#404040;margin-top:2px">Accent-Aware Emotion Recognition • Generated ${dateStr}</div>
        </div>
    </div>
</body>
</html>`;
}

/** Generate a styled HTML summary report for multiple analyses */
export function generateSummaryReportHTML(
    results: AnalysisResult[],
    emotionStats: Array<{ emotion: string; count: number; avgConfidence: number; emoji: string }>,
): string {
    const dateStr = new Date().toLocaleString();
    const totalAnalyses = results.length;

    // Most common emotion
    const topEmotion = emotionStats.length > 0 ? emotionStats[0] : null;

    // Average confidence across all
    const avgConf = results.length > 0
        ? results.reduce((sum, r) => sum + r.confidence_score, 0) / results.length
        : 0;

    // Emotion stats bars
    const statsBars = emotionStats.map(stat => `
        <div style="display:flex;align-items:center;gap:10px;margin:6px 0">
            <span style="width:22px;text-align:center;font-size:18px">${stat.emoji}</span>
            <span style="width:80px;font-weight:500;text-transform:capitalize;color:#e5e7eb;font-size:14px">${stat.emotion}</span>
            <div style="flex:1;background:rgba(255,255,255,0.08);border-radius:8px;height:12px;overflow:hidden">
                <div style="width:${totalAnalyses > 0 ? (stat.count / totalAnalyses) * 100 : 0}%;height:100%;background:${getColor(stat.emotion)};border-radius:8px"></div>
            </div>
            <span style="width:35px;text-align:right;font-weight:600;color:#e5e7eb">${stat.count}</span>
            <span style="width:50px;text-align:right;font-size:12px;color:#9ca3af">${(stat.avgConfidence * 100).toFixed(0)}% avg</span>
        </div>
    `).join('');

    // Recent analyses table
    const recentRows = results.slice(0, 15).map(r => `
        <tr>
            <td style="padding:8px 10px;color:#e5e7eb;font-size:13px">${getEmoji(r.emotion_label)} <span style="text-transform:capitalize">${r.emotion_label}</span></td>
            <td style="padding:8px 10px;color:#9ca3af;font-size:13px;text-align:center">${(r.confidence_score * 100).toFixed(0)}%</td>
            <td style="padding:8px 10px;color:#9ca3af;font-size:13px;text-align:center;text-transform:capitalize">${r.input_type}</td>
            <td style="padding:8px 10px;color:#6b7280;font-size:12px;text-align:right">${new Date(r.timestamp).toLocaleDateString()}</td>
        </tr>
    `).join('');

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEMER Analytics Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #050505 0%, #0a0a0a 50%, #050505 100%);
            color: #d4d4d8;
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container { max-width: 700px; margin: 0 auto; }
        .section {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }
        table { width: 100%; border-collapse: collapse; }
        tr:not(:last-child) { border-bottom: 1px solid rgba(255,255,255,0.04); }
        @media print {
            body { background: #050505; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div style="text-align:center;margin-bottom:28px">
            <div style="font-size:28px;font-weight:800;color:#ffffff;letter-spacing:-0.02em">AEMER</div>
            <div style="font-size:12px;color:#525252;margin-top:2px">Accent-Aware Emotion Recognition</div>
            <div style="font-size:20px;font-weight:700;color:#e5e7eb;margin-top:12px">Analytics Report</div>
            <div style="font-size:12px;color:#6b7280;margin-top:4px">Generated on ${dateStr}</div>
        </div>

        <!-- Overview Cards -->
        <div style="display:flex;gap:12px;margin-bottom:16px">
            <div style="flex:1;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.15);border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:28px;font-weight:800;color:#06b6d4">${totalAnalyses}</div>
                <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Total Analyses</div>
            </div>
            ${topEmotion ? `
            <div style="flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:28px">${topEmotion.emoji}</div>
                <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Most Common</div>
            </div>
            ` : ''}
            <div style="flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:28px;font-weight:800;color:#fbbf24">${(avgConf * 100).toFixed(0)}%</div>
                <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Avg Confidence</div>
            </div>
        </div>

        <!-- Emotion Breakdown -->
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:14px">📊 Emotion Breakdown</h3>
            ${statsBars}
        </div>

        <!-- Recent Analyses -->
        <div class="section">
            <h3 style="font-size:16px;font-weight:600;color:#e5e7eb;margin-bottom:14px">📋 Recent Analyses${results.length > 15 ? ' (Last 15)' : ''}</h3>
            <table>
                <thead>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                        <th style="padding:8px 10px;text-align:left;font-size:11px;color:#6b7280;text-transform:uppercase;font-weight:600">Emotion</th>
                        <th style="padding:8px 10px;text-align:center;font-size:11px;color:#6b7280;text-transform:uppercase;font-weight:600">Confidence</th>
                        <th style="padding:8px 10px;text-align:center;font-size:11px;color:#6b7280;text-transform:uppercase;font-weight:600">Input</th>
                        <th style="padding:8px 10px;text-align:right;font-size:11px;color:#6b7280;text-transform:uppercase;font-weight:600">Date</th>
                    </tr>
                </thead>
                <tbody>
                    ${recentRows}
                </tbody>
            </table>
        </div>

        <!-- Disclaimer -->
        <div style="text-align:center;padding:20px;margin-top:8px">
            <p style="font-size:11px;color:#404040;line-height:1.6;max-width:500px;margin:0 auto">⚠️ ${DISCLAIMER_TEXT}</p>
        </div>

        <!-- Footer -->
        <div style="text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,0.05);margin-top:8px">
            <div style="font-size:13px;font-weight:700;color:#525252">AEMER</div>
            <div style="font-size:10px;color:#404040;margin-top:2px">Accent-Aware Emotion Recognition • Generated ${dateStr}</div>
        </div>
    </div>
</body>
</html>`;
}

/** Download HTML report as a PDF file using html2pdf.js */
export async function downloadPDF(html: string, filename: string) {
    // Create a hidden container to render the HTML
    const container = document.createElement('div');
    container.style.position = 'fixed';
    container.style.left = '-9999px';
    container.style.top = '0';
    container.style.width = '700px';
    container.innerHTML = html;

    // Extract just the body content if it's a full HTML document
    const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/);
    if (bodyMatch) {
        container.innerHTML = bodyMatch[1];
    }

    // Apply the dark background styles inline
    container.style.background = '#050505';
    container.style.color = '#d4d4d8';
    container.style.fontFamily = "'Segoe UI', system-ui, -apple-system, sans-serif";
    container.style.padding = '40px 20px';

    document.body.appendChild(container);

    // Dynamically import html2pdf.js
    const html2pdf = (await import('html2pdf.js')).default;

    const pdfFilename = filename.replace(/\.html$/i, '.pdf');

    await html2pdf()
        .set({
            margin: 0,
            filename: pdfFilename,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: {
                scale: 2,
                useCORS: true,
                backgroundColor: '#050505',
                logging: false,
            },
            jsPDF: {
                unit: 'mm',
                format: 'a4',
                orientation: 'portrait',
            },
            pagebreak: { mode: ['avoid-all', 'css', 'legacy'] },
        } as any)
        .from(container)
        .save();

    // Clean up
    document.body.removeChild(container);
}

/** @deprecated Use downloadPDF instead */
export function downloadHTML(html: string, filename: string) {
    downloadPDF(html, filename);
}
