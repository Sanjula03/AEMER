/**
 * Emotion Insights Engine
 * Generates human-readable mental health insights, recommendations,
 * and contextual information from emotion analysis results.
 */

// ─── Types ───────────────────────────────────────────────────────────
export type Valence = 'positive' | 'negative' | 'neutral';
export type ArousalLevel = 'high' | 'moderate' | 'low';
export type WellbeingLevel = 'stable' | 'mild_concern' | 'elevated_concern';
export type ConfidenceLabel = 'Very Low' | 'Low' | 'Moderate' | 'High' | 'Very High';

export interface WellbeingIndicator {
    level: WellbeingLevel;
    color: string;
    emoji: string;
    label: string;
    description: string;
}

export interface EmotionRank {
    emotion: string;
    probability: number;
}

export interface DominantSecondary {
    dominant: EmotionRank;
    secondary: EmotionRank;
    gap: number;
    interpretation: string;
}

export interface CopingStrategy {
    icon: string;
    title: string;
    description: string;
}

export interface HelplineInfo {
    name: string;
    number: string;
    region: string;
    available: string;
}

// ─── Emotional Valence ───────────────────────────────────────────────
const VALENCE_MAP: Record<string, Valence> = {
    happy: 'positive',
    surprise: 'positive',
    neutral: 'neutral',
    sad: 'negative',
    angry: 'negative',
    fear: 'negative',
    disgust: 'negative',
};

export function getEmotionalValence(emotion: string): Valence {
    return VALENCE_MAP[emotion.toLowerCase()] || 'neutral';
}

export function getValenceInfo(valence: Valence) {
    const info = {
        positive: { emoji: '☀️', label: 'Positive', color: 'text-green-400', bg: 'bg-green-900/30 border-green-700/50' },
        negative: { emoji: '🌧️', label: 'Negative', color: 'text-red-400', bg: 'bg-red-900/30 border-red-700/50' },
        neutral: { emoji: '⚖️', label: 'Neutral', color: 'text-gray-400', bg: 'bg-gray-700/30 border-gray-600/50' },
    };
    return info[valence];
}

// ─── Arousal Level ───────────────────────────────────────────────────
const AROUSAL_MAP: Record<string, ArousalLevel> = {
    angry: 'high',
    fear: 'high',
    surprise: 'high',
    happy: 'moderate',
    disgust: 'moderate',
    sad: 'low',
    neutral: 'low',
};

export function getArousalLevel(emotion: string): ArousalLevel {
    return AROUSAL_MAP[emotion.toLowerCase()] || 'low';
}

export function getArousalInfo(arousal: ArousalLevel) {
    const info = {
        high: { emoji: '⚡', label: 'High Arousal', color: 'text-orange-400', bg: 'bg-orange-900/30 border-orange-700/50', description: 'Heightened physiological activation' },
        moderate: { emoji: '🔄', label: 'Moderate Arousal', color: 'text-yellow-400', bg: 'bg-yellow-900/30 border-yellow-700/50', description: 'Balanced activation state' },
        low: { emoji: '🌊', label: 'Low Arousal', color: 'text-blue-400', bg: 'bg-blue-900/30 border-blue-700/50', description: 'Calm physiological state' },
    };
    return info[arousal];
}

// ─── Confidence Interpretation ───────────────────────────────────────
export function getConfidenceLabel(confidence: number): ConfidenceLabel {
    if (confidence >= 0.85) return 'Very High';
    if (confidence >= 0.70) return 'High';
    if (confidence >= 0.50) return 'Moderate';
    if (confidence >= 0.30) return 'Low';
    return 'Very Low';
}

export function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.80) return 'text-green-400';
    if (confidence >= 0.60) return 'text-yellow-400';
    return 'text-orange-400';
}

// ─── Well-being Indicator ────────────────────────────────────────────
export function getWellbeingIndicator(emotion: string, confidence: number): WellbeingIndicator {
    const valence = getEmotionalValence(emotion);
    const arousal = getArousalLevel(emotion);

    // Positive emotions → always stable
    if (valence === 'positive') {
        return {
            level: 'stable',
            color: 'text-green-400',
            emoji: '🟢',
            label: 'Stable',
            description: 'Your emotional state appears positive and balanced.',
        };
    }

    // Neutral → stable
    if (valence === 'neutral') {
        return {
            level: 'stable',
            color: 'text-green-400',
            emoji: '🟢',
            label: 'Stable',
            description: 'Your emotional state appears calm and neutral.',
        };
    }

    // Negative + high confidence + high arousal → elevated concern
    if (confidence >= 0.70 && arousal === 'high') {
        return {
            level: 'elevated_concern',
            color: 'text-red-400',
            emoji: '🔴',
            label: 'Elevated Concern',
            description: 'Strong indicators of emotional distress detected. Consider reaching out for support.',
        };
    }

    // Negative + moderate/high confidence → mild concern
    if (confidence >= 0.50) {
        return {
            level: 'mild_concern',
            color: 'text-yellow-400',
            emoji: '🟡',
            label: 'Mild Concern',
            description: 'Some signs of emotional difficulty detected. Self-care is recommended.',
        };
    }

    // Negative + low confidence → stable (not confident enough to raise concern)
    return {
        level: 'stable',
        color: 'text-green-400',
        emoji: '🟢',
        label: 'Stable',
        description: 'Emotional indicators are within normal range.',
    };
}

// ─── Mental State Summary ────────────────────────────────────────────
export function getMentalStateSummary(emotion: string, confidence: number, modalities?: string[]): string {
    const confidenceLabel = getConfidenceLabel(confidence).toLowerCase();
    const valence = getEmotionalValence(emotion);
    const emo = emotion.toLowerCase();

    const modalityText = modalities && modalities.length > 1
        ? `across ${modalities.length} analysis modalities (${modalities.join(', ')})`
        : 'from the provided input';

    const emotionDescriptions: Record<string, string> = {
        happy: `The analysis ${modalityText} indicates a predominantly happy emotional state with ${confidenceLabel} confidence. This suggests positive well-being and emotional contentment.`,
        sad: `The analysis ${modalityText} suggests a predominantly sad emotional state with ${confidenceLabel} confidence. This may indicate feelings of melancholy or emotional heaviness.`,
        angry: `The analysis ${modalityText} detects a predominantly angry emotional state with ${confidenceLabel} confidence. This suggests heightened frustration or irritation.`,
        neutral: `The analysis ${modalityText} shows a predominantly neutral emotional state with ${confidenceLabel} confidence. This indicates a calm, balanced emotional baseline.`,
        fear: `The analysis ${modalityText} identifies a predominantly fearful emotional state with ${confidenceLabel} confidence. This may indicate anxiety or apprehension.`,
        surprise: `The analysis ${modalityText} reveals a predominantly surprised emotional state with ${confidenceLabel} confidence. This suggests an unexpected reaction or heightened alertness.`,
        disgust: `The analysis ${modalityText} detects a predominantly disgusted emotional state with ${confidenceLabel} confidence. This may indicate strong aversion or discomfort.`,
    };

    let summary = emotionDescriptions[emo] ||
        `The analysis ${modalityText} detects a predominantly ${emo} emotional state with ${confidenceLabel} confidence.`;

    // Add valence note
    if (valence === 'negative' && confidence >= 0.6) {
        summary += ' If these feelings persist, consider speaking with a trusted person or professional.';
    }

    return summary;
}

// ─── Dominant & Secondary Emotion ────────────────────────────────────
export function getDominantAndSecondary(probabilities: Record<string, number>): DominantSecondary | null {
    const sorted = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a)
        .map(([emotion, probability]) => ({ emotion, probability }));

    if (sorted.length < 2) return null;

    const dominant = sorted[0];
    const secondary = sorted[1];
    const gap = dominant.probability - secondary.probability;

    let interpretation: string;
    if (gap > 0.4) {
        interpretation = `Clear dominance of ${dominant.emotion} — the signal is strong and unambiguous.`;
    } else if (gap > 0.2) {
        interpretation = `${dominant.emotion} is the primary emotion, with ${secondary.emotion} as a notable secondary signal.`;
    } else if (gap > 0.1) {
        interpretation = `Mixed emotional signals between ${dominant.emotion} and ${secondary.emotion} — both are significant.`;
    } else {
        interpretation = `Highly ambiguous — ${dominant.emotion} and ${secondary.emotion} are nearly equal, suggesting complex or mixed emotions.`;
    }

    return { dominant, secondary, gap, interpretation };
}

// ─── Coping Strategies ───────────────────────────────────────────────
export function getCopingStrategies(emotion: string): CopingStrategy[] {
    const strategies: Record<string, CopingStrategy[]> = {
        sad: [
            { icon: '🗣️', title: 'Talk to Someone', description: 'Reach out to a friend, family member, or counsellor. Sharing your feelings can lighten the emotional burden.' },
            { icon: '🚶', title: 'Gentle Physical Activity', description: 'Take a short walk or do light stretching. Movement helps release endorphins and improve mood.' },
            { icon: '📝', title: 'Journal Your Thoughts', description: 'Write down what you are feeling without judgment. This helps process emotions and gain clarity.' },
        ],
        angry: [
            { icon: '🧘', title: 'Deep Breathing Exercise', description: 'Try 4-7-8 breathing: inhale for 4 seconds, hold for 7, exhale for 8. Repeat 3-4 times.' },
            { icon: '⏸️', title: 'Take a Pause', description: 'Step away from the situation for 5-10 minutes. Give yourself space to cool down before responding.' },
            { icon: '💪', title: 'Physical Outlet', description: 'Channel the energy into exercise — a brisk walk, workout, or even squeezing a stress ball.' },
        ],
        fear: [
            { icon: '🫁', title: 'Grounding Technique', description: 'Try the 5-4-3-2-1 method: name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.' },
            { icon: '🤝', title: 'Seek Reassurance', description: 'Talk to someone you trust about what is making you anxious. Verbalizing fears reduces their power.' },
            { icon: '📋', title: 'Break It Down', description: 'If overwhelmed, break the feared task into tiny steps. Focus only on the very next small action.' },
        ],
        disgust: [
            { icon: '🌬️', title: 'Change Your Environment', description: 'Move to a different space or step outside for fresh air. A change of scenery can shift perspective.' },
            { icon: '🎵', title: 'Sensory Reset', description: 'Listen to calming music or a favourite song to redirect your emotional state.' },
            { icon: '💭', title: 'Reframe the Experience', description: 'Ask yourself: what can I learn from this? Reframing helps reduce the intensity of aversion.' },
        ],
        happy: [
            { icon: '🙏', title: 'Savour the Moment', description: 'Take a moment to fully appreciate your positive feelings. Mindful savouring strengthens happiness.' },
            { icon: '📱', title: 'Share Your Joy', description: 'Tell someone about what made you happy. Sharing positive experiences amplifies their effect.' },
            { icon: '📸', title: 'Capture It', description: 'Write it down or take a photo. Positive memories serve as future mood boosters.' },
        ],
        surprise: [
            { icon: '🧠', title: 'Process the Experience', description: 'Take a moment to understand what surprised you and how it made you feel.' },
            { icon: '✍️', title: 'Reflect and Record', description: 'Jot down what happened. Reflection turns surprising experiences into valuable insights.' },
            { icon: '🔄', title: 'Stay Adaptable', description: 'Embrace the unexpected. Flexibility in response to surprises builds emotional resilience.' },
        ],
        neutral: [
            { icon: '🎯', title: 'Set an Intention', description: 'Use this balanced state to set a positive intention for the rest of your day.' },
            { icon: '🧘', title: 'Practice Mindfulness', description: 'A neutral state is ideal for meditation or mindfulness practice.' },
            { icon: '📖', title: 'Engage Creatively', description: 'Read, draw, write, or explore — a calm mind is fertile ground for creativity.' },
        ],
    };

    return strategies[emotion.toLowerCase()] || strategies.neutral;
}

// ─── Contextual Tips ─────────────────────────────────────────────────
export function getContextualTips(emotion: string): string {
    const tips: Record<string, string> = {
        happy: '💡 Positive emotions broaden your thinking and build resilience. This is a great time for creative or social activities!',
        sad: '💡 Sadness is a natural emotion that signals a need for comfort and connection. Be gentle with yourself today.',
        angry: '💡 Anger often signals that a boundary has been crossed. Once calm, reflect on what triggered this feeling.',
        neutral: '💡 A neutral emotional state provides a stable foundation. Use this clarity for reflection or decision-making.',
        fear: '💡 Fear is your brain\'s protective mechanism. Assess if the threat is real or perceived, then respond accordingly.',
        surprise: '💡 Surprise heightens your attention and awareness. Use this alert state to process new information effectively.',
        disgust: '💡 Disgust helps you avoid harmful situations. Consider what values or boundaries this reaction highlights.',
    };

    return tips[emotion.toLowerCase()] || '💡 Take a moment to check in with yourself and your emotional needs.';
}

// ─── Helpline Information ────────────────────────────────────────────
export function getHelplineInfo(): HelplineInfo[] {
    return [
        { name: 'Samaritans', number: '116 123', region: 'UK & Ireland', available: '24/7, free' },
        { name: 'Crisis Text Line', number: 'Text HOME to 741741', region: 'US, UK, Canada', available: '24/7, free' },
        { name: 'National Suicide Prevention Lifeline', number: '988', region: 'USA', available: '24/7, free' },
        { name: 'Befrienders Worldwide', number: 'befrienders.org', region: 'International', available: 'Online directory' },
        { name: 'Sumithrayo', number: '011 2682535', region: 'Sri Lanka', available: '24/7, free' },
    ];
}

// ─── Modality Agreement ──────────────────────────────────────────────
export function getModalityAgreement(
    audioEmotion?: string,
    textEmotion?: string,
    videoEmotion?: string,
): { level: string; emoji: string; description: string } {
    const emotions = [audioEmotion, textEmotion, videoEmotion].filter(Boolean) as string[];
    if (emotions.length < 2) return { level: 'single', emoji: '➖', description: 'Single modality used' };

    const unique = new Set(emotions.map(e => e.toLowerCase()));

    if (unique.size === 1) {
        return { level: 'full', emoji: '✅', description: `All ${emotions.length} modalities agree` };
    }

    if (unique.size < emotions.length) {
        return { level: 'partial', emoji: '⚠️', description: `${emotions.length - unique.size + 1} of ${emotions.length} modalities agree` };
    }

    return { level: 'conflict', emoji: '❌', description: 'Conflicting signals across modalities' };
}

// ─── Disclaimer ──────────────────────────────────────────────────────
export const DISCLAIMER_TEXT = 'This analysis is generated by an AI system for informational and educational purposes only. It does not constitute a medical or psychological diagnosis, professional advice, or treatment recommendation. If you are experiencing emotional distress or mental health difficulties, please consult a qualified healthcare professional.';
