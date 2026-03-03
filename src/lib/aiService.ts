/**
 * AI Service — Frontend API client for AEMER AI endpoints.
 * Communicates with the Python backend's /ai/* endpoints.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ── Types ────────────────────────────────────────────────────────────

export interface AIInsightsRequest {
    emotion_label: string;
    confidence_score: number;
    input_type?: string;
    all_probabilities?: Record<string, number>;
    detected_accent?: string | null;
    modalities_used?: string[];
    audio_result?: Record<string, any>;
    text_result?: Record<string, any>;
    video_result?: Record<string, any>;
}

export interface AIInsightsResponse {
    success: boolean;
    insights: string;
    source: 'gemini' | 'fallback' | 'error';
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface AIChatResponse {
    success: boolean;
    response: string;
    source: 'gemini' | 'fallback' | 'error';
}

export interface AIReportResponse {
    success: boolean;
    narrative: string;
    source: 'gemini' | 'fallback' | 'error';
}

// ── API Functions ────────────────────────────────────────────────────

/**
 * Fetch AI-generated insights for emotion analysis results.
 */
export async function fetchAIInsights(
    data: AIInsightsRequest
): Promise<AIInsightsResponse | null> {
    try {
        const res = await fetch(`${API_BASE}/ai/insights`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) return null;
        return await res.json();
    } catch (err) {
        console.warn('AI insights fetch failed:', err);
        return null;
    }
}

/**
 * Send a chat message to the AI chatbot.
 */
export async function sendChatMessage(
    message: string,
    emotionContext?: Record<string, any> | null,
    history?: ChatMessage[]
): Promise<AIChatResponse | null> {
    try {
        const res = await fetch(`${API_BASE}/ai/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                emotion_context: emotionContext || null,
                history: history || [],
            }),
        });
        if (!res.ok) return null;
        return await res.json();
    } catch (err) {
        console.warn('AI chat fetch failed:', err);
        return null;
    }
}

/**
 * Fetch an AI-generated narrative for reports.
 */
export async function fetchAIReport(
    results: Record<string, any>[]
): Promise<AIReportResponse | null> {
    try {
        const res = await fetch(`${API_BASE}/ai/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results }),
        });
        if (!res.ok) return null;
        return await res.json();
    } catch (err) {
        console.warn('AI report fetch failed:', err);
        return null;
    }
}
