/**
 * Local storage for analysis results
 * Since Supabase is unavailable, we store results locally in the browser
 */

export interface AnalysisResult {
    id: string;
    timestamp: string;
    input_type: 'audio' | 'video' | 'text';
    filename?: string;
    emotion_label: string;
    confidence_score: number;
    all_probabilities?: Record<string, number>;
}

const STORAGE_KEY = 'aemer_analysis_results';

/**
 * Get all stored analysis results
 */
export function getStoredResults(): AnalysisResult[] {
    try {
        const data = localStorage.getItem(STORAGE_KEY);
        return data ? JSON.parse(data) : [];
    } catch {
        return [];
    }
}

/**
 * Save a new analysis result
 */
export function saveResult(result: Omit<AnalysisResult, 'id' | 'timestamp'>): AnalysisResult {
    const newResult: AnalysisResult = {
        ...result,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
    };

    const results = getStoredResults();
    results.unshift(newResult); // Add to beginning (most recent first)

    // Keep only last 100 results
    const trimmed = results.slice(0, 100);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));

    return newResult;
}

/**
 * Clear all stored results
 */
export function clearResults(): void {
    localStorage.removeItem(STORAGE_KEY);
}

/**
 * Delete a specific result
 */
export function deleteResult(id: string): void {
    const results = getStoredResults();
    const filtered = results.filter(r => r.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
}
