import { useState, useRef, useEffect, useCallback } from 'react';
import { MessageCircle, X, Send, Loader2, Sparkles, Bot, User, Trash2 } from 'lucide-react';
import { sendChatMessage, ChatMessage } from '../lib/aiService';
import { getStoredResults } from '../lib/storage';

/**
 * Floating AI Chatbot Widget
 * Appears as a small button in the bottom-right corner,
 * expands to a full chat panel when clicked.
 */
export function AIChatbot() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            role: 'assistant',
            content: "Hi! I'm AEMER's AI assistant 🤖\n\nI can help you understand your emotion analysis results, suggest coping strategies, or just chat about how you're feeling.\n\nHow can I help you today?"
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Focus input when chat opens
    useEffect(() => {
        if (isOpen) {
            setTimeout(() => inputRef.current?.focus(), 300);
        }
    }, [isOpen]);

    const getLatestEmotionContext = useCallback(() => {
        const results = getStoredResults();
        if (results.length === 0) return null;
        const latest = results[0];
        return {
            emotion_label: latest.emotion_label,
            confidence_score: latest.confidence_score,
            input_type: latest.input_type,
            all_probabilities: latest.all_probabilities,
            detected_accent: latest.detected_accent,
            modalities_used: latest.modalities_used,
            audio_result: latest.audio_result,
            text_result: latest.text_result,
            video_result: latest.video_result,
        };
    }, []);

    const handleSend = useCallback(async () => {
        const trimmed = input.trim();
        if (!trimmed || isLoading) return;

        const userMessage: ChatMessage = { role: 'user', content: trimmed };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const context = getLatestEmotionContext();
            // Send only user/assistant messages as history
            const history = messages
                .filter(m => m.role === 'user' || m.role === 'assistant')
                .slice(-10);

            const response = await sendChatMessage(trimmed, context, history);

            const assistantMessage: ChatMessage = {
                role: 'assistant',
                content: response?.response || "I'm having trouble connecting. Please try again in a moment."
            };
            setMessages(prev => [...prev, assistantMessage]);
        } catch {
            setMessages(prev => [
                ...prev,
                { role: 'assistant', content: "Sorry, I couldn't process that. Please try again." }
            ]);
        } finally {
            setIsLoading(false);
        }
    }, [input, isLoading, messages, getLatestEmotionContext]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleClear = () => {
        setMessages([{
            role: 'assistant',
            content: "Chat cleared! How can I help you? 🤖"
        }]);
    };

    // ── Render ────────────────────────────────────────────────────────

    // Floating button (collapsed state)
    if (!isOpen) {
        return (
            <button
                id="ai-chatbot-toggle"
                aria-label="Toggle AI Chatbot"
                onClick={() => setIsOpen(true)}
                className="fixed z-[9999] flex items-center justify-center transition-all duration-300 hover:scale-110"
                style={{
                    bottom: '90px',
                    right: '20px',
                    width: '56px',
                    height: '56px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #06b6d4, #0e7490)',
                    boxShadow: '0 4px 20px rgba(6, 182, 212, 0.4), 0 0 40px rgba(6, 182, 212, 0.15)',
                    border: '2px solid rgba(6, 182, 212, 0.3)',
                }}
            >
                <MessageCircle className="w-6 h-6 text-white" />
                {/* Pulse ring */}
                <div
                    className="absolute inset-0 rounded-full animate-ping"
                    style={{
                        background: 'rgba(6, 182, 212, 0.2)',
                        animationDuration: '3s',
                    }}
                />
            </button>
        );
    }

    // Expanded chat panel
    return (
        <div
            id="ai-chatbot-panel"
            className="fixed z-[9999] flex flex-col"
            style={{
                bottom: '90px',
                right: '20px',
                width: '380px',
                maxWidth: 'calc(100vw - 40px)',
                height: '520px',
                maxHeight: 'calc(100vh - 120px)',
                borderRadius: '20px',
                background: 'rgba(10, 10, 10, 0.95)',
                backdropFilter: 'blur(24px)',
                border: '1px solid rgba(6, 182, 212, 0.15)',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(6, 182, 212, 0.1)',
                animation: 'chatSlideUp 0.3s ease-out',
            }}
        >
            {/* Header */}
            <div
                className="flex items-center justify-between px-4 py-3"
                style={{
                    borderBottom: '1px solid rgba(6, 182, 212, 0.1)',
                    borderRadius: '20px 20px 0 0',
                    background: 'rgba(6, 182, 212, 0.05)',
                }}
            >
                <div className="flex items-center gap-2.5">
                    <div
                        className="p-1.5 rounded-lg"
                        style={{
                            background: 'linear-gradient(135deg, #06b6d4, #0e7490)',
                        }}
                    >
                        <Sparkles className="w-4 h-4 text-white" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-white" style={{ letterSpacing: '-0.01em' }}>
                            AEMER AI Chat
                        </h3>
                        <p style={{ fontSize: '10px', color: '#06b6d4' }}>Emotion Wellbeing Assistant</p>
                    </div>
                </div>
                <div className="flex items-center gap-1">
                    <button
                        onClick={handleClear}
                        aria-label="Clear chat"
                        className="p-1.5 rounded-lg transition-colors"
                        style={{ color: '#525252' }}
                        onMouseEnter={e => (e.currentTarget.style.color = '#a3a3a3')}
                        onMouseLeave={e => (e.currentTarget.style.color = '#525252')}
                        title="Clear chat"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => setIsOpen(false)}
                        aria-label="Close Chat"
                        className="p-1.5 rounded-lg transition-colors"
                        style={{ color: '#525252' }}
                        onMouseEnter={e => (e.currentTarget.style.color = '#a3a3a3')}
                        onMouseLeave={e => (e.currentTarget.style.color = '#525252')}
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Messages */}
            <div
                className="flex-1 overflow-y-auto px-4 py-3 space-y-3"
                style={{
                    scrollbarWidth: 'thin',
                    scrollbarColor: 'rgba(6, 182, 212, 0.2) transparent',
                }}
            >
                {messages.map((msg, i) => (
                    <div
                        key={i}
                        className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                    >
                        {/* Avatar */}
                        <div
                            className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center"
                            style={{
                                background: msg.role === 'user'
                                    ? 'rgba(6, 182, 212, 0.15)'
                                    : 'linear-gradient(135deg, #06b6d4, #0e7490)',
                            }}
                        >
                            {msg.role === 'user'
                                ? <User className="w-3.5 h-3.5" style={{ color: '#22d3ee' }} />
                                : <Bot className="w-3.5 h-3.5 text-white" />
                            }
                        </div>

                        {/* Bubble */}
                        <div
                            className="max-w-[80%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed"
                            style={{
                                background: msg.role === 'user'
                                    ? 'rgba(6, 182, 212, 0.12)'
                                    : 'rgba(255, 255, 255, 0.04)',
                                color: msg.role === 'user' ? '#e0f7fa' : '#d4d4d4',
                                border: msg.role === 'user'
                                    ? '1px solid rgba(6, 182, 212, 0.15)'
                                    : '1px solid rgba(255, 255, 255, 0.06)',
                                borderBottomRightRadius: msg.role === 'user' ? '6px' : undefined,
                                borderBottomLeftRadius: msg.role === 'assistant' ? '6px' : undefined,
                                whiteSpace: 'pre-wrap',
                            }}
                        >
                            {msg.content}
                        </div>
                    </div>
                ))}

                {/* Loading indicator */}
                {isLoading && (
                    <div className="flex gap-2">
                        <div
                            className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center"
                            style={{ background: 'linear-gradient(135deg, #06b6d4, #0e7490)' }}
                        >
                            <Bot className="w-3.5 h-3.5 text-white" />
                        </div>
                        <div
                            className="rounded-2xl px-4 py-3 flex items-center gap-2"
                            style={{
                                background: 'rgba(255, 255, 255, 0.04)',
                                border: '1px solid rgba(255, 255, 255, 0.06)',
                            }}
                        >
                            <Loader2 className="w-4 h-4 animate-spin" style={{ color: '#06b6d4' }} />
                            <span className="text-xs" style={{ color: '#737373' }}>Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div
                className="px-3 py-3"
                style={{ borderTop: '1px solid rgba(6, 182, 212, 0.08)' }}
            >
                <div
                    className="flex items-center gap-2 rounded-xl px-3 py-2"
                    style={{
                        background: 'rgba(255, 255, 255, 0.04)',
                        border: '1px solid rgba(6, 182, 212, 0.1)',
                    }}
                >
                    <input
                        ref={inputRef}
                        id="ai-chat-input"
                        aria-label="Ask about your emotions"
                        type="text"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about your emotions..."
                        disabled={isLoading}
                        className="flex-1 bg-transparent text-sm text-white placeholder:text-neutral-600 outline-none"
                    />
                    <button
                        onClick={handleSend}
                        aria-label="Send message"
                        disabled={!input.trim() || isLoading}
                        className="p-2 rounded-lg transition-all"
                        style={{
                            background: input.trim()
                                ? 'linear-gradient(135deg, #06b6d4, #0e7490)'
                                : 'rgba(255, 255, 255, 0.05)',
                            opacity: input.trim() ? 1 : 0.4,
                            cursor: input.trim() ? 'pointer' : 'default',
                        }}
                    >
                        <Send className="w-4 h-4 text-white" />
                    </button>
                </div>
                <p className="text-center mt-1.5" style={{ fontSize: '9px', color: '#404040' }}>
                    AI responses are for informational purposes only
                </p>
            </div>

            {/* Slide-up animation keyframes */}
            <style>{`
                @keyframes chatSlideUp {
                    from { opacity: 0; transform: translateY(20px) scale(0.95); }
                    to { opacity: 1; transform: translateY(0) scale(1); }
                }
            `}</style>
        </div>
    );
}
