"""
AEMER AI Handler — Google Gemini API Integration
Provides AI-powered emotion insights, chatbot, and report generation.
Uses the free-tier Gemini API (15 RPM / 1,500 RPD).
"""

import os
import json
import traceback
from pathlib import Path

# Load .env from project root (one level up from backend/)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on system env vars

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. AI features disabled.")

# ── Configuration ─────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# System prompts for different AI modes
INSIGHTS_SYSTEM_PROMPT = """You are AEMER's Emotion Insight AI — an empathetic, professional emotional-wellness assistant.

Given emotion analysis results (detected emotion, confidence score, probability distribution, and modality), provide:
1. **Emotional Interpretation** (2-3 sentences): What this emotion means in context, possible triggers.
2. **Psychological Context** (2-3 sentences): The science behind this emotion — what's happening in the brain/body.
3. **Wellness Suggestion** (2-3 sentences): A practical, actionable coping tip suitable for the detected emotion.
4. **Cross-Modal Insight** (1-2 sentences, only if multiple modalities): How the different modalities (audio, text, video) agree or disagree and what that implies.

Keep the total response under 200 words. Be warm, supportive, and non-judgmental. Use simple language.
Do NOT diagnose or provide medical advice. Include a brief disclaimer at the end.
Format using markdown with **bold** for section headers."""

CHAT_SYSTEM_PROMPT = """You are AEMER's Emotional Wellbeing Chatbot — a warm, empathetic AI assistant that helps users understand and manage their emotions.

Context: The user has been using AEMER (Automatic Emotion Recognition) to analyze their emotions through audio, text, and video. You have access to their latest analysis results.

Guidelines:
- Be warm, supportive, and non-judgmental
- Provide practical, evidence-based coping strategies
- Ask thoughtful follow-up questions to understand context
- Keep responses concise (under 150 words)
- Never diagnose mental health conditions
- If someone seems in crisis, gently recommend professional help and provide helpline numbers
- Reference the user's emotion analysis results when relevant
- Use emoji sparingly for warmth 💛"""

REPORT_SYSTEM_PROMPT = """You are AEMER's Report Narrative AI — a professional writer that creates concise, insightful summaries of emotion analysis results.

Given one or more emotion analysis results, write a professional narrative summary that includes:
1. **Overview**: A 2-3 sentence summary of the overall emotional profile
2. **Key Findings**: Notable patterns, dominant emotions, confidence levels
3. **Modality Comparison** (if applicable): How different input types (audio/text/video) compare
4. **Recommendations**: 2-3 brief wellness suggestions based on the findings

Keep the total under 250 words. Use a professional but warm tone. Format with markdown.
Do NOT diagnose. Include a brief disclaimer."""


class AIHandler:
    """Handles all AI-powered features using Google Gemini API."""

    def __init__(self):
        self.model = None
        self.available = False

        if not GEMINI_AVAILABLE:
            print("❌ AI Handler: google-generativeai package not available")
            return

        if not GEMINI_API_KEY:
            print("⚠️  AI Handler: No GEMINI_API_KEY set. AI features will use fallback responses.")
            return

        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            self.available = True
            print("✅ AI Handler: Gemini API configured successfully")
        except Exception as e:
            print(f"❌ AI Handler: Failed to configure Gemini — {e}")

    def _format_emotion_context(self, emotion_data: dict) -> str:
        """Format emotion data into a readable context string."""
        lines = []
        lines.append(f"Detected Emotion: {emotion_data.get('emotion_label', 'unknown')}")
        lines.append(f"Confidence: {emotion_data.get('confidence_score', 0):.1%}")

        if emotion_data.get("input_type"):
            lines.append(f"Input Type: {emotion_data['input_type']}")

        if emotion_data.get("all_probabilities"):
            probs = emotion_data["all_probabilities"]
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            prob_str = ", ".join([f"{e}: {p:.1%}" for e, p in sorted_probs[:5]])
            lines.append(f"Probability Distribution: {prob_str}")

        if emotion_data.get("detected_accent"):
            lines.append(f"Detected Accent: {emotion_data['detected_accent']}")

        if emotion_data.get("modalities_used"):
            lines.append(f"Modalities Used: {', '.join(emotion_data['modalities_used'])}")

        # Cross-modal results
        for mod in ["audio_result", "text_result", "video_result"]:
            if emotion_data.get(mod):
                m_data = emotion_data[mod]
                mod_name = mod.replace("_result", "").title()
                lines.append(
                    f"{mod_name} Analysis: {m_data.get('emotion_label', '?')} "
                    f"({m_data.get('confidence_score', 0):.1%} confidence, "
                    f"weight: {m_data.get('weight', 0):.0%})"
                )

        return "\n".join(lines)

    async def generate_insights(self, emotion_data: dict) -> dict:
        """Generate AI insights for emotion analysis results."""
        if not self.available:
            return self._fallback_insights(emotion_data)

        try:
            context = self._format_emotion_context(emotion_data)
            prompt = f"{INSIGHTS_SYSTEM_PROMPT}\n\n--- Emotion Analysis Results ---\n{context}\n\nPlease provide your emotional insights:"

            response = self.model.generate_content(prompt)
            return {
                "success": True,
                "insights": response.text,
                "source": "gemini"
            }
        except Exception as e:
            print(f"⚠️  Gemini insights error: {e}")
            traceback.print_exc()
            return self._fallback_insights(emotion_data)

    async def chat(self, message: str, emotion_context: dict = None, history: list = None) -> dict:
        """Chat with the AI about emotions."""
        if not self.available:
            return {
                "success": True,
                "response": "I'm currently unavailable. Please set up a Gemini API key to enable AI chat features. "
                            "In the meantime, if you're feeling distressed, please reach out to a mental health helpline.",
                "source": "fallback"
            }

        try:
            # Build conversation prompt
            prompt_parts = [CHAT_SYSTEM_PROMPT]

            if emotion_context:
                context = self._format_emotion_context(emotion_context)
                prompt_parts.append(f"\n--- User's Latest Emotion Analysis ---\n{context}")

            if history:
                prompt_parts.append("\n--- Conversation History ---")
                for msg in history[-10:]:  # Keep last 10 messages for context
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    prompt_parts.append(f"{role}: {msg.get('content', '')}")

            prompt_parts.append(f"\nUser: {message}\nAssistant:")

            full_prompt = "\n".join(prompt_parts)
            response = self.model.generate_content(full_prompt)

            return {
                "success": True,
                "response": response.text,
                "source": "gemini"
            }
        except Exception as e:
            print(f"⚠️  Gemini chat error: {e}")
            traceback.print_exc()
            return {
                "success": True,
                "response": "I'm having trouble connecting right now. Please try again in a moment. "
                            "If you need immediate support, please contact a mental health helpline.",
                "source": "error"
            }

    async def generate_report_narrative(self, results: list) -> dict:
        """Generate AI narrative for reports."""
        if not self.available:
            return self._fallback_report(results)

        try:
            # Format all results
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"\n--- Analysis {i} ---")
                context_parts.append(self._format_emotion_context(result))

            prompt = (
                f"{REPORT_SYSTEM_PROMPT}\n\n"
                f"Total analyses: {len(results)}\n"
                f"{''.join(context_parts)}\n\n"
                f"Please provide the report narrative:"
            )

            response = self.model.generate_content(prompt)
            return {
                "success": True,
                "narrative": response.text,
                "source": "gemini"
            }
        except Exception as e:
            print(f"⚠️  Gemini report error: {e}")
            traceback.print_exc()
            return self._fallback_report(results)

    # ── Fallback responses (when API is unavailable) ──────────────────

    def _fallback_insights(self, emotion_data: dict) -> dict:
        """Provide basic insights without AI."""
        emotion = emotion_data.get("emotion_label", "neutral").lower()
        confidence = emotion_data.get("confidence_score", 0)

        insights_map = {
            "happy": "Your analysis shows signs of happiness and positive affect. This is associated with increased dopamine and serotonin activity. Continue engaging in activities that bring you joy!",
            "sad": "Your analysis detected sadness. It's natural to experience this emotion — it often signals a need for self-care and connection. Consider reaching out to someone you trust or engaging in a comforting activity.",
            "angry": "Your analysis detected anger. This high-arousal emotion often reflects unmet needs or boundary violations. Try deep breathing or a brief walk to regulate before addressing the source.",
            "fear": "Your analysis detected fear or anxiety. This is your body's protective response. Grounding techniques like the 5-4-3-2-1 method can help you feel centered and safe.",
            "surprise": "Your analysis detected surprise. This brief, high-arousal state indicates your brain processing unexpected information. Take a moment to evaluate and adapt to the new situation.",
            "disgust": "Your analysis detected disgust. This protective emotion helps you avoid things that could be harmful. Reflect on what triggered this feeling and consider if boundaries need to be set.",
            "neutral": "Your analysis shows a neutral emotional state. This balanced state is healthy and indicates emotional regulation. It's a good time for reflection and mindful observation.",
        }

        insight_text = insights_map.get(
            emotion,
            "Your emotional state has been analyzed. Consider reflecting on how you're feeling and take a moment for self-care."
        )

        if confidence < 0.5:
            insight_text += "\n\n*Note: The confidence score is relatively low, which means the reading may not fully capture your emotional nuance.*"

        insight_text += "\n\n*⚠️ This is a basic analysis. Enable the Gemini API key for detailed AI-powered insights.*"

        return {
            "success": True,
            "insights": insight_text,
            "source": "fallback"
        }

    def _fallback_report(self, results: list) -> dict:
        """Provide basic report narrative without AI."""
        if not results:
            return {
                "success": True,
                "narrative": "No analysis results available for report generation.",
                "source": "fallback"
            }

        emotions = [r.get("emotion_label", "unknown") for r in results]
        from collections import Counter
        emotion_counts = Counter(emotions)
        dominant = emotion_counts.most_common(1)[0]

        narrative = (
            f"## Emotion Analysis Summary\n\n"
            f"This report covers **{len(results)}** emotion analysis session(s). "
            f"The dominant emotion detected was **{dominant[0]}** "
            f"(appearing in {dominant[1]} of {len(results)} analyses).\n\n"
            f"*Enable the Gemini API key for a detailed AI-generated narrative.*"
        )

        return {
            "success": True,
            "narrative": narrative,
            "source": "fallback"
        }
