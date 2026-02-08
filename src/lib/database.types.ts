export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      analyses: {
        Row: {
          id: string
          created_at: string
          input_type: 'audio' | 'video' | 'text' | 'multimodal'
          emotion_label: string
          confidence_score: number | null
          detected_accent: string | null
          audio_weight: number | null
          text_weight: number | null
          visual_weight: number | null
          audio_score: number | null
          text_score: number | null
          visual_score: number | null
          dataset_tag: string | null
          notes: string | null
          file_url: string | null
        }
        Insert: {
          id?: string
          created_at?: string
          input_type: 'audio' | 'video' | 'text' | 'multimodal'
          emotion_label: string
          confidence_score?: number | null
          detected_accent?: string | null
          audio_weight?: number | null
          text_weight?: number | null
          visual_weight?: number | null
          audio_score?: number | null
          text_score?: number | null
          visual_score?: number | null
          dataset_tag?: string | null
          notes?: string | null
          file_url?: string | null
        }
        Update: {
          id?: string
          created_at?: string
          input_type?: 'audio' | 'video' | 'text' | 'multimodal'
          emotion_label?: string
          confidence_score?: number | null
          detected_accent?: string | null
          audio_weight?: number | null
          text_weight?: number | null
          visual_weight?: number | null
          audio_score?: number | null
          text_score?: number | null
          visual_score?: number | null
          dataset_tag?: string | null
          notes?: string | null
          file_url?: string | null
        }
      }
      emotion_reports: {
        Row: {
          id: string
          created_at: string
          date_range_start: string
          date_range_end: string
          emotion_label: string
          total_count: number | null
          avg_confidence: number | null
          accuracy_rate: number | null
        }
        Insert: {
          id?: string
          created_at?: string
          date_range_start: string
          date_range_end: string
          emotion_label: string
          total_count?: number | null
          avg_confidence?: number | null
          accuracy_rate?: number | null
        }
        Update: {
          id?: string
          created_at?: string
          date_range_start?: string
          date_range_end?: string
          emotion_label?: string
          total_count?: number | null
          avg_confidence?: number | null
          accuracy_rate?: number | null
        }
      }
      accent_reports: {
        Row: {
          id: string
          created_at: string
          accent_profile: string
          total_analyses: number | null
          avg_confidence: number | null
          accuracy_rate: number | null
          f1_score: number | null
          error_rate: number | null
        }
        Insert: {
          id?: string
          created_at?: string
          accent_profile: string
          total_analyses?: number | null
          avg_confidence?: number | null
          accuracy_rate?: number | null
          f1_score?: number | null
          error_rate?: number | null
        }
        Update: {
          id?: string
          created_at?: string
          accent_profile?: string
          total_analyses?: number | null
          avg_confidence?: number | null
          accuracy_rate?: number | null
          f1_score?: number | null
          error_rate?: number | null
        }
      }
    }
  }
}
