import os
import sys
import numpy as np
import torch
import librosa
import random
from gtts import gTTS
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from openai import OpenAI

class PronunciationAnalyzer:
    """A class to analyze pronunciation by comparing TTS and user audio."""
    
    def __init__(self):
        self.TTS_AUDIO_PATH = 'actual_audio.wav'
        # Initialize OpenAI client with NVIDIA API settings
        self.llm_client = OpenAI(
            api_key="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",
            base_url="https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/d88351c8-5cf3-47d5-a49a-6c005840431e",
            default_headers={
                "Authorization": f"Bearer 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"
            }
        )
        # Initialize speech models
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def _compare_letters(self, text1, text2):
        """Compare letters between two texts."""
        text1 = text1.lower().replace(" ", "")
        text2 = text2.lower().replace(" ", "")
        
        mismatched = []
        for i, letter in enumerate(text1):
            if i >= len(text2) or letter != text2[i]:
                mismatched.append(letter)
        
        if len(text2) > len(text1):
            mismatched.extend(text2[len(text1):])
            
        return mismatched

    def _pad_segment_to_length(self, seg1, seg2):
        """Pad audio segments to equal length."""
        max_len = max(seg1.shape[1], seg2.shape[1])
        seg1 = np.pad(seg1, ((0, 0), (0, max_len - seg1.shape[1])), mode='constant')
        seg2 = np.pad(seg2, ((0, 0), (0, max_len - seg2.shape[1])), mode='constant')
        return seg1, seg2

    def _compute_overall_similarity(self, scores):
        """Compute overall similarity score from segment scores."""
        if not scores:
            return 0
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        normalized_score = ((max_score - avg_score) / (max_score - min_score + 1e-6)) * 2.0 + 0.1
        return max(0, min(100, normalized_score * 100))

    def _extract_mfcc_features(self, audio_path):
        """Extract MFCC features from audio file."""
        audio, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return (mfcc - np.mean(mfcc)) / np.std(mfcc)

    def _generate_similarity_score(self, mfcc1, mfcc2):
        """Generate similarity score between two MFCC features."""
        mfcc1 = (mfcc1 - np.mean(mfcc1, axis=1, keepdims=True)) / np.std(mfcc1, axis=1, keepdims=True)
        mfcc2 = (mfcc2 - np.mean(mfcc2, axis=1, keepdims=True)) / np.std(mfcc2, axis=1, keepdims=True)

        segment_length = 20
        segments1 = [mfcc1[:, i:i+segment_length] for i in range(0, mfcc1.shape[1], segment_length)]
        segments2 = [mfcc2[:, i:i+segment_length] for i in range(0, mfcc2.shape[1], segment_length)]

        similarities = [
            np.linalg.norm(
                self._pad_segment_to_length(s1, s2)[0] - self._pad_segment_to_length(s1, s2)[1]
            ) for s1, s2 in zip(segments1, segments2)
        ]

        return self._compute_overall_similarity(similarities)

    def _transcribe_audio(self, audio_path):
        """Transcribe audio file to text."""
        audio, _ = librosa.load(audio_path, sr=16000)
        input_values = self.tokenizer(audio, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return self.tokenizer.decode(predicted_ids[0]).lower()

    def _create_fallback_tips(self, mis_matching):
        """Create fallback pronunciation tips when LLM is unavailable."""
        tips = ["Hi! Let's work on improving your pronunciation together."]
        
        for word in mis_matching:
            tips.append(f"""
For the word '{word}':
- Try saying it slowly: {'-'.join(word)}
- Practice each part separately
- Listen carefully and try to match the sound""")
        
        tips.append("""
General tips:
1. Take your time - speaking slowly is okay!
2. Practice in front of a mirror
3. Keep trying - you're doing great!""")
        
        return "\n".join(tips)

    def _get_pronunciation_tips(self, tts_transcription, user_transcription, mis_matching):
        """Generate pronunciation tips using LLM."""
        if not mis_matching:
            return "Great job! Your pronunciation is clear and accurate! Keep up the excellent work!"

        try:
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Speaky, a friendly and encouraging speech therapist specialized in helping children aged 10-12 improve their pronunciation. Your responses should be clear, supportive, and easy to understand."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze the pronunciation difference and provide tips:

Target word/phrase: "{tts_transcription}"
User's pronunciation: "{user_transcription}"
Sounds needing improvement: {', '.join(mis_matching)}

Provide a response that includes:
1. A warm greeting (1 sentence)
2. For each challenging sound:
   - Exact mouth position
   - Step-by-step breakdown
   - Simple practice word examples
3. Two simple, fun practice exercises
4. A short encouraging message"""
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False
            }

            try:
                response = self.llm_client.post("", json=payload)
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        tips = response_data['choices'][0]['message']['content'].strip()
                        return tips if len(tips) >= 50 else self._create_fallback_tips(mis_matching)
                return self._create_fallback_tips(mis_matching)
                
            except Exception as api_error:
                print(f"API Error: {str(api_error)}")
                return self._create_fallback_tips(mis_matching)

        except Exception as e:
            print(f"General Error in tips generation: {str(e)}")
            return self._create_fallback_tips(mis_matching)

    # Only showing the relevant changes needed to fix the error

    def analyze(self, input_text, user_audio_file):
        """Main method to analyze pronunciation."""
        print('You are in analyze pronunciation function')
        
        if not os.path.exists(user_audio_file):
            return {
                'error': 'Audio file not found',
                'similarity_score': 0,
                'mismatched_words': [],
                'tips': 'Please provide a valid audio file.'
            }

        # Generate TTS audio
        tts = gTTS(input_text)
        tts.save(self.TTS_AUDIO_PATH)

        try:
            # Extract features and compare
            tts_mfcc = self._extract_mfcc_features(self.TTS_AUDIO_PATH)
            user_mfcc = self._extract_mfcc_features(user_audio_file)
            similarity_score = self._generate_similarity_score(tts_mfcc, user_mfcc)

            # Transcribe and analyze
            user_transcription = self._transcribe_audio(user_audio_file)
            mis_matching = self._compare_letters(input_text, user_transcription)

            # Adjust similarity score based on performance
            if len(mis_matching) == 0 and similarity_score >= 100:
                similarity_score = 85 + random.uniform(1, 4)
            elif len(mis_matching) != 0 and similarity_score >= 100:
                similarity_score = 70 + random.uniform(1, 8)
            else:
                similarity_score = max(0, similarity_score - 10)

            tips = self._get_pronunciation_tips(input_text, user_transcription, mis_matching)

            return {
                'similarity_score': similarity_score,
                'mismatched_words': mis_matching,
                'tips': tips,
                'transcription': user_transcription
            }

        finally:
            # Clean up TTS audio file
            if os.path.exists(self.TTS_AUDIO_PATH):
                os.remove(self.TTS_AUDIO_PATH)

# Create a global instance of the analyzer
_analyzer = PronunciationAnalyzer()

def analyze(input_text, user_audio_file):
    """Global function to analyze pronunciation."""
    return _analyzer.analyze(input_text, user_audio_file)