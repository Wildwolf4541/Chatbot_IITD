import os
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import threading
import asyncio
import edge_tts
from io import BytesIO
import gc
import tempfile
from pygame import mixer

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Chinese (Mandarin)": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Italian": "it",
}

VOICE_MAPPING = {
    "en": "en-US-GuyNeural",
    "hi": "hi-IN-MadhurNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "ru": "ru-RU-DmitryNeural",
    "it": "it-IT-IsabellaNeural",
}

class VoiceAIAssistant:
    def __init__(self, language='en'):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.recognizer = sr.Recognizer()
        self.language = language
        self.whisper_model = WhisperModel("base", device="cpu")

    def audio_to_text(self, audio_bytes):
        try:
            audio_file = BytesIO(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                wav_path = "temp.wav"
                with open(wav_path, "wb") as f:
                    f.write(audio_bytes)
                segments, _ = self.whisper_model.transcribe(wav_path, beam_size=5)
                text = " ".join([seg.text for seg in segments])
                os.remove(wav_path)
                return text
        except Exception as e:
            return f"Error in STT: {str(e)}"

    def get_gemini_response(self, text):
        try:
            prompt = f"Respond in {self.get_language_name()}:\n{text}"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting AI response: {str(e)}"

    async def text_to_speech(self, text):
        try:
            voice = VOICE_MAPPING.get(self.language, "en-US-GuyNeural")
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            communicate = edge_tts.Communicate(text, voice=voice)
            await communicate.save(output_path)

            mixer.init()
            mixer.music.load(output_path)
            mixer.music.play()
            while mixer.music.get_busy():
                await asyncio.sleep(0.1)
            mixer.music.unload()
            mixer.quit()

            os.remove(output_path)
        except Exception as e:
            st.error(f"Error in TTS: {str(e)}")

    def get_language_name(self):
        for name, code in LANGUAGES.items():
            if code == self.language:
                return name
        return "English"

    def process_voice_input(self, audio_bytes):
        if audio_bytes:
            try:
                user_text = self.audio_to_text(audio_bytes)
                st.write("🎤 *You said:*")
                st.write(user_text)

                if "Error" not in user_text:
                    st.write("🤖 *AI Response:*")
                    ai_response = self.get_gemini_response(user_text)
                    st.write(ai_response)

                    st.write("🔊 *Playing AI response...*")
                    asyncio.run(self.text_to_speech(ai_response))

                    if 'conversation_history' not in st.session_state:
                        st.session_state.conversation_history = []
                    st.session_state.conversation_history.append((user_text, ai_response))

                    return user_text, ai_response
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            finally:
                gc.collect()
        return None, None

def main():
    st.set_page_config(page_title="Voice AI Assistant", page_icon="🎤", layout="wide")
    st.title("🎤 Voice-to-Voice AI Assistant (Optimized)")

    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input("Enter your Gemini API Key:", type="password")
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
            st.success("✅ API Key set successfully!")
        else:
            st.warning("⚠ Please enter your Gemini API key to continue")

        st.markdown("### 🌐 Select Language")
        selected_lang_name = st.selectbox("Choose Language:", options=list(LANGUAGES.keys()), index=0)
        selected_lang_code = LANGUAGES[selected_lang_name]

        if st.button("🧹 Clear Conversation History"):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")

    if not api_key:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started")
        st.stop()

    if 'assistant' not in st.session_state or st.session_state.assistant.language != selected_lang_code:
        with st.spinner("Initializing Voice AI Assistant..."):
            try:
                st.session_state.assistant = VoiceAIAssistant(language=selected_lang_code)
                st.success("Voice AI Assistant initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing assistant: {str(e)}")
                st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎙 Record Your Voice")
        audio_bytes = audio_recorder(text="Click to start recording")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("🚀 Process Voice Input", type="primary"):
                with st.spinner("Processing your voice..."):
                    user_text, ai_response = st.session_state.assistant.process_voice_input(audio_bytes)
                    if user_text and ai_response:
                        st.success("✅ Voice processing completed!")

    with col2:
        st.markdown("### 💬 Text Chat (Optional)")
        user_input = st.text_input("Or type your message here:")
        if st.button("Send Text", type="secondary"):
            if user_input:
                try:
                    st.write("🤖 *AI Response:*")
                    ai_response = st.session_state.assistant.get_gemini_response(user_input)
                    st.write(ai_response)
                    st.write("🔊 *Playing AI response...*")
                    asyncio.run(st.session_state.assistant.text_to_speech(ai_response))
                    if 'conversation_history' not in st.session_state:
                        st.session_state.conversation_history = []
                    st.session_state.conversation_history.append((user_input, ai_response))
                except Exception as e:
                    st.error(f"Error processing text input: {str(e)}")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if st.session_state.conversation_history:
        st.markdown("### 📝 Conversation History")
        for i, (user_msg, ai_msg) in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"Recent Conversation {len(st.session_state.conversation_history) - i}"):
                st.write(f"*You:* {user_msg}")
                st.write(f"*AI:* {ai_msg}")

if __name__ == "__main__":
    main()
