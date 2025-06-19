import os
import tempfile
import streamlit as st
from gtts import gTTS
from io import BytesIO
import google.generativeai as genai
from audio_recorder_streamlit import audio_recorder
import time
import gc
import base64
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
from textwrap import wrap

SUPPORTED_LANGUAGES = {
    'en-IN': 'English',
    'hi-IN': 'Hindi',
    'bn-IN': 'Bengali',
    'ta-IN': 'Tamil',
    'te-IN': 'Telugu',
    'kn-IN': 'Kannada',
    'gu-IN': 'Gujarati',
    'mr-IN': 'Marathi',
    'ml-IN': 'Malayalam',
    'ur-IN': 'Urdu'
}

class VoiceAIAssistant:
    def __init__(self):
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Failed to configure Gemini API: {e}")
            st.stop()
        self.whisper_model = WhisperModel("base", device="cpu")

    def audio_to_text(self, audio_data, lang_code='en-IN'):
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                temp_file_path = tmp_file.name

            short_lang_code = lang_code.split("-")[0]
            segments, _ = self.whisper_model.transcribe(temp_file_path, beam_size=5, language=short_lang_code)
            text = " ".join([segment.text for segment in segments])
            return text.strip()

        except Exception as e:
            return f"Error in STT: {str(e)}"
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    def get_gemini_response(self, text, language_name):
        try:
            prompt = f"Please respond in {language_name}. User's query: '{text}'"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting AI response: {str(e)}"

    def split_text_for_tts(self, text, max_len=200):
        return wrap(text, max_len, break_long_words=False, replace_whitespace=False)

    def text_to_speech_autoplay(self, text, lang_code='en-IN'):
        if not text:
            st.warning("No text to convert to speech.")
            return
        try:
            lang = lang_code.split('-')[0]
            text_chunks = self.split_text_for_tts(text)
            for chunk in text_chunks:
                tts = gTTS(text=chunk, lang=lang, slow=False)
                mp3_fp = BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                b64 = base64.b64encode(mp3_fp.read()).decode()
                audio_html = f"""
                <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
                components.html(audio_html, height=0, width=0)
                time.sleep(1.5)
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")

    def process_voice_input(self, audio_bytes, lang_code, lang_name):
        if not audio_bytes:
            return None, None

        user_text, ai_response = None, None
        try:
            with st.spinner(f"Transcribing your voice..."):
                user_text = self.audio_to_text(audio_bytes, lang_code=lang_code)
            st.info(f"🎤 You said: {user_text}")

            if "Sorry, I couldn't understand" not in user_text and user_text:
                with st.spinner(f"Generating response in {lang_name}..."):
                    ai_response = self.get_gemini_response(user_text, lang_name)
                st.success("🤖 AI Response:")
                st.write(ai_response)
                st.info("🔊 Playing AI response...")
                self.text_to_speech_autoplay(ai_response, lang_code=lang_code)

        except Exception as e:
            st.error(f"An error occurred in the processing pipeline: {str(e)}")
        finally:
            gc.collect()

        return user_text, ai_response

def main():
    st.set_page_config(page_title="Multilingual AI Assistant", page_icon="🌐", layout="wide")
    st.title("🌐 Multilingual AI Assistant: Voice & Text")

    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input("Enter your Gemini API Key:", type="password")
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
            st.success("✅ API Key Set!")
        else:
            st.warning("⚠ Please enter your Gemini API key.")

        st.markdown("---")
        st.markdown("### ⚙ Settings")
        selected_lang_name = st.selectbox("Choose your language:", list(SUPPORTED_LANGUAGES.values()))
        lang_code = [code for code, name in SUPPORTED_LANGUAGES.items() if name == selected_lang_name][0]
        st.session_state.language_code = lang_code
        st.session_state.language_name = selected_lang_name

        if st.button("🧹 Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.last_audio_processed = None
            st.success("Conversation history cleared!")

    if not api_key:
        st.info("👈 Enter your Gemini API key in the sidebar to start.")
        st.stop()

    if 'assistant' not in st.session_state:
        with st.spinner("Initializing AI Assistant..."):
            st.session_state.assistant = VoiceAIAssistant()

    if 'last_audio_processed' not in st.session_state:
        st.session_state.last_audio_processed = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 🎙 Record Your Voice")
        st.write(f"(Input language: *{st.session_state.language_name}*)")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone-lines",
            icon_size="2x"
        )

        if audio_bytes and audio_bytes != st.session_state.get('last_audio_processed'):
            st.audio(audio_bytes, format="audio/wav")
            user_text, ai_response = st.session_state.assistant.process_voice_input(
                audio_bytes,
                st.session_state.language_code,
                st.session_state.language_name
            )
            st.session_state.last_audio_processed = audio_bytes

            if user_text and ai_response:
                if 'conversation_history' not in st.session_state:
                    st.session_state.conversation_history = []
                st.session_state.conversation_history.append((user_text, ai_response))

    with col2:
        st.markdown("### 💬 Or, Chat with Text")
        st.write(f"(Response language: *{st.session_state.language_name}*)")
        user_input = st.text_area("Type your message here:", key="text_input")

        if st.button("✉ Send Text", key="send_text"):
            if user_input:
                try:
                    with st.spinner(f"Getting response in {st.session_state.language_name}..."):
                        ai_response = st.session_state.assistant.get_gemini_response(
                            user_input, st.session_state.language_name
                        )
                    st.success("🤖 AI Response:")
                    st.write(ai_response)
                    st.info("🔊 Playing AI response...")
                    st.session_state.assistant.text_to_speech_autoplay(
                        ai_response, st.session_state.language_code
                    )
                    if 'conversation_history' not in st.session_state:
                        st.session_state.conversation_history = []
                    st.session_state.conversation_history.append((user_input, ai_response))
                except Exception as e:
                    st.error(f"Error processing text input: {e}")
            else:
                st.warning("Please enter a message to send.")

    st.markdown("---")
    st.markdown("### 📝 Conversation History")
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if not st.session_state.conversation_history:
        st.info("Your conversation will appear here.")
    else:
        for i, (user_msg, ai_msg) in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"Conversation #{len(st.session_state.conversation_history) - i}", expanded=(i == 0)):
                st.markdown(f"*You:*\n> {user_msg}")
                st.markdown(f"*AI:*\n> {ai_msg}")

if __name__ == "__main__":
    main()
