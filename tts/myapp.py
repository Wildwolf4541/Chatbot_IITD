import os
import tempfile
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
from audio_recorder_streamlit import audio_recorder
import threading
import queue
import time
import gc
from gtts import gTTS
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
    # add more if you want
}

class VoiceAIAssistant:
    def __init__(self, language='en'):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def audio_to_text(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data, language=self.language)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

    def get_gemini_response(self, text):
        try:
            # Optional: force Gemini to respond in selected language
            prompt = f"Respond in {self.get_language_name()}:\n{text}"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting AI response: {str(e)}"

    def text_to_speech(self, text):
        try:
            tts = gTTS(text=text, lang=self.language)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                tmp_file.flush()
                temp_path = tmp_file.name
            
            mixer.init()
            mixer.music.load(temp_path)
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            mixer.music.unload()
            mixer.quit()
            self.safe_file_cleanup(temp_path)
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")

    def safe_file_cleanup(self, file_path, max_attempts=5, delay=0.1):
        for attempt in range(max_attempts):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                return True
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    st.warning(f"Could not delete temporary file: {file_path}")
                    return False
            except Exception as e:
                st.warning(f"Unexpected error deleting file: {e}")
                return False
        return False

    def process_voice_input(self, audio_bytes):
        if audio_bytes:
            temp_file_path = None
            audio_data = None
            
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file.flush()
                    temp_file_path = tmp_file.name
                
                with sr.AudioFile(temp_file_path) as source:
                    audio_data = self.recognizer.record(source)

                user_text = self.audio_to_text(audio_data)
                st.write("🎤 *You said:*")
                st.write(user_text)
                
                if "Sorry, I couldn't understand" not in user_text:
                    st.write("🤖 *AI Response:*")
                    ai_response = self.get_gemini_response(user_text)
                    st.write(ai_response)

                    st.write("🔊 *Playing AI response...*")
                    speech_thread = threading.Thread(
                        target=self.text_to_speech,
                        args=(ai_response,)
                    )
                    speech_thread.daemon = True
                    speech_thread.start()

                    if 'conversation_history' not in st.session_state:
                        st.session_state.conversation_history = []
                    st.session_state.conversation_history.append((user_text, ai_response))

                    return user_text, ai_response

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

            finally:
                if audio_data:
                    del audio_data
                gc.collect()

                if temp_file_path:
                    self.safe_file_cleanup(temp_file_path)
        return None, None

    def get_language_name(self):
        # Return language name from code for prompt clarity
        for name, code in LANGUAGES.items():
            if code == self.language:
                return name
        return "English"

def main():
    st.set_page_config(page_title="Voice AI Assistant", page_icon="🎤", layout="wide")
    st.title("🎤 Voice-to-Voice AI Assistant")
    st.markdown("*Speak → Gemini AI → Hear Response*")

    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input("Enter your Gemini API Key:", type="password", help="Get your API key from: https://makersuite.google.com/app/apikey")
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
            st.success("✅ API Key set successfully!")
        else:
            st.warning("⚠ Please enter your Gemini API key to continue")

        st.markdown("### 🌐 Select Language")
        selected_lang_name = st.selectbox("Choose Language:", options=list(LANGUAGES.keys()), index=0)
        selected_lang_code = LANGUAGES[selected_lang_name]

        st.markdown("### ⚙ Settings")
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
        audio_bytes = audio_recorder(
            text="Click to start recording",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone-lines",
            icon_size="2x"
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("🚀 Process Voice Input", type="primary"):
                with st.spinner("Processing your voice..."):
                    try:
                        user_text, ai_response = st.session_state.assistant.process_voice_input(audio_bytes)
                        if user_text and ai_response:
                            st.success("✅ Voice processing completed!")
                    except Exception as e:
                        st.error(f"Error processing voice: {str(e)}")

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
                    speech_thread = threading.Thread(
                        target=st.session_state.assistant.text_to_speech,
                        args=(ai_response,)
                    )
                    speech_thread.daemon = True
                    speech_thread.start()
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
