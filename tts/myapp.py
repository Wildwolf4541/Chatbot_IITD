import os
import tempfile
import streamlit as st
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from audio_recorder_streamlit import audio_recorder
import threading
import queue
import time
import gc

class VoiceAIAssistant:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise (only once)
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def _initialize_tts_engine(self):
        """Initializes or re-initializes the pyttsx3 engine."""
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        return engine

    def audio_to_text(self, audio_data):
        """Convert audio to text using speech recognition"""
        try:
            # Use Google's speech recognition (works with Whisper-like accuracy)
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"
    
    def get_gemini_response(self, text):
        """Get response from Gemini AI"""
        try:
            response = self.model.generate_content(text)
            return response.text
        except Exception as e:
            return f"Error getting AI response: {str(e)}"
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        engine = None
        try:
            engine = self._initialize_tts_engine()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")
        finally:
            if engine:
                # This ensures the speech engine is shut down cleanly after use
                engine.stop()
                # Consider adding a small delay if issues persist
                # time.sleep(0.05)
    
    def safe_file_cleanup(self, file_path, max_attempts=5, delay=0.1):
        """Safely delete temporary file with retry logic"""
        for attempt in range(max_attempts):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                return True
            except PermissionError:
                if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    st.warning(f"Could not delete temporary file: {file_path}")
                    return False
            except Exception as e:
                st.warning(f"Unexpected error deleting file: {e}")
                return False
        return False
    
    def process_voice_input(self, audio_bytes):
        """Process the complete voice-to-voice pipeline"""
        if audio_bytes:
            temp_file_path = None
            audio_data = None
            
            try:
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file.flush()
                    temp_file_path = tmp_file.name
                
                # Convert audio to text
                with sr.AudioFile(temp_file_path) as source:
                    audio_data = self.recognizer.record(source)
                
                # Clear the audio_data reference to help with file unlocking
                del audio_data
                gc.collect()  # Force garbage collection
                
                # Small delay to ensure file is released
                time.sleep(0.1)
                
                # Now process the audio (re-reading it is fine, the file is there)
                with sr.AudioFile(temp_file_path) as source:
                    audio_data = self.recognizer.record(source)
                
                st.write("🎤 *You said:*")
                user_text = self.audio_to_text(audio_data)
                st.write(f"{user_text}")
                
                if "Sorry, I couldn't understand" not in user_text:
                    # Get AI response
                    st.write("🤖 *AI Response:*")
                    ai_response = self.get_gemini_response(user_text)
                    st.write(ai_response)
                    
                    # Convert response to speech
                    st.write("🔊 *Playing AI response...*")
                    
                    # Use threading to prevent blocking
                    speech_thread = threading.Thread(
                        target=self.text_to_speech, 
                        args=(ai_response,)
                    )
                    speech_thread.daemon = True
                    speech_thread.start()
                    
                    # Store conversation history
                    if 'conversation_history' not in st.session_state:
                        st.session_state.conversation_history = []
                    st.session_state.conversation_history.append((user_text, ai_response))
                    
                    return user_text, ai_response
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            
            finally:
                # Clean up references
                if audio_data:
                    del audio_data
                gc.collect()
                
                # Clean up temporary file with retry logic
                if temp_file_path:
                    self.safe_file_cleanup(temp_file_path)
            
        return None, None

def main():
    st.set_page_config(
        page_title="Voice AI Assistant", 
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Voice-to-Voice AI Assistant")
    st.markdown("*Speak → Gemini AI → Hear Response*")
    
    # API Key input in sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from: https://makersuite.google.com/app/apikey"
        )
        
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
            st.success("✅ API Key set successfully!")
        else:
            st.warning("⚠ Please enter your Gemini API key to continue")
        
        # Add settings
        st.markdown("### ⚙ Settings")
        if st.button("🧹 Clear Conversation History"):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
    
    # Check for API key
    if not api_key:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started")
        st.markdown("### 📝 How to get your API key:")
        st.markdown("1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)")
        st.markdown("2. Create a new API key")
        st.markdown("3. Copy and paste it in the sidebar")
        st.stop()
    
    # Initialize the assistant
    if 'assistant' not in st.session_state:
        with st.spinner("Initializing Voice AI Assistant..."):
            try:
                st.session_state.assistant = VoiceAIAssistant()
                st.success("Voice AI Assistant initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing assistant: {str(e)}")
                st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎙 Record Your Voice")
        
        # Audio recorder
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
        
        # Text input as alternative
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
                    
                    # Store conversation history
                    if 'conversation_history' not in st.session_state:
                        st.session_state.conversation_history = []
                    st.session_state.conversation_history.append((user_input, ai_response))
                    
                except Exception as e:
                    st.error(f"Error processing text input: {str(e)}")
    
    # Instructions
    with st.expander("📋 How to Use"):
        st.markdown("""
        1. *Set up your environment:*
            bash
            pip install streamlit speechrecognition pyttsx3 google-generativeai audio-recorder-streamlit pyaudio
            
        
        2. *Voice Mode:*
            - Click the microphone button to start recording
            - Speak your question or message clearly
            - Click "Process Voice Input" to get AI response
            - Listen to the AI response
        
        3. *Text Mode:*
            - Type your message in the text box
            - Click "Send Text" to get AI response
            - Listen to the AI response
        
        4. *Troubleshooting:*
            - If you encounter audio issues, try restarting the application
            - Make sure your microphone is working and accessible
            - Check your internet connection for speech recognition
        """)
    
    # Conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if st.session_state.conversation_history:
        st.markdown("### 📝 Conversation History")
        for i, (user_msg, ai_msg) in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5 conversations
            with st.expander(f"Recent Conversation {len(st.session_state.conversation_history) - i}"):
                st.write(f"*You:* {user_msg}")
                st.write(f"*AI:* {ai_msg}")

if __name__ == "__main__":
    main()