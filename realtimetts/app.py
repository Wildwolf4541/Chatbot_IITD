import os
import base64
import threading
import io # For in-memory file operations

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import speech_recognition as sr
from pydub import AudioSegment # For audio conversion

# Initialize Flask & SocketIO
app = Flask("voice_app")
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'your-default-secret-key') # Use env var for secret
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Configure Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model = None # Initialize to None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9, # Slightly increased for more naturalness
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    try:
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash", # Or "gemini-1.5-pro-latest"
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=(
                "You are a friendly and engaging voice assistant. "
                "Keep your responses conversational, concise (generally 1-3 sentences unless more detail is clearly needed), "
                "and natural for speech. Refer to previous parts of the conversation if relevant."
            )
        )
        print("✅ Gemini API configured successfully.")
    except Exception as e:
        print(f"❌ Error initializing Gemini Model: {e}")
        gemini_model = None
else:
    print("⚠️ GEMINI_API_KEY not found. AI features will use fallback responses.")

# --- Speech Recognition Setup ---
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300 
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8 # Less relevant when processing complete client-side chunks

# --- Conversation State Management ---
chat_sessions = {} # user_id (socket.sid) -> Gemini ChatSession object

# --- Flask Routes ---
@app.route("/")
def index():
    """Serves the main HTML page."""
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"❌ Error rendering index.html: {e}")
        return "Error: Could not load application page.", 500

@app.route("/test-gemini", methods=["GET"])
def test_gemini_route():
    """Test endpoint for Gemini API connectivity."""
    if not gemini_model:
        return jsonify({"error": "Gemini API not configured or model initialization failed"}), 500
    try:
        # Using generate_content for a simple, non-chat test
        response = gemini_model.generate_content("Say hello in a friendly way for an API test!")
        return jsonify({
            "success": True,
            "response": response.text,
            "model_name": gemini_model.model_name # Use attribute if available, else hardcode
        })
    except Exception as e:
        print(f"❌ Error during /test-gemini: {e}")
        return jsonify({"error": str(e)}), 500

# --- Helper Functions ---
def process_audio_with_gemini(audio_data_base64, user_id):
    """
    Transcribes audio, gets a response from Gemini, and handles conversation context.
    """
    try:
        audio_bytes = base64.b64decode(audio_data_base64.split(',')[1] if ',' in audio_data_base64 else audio_data_base64)
        # print(f"🎤 Received audio for STT from {user_id}. Raw bytes: {len(audio_bytes)}") # Verbose log

        if len(audio_bytes) < 500: # Very short audio is unlikely to be valid speech
            print(f"⚠️ Audio from {user_id} is very short ({len(audio_bytes)} bytes), likely STT will fail.")
            # Optionally return error early, or let STT try
            # return {"success": False, "error": "Audio too short", "user_id": user_id}


        # Convert webm/opus (from client) to WAV for speech_recognition
        # This requires FFmpeg to be installed and in the system PATH
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source) # Record the entire WAV content
        
        user_text = recognizer.recognize_google(audio)
        print(f"🎤 User ({user_id}): \"{user_text}\"")

        ai_response_text = "Sorry, I'm having trouble thinking right now." # Default
        if gemini_model and user_id in chat_sessions:
            chat = chat_sessions[user_id]
            try:
                gemini_response = chat.send_message(user_text)
                ai_response_text = gemini_response.text
                # print(f"🤖 AI ({user_id}): \"{ai_response_text}\"") # Verbose log
            except Exception as e_gemini:
                print(f"❌ Gemini API error for {user_id}: {e_gemini}")
                # Check for specific Gemini block reasons
                if hasattr(e_gemini, 'response') and hasattr(e_gemini.response, 'prompt_feedback') and e_gemini.response.prompt_feedback.block_reason:
                    ai_response_text = f"I can't respond to that due to content safety: {e_gemini.response.prompt_feedback.block_reason_message}"
                else:
                    ai_response_text = "I encountered an issue with the AI model."
        elif not gemini_model:
            ai_response_text = get_fallback_response(user_text.lower())
        
        return {
            "success": True,
            "transcription": user_text,
            "response": ai_response_text,
            "user_id": user_id
        }

    except sr.UnknownValueError:
        print(f"🚫 STT Error ({user_id}): Could not understand audio. (Bytes: {len(audio_bytes) if 'audio_bytes' in locals() else 'N/A'})")
        return {"success": False, "error": "Could not understand audio", "user_id": user_id}
    except sr.RequestError as e:
        print(f"🚫 STT Error ({user_id}): RequestError - {e}")
        return {"success": False, "error": f"Speech recognition service error: {e}", "user_id": user_id}
    except Exception as e:
        print(f"❌ Unexpected error in process_audio_with_gemini for {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": "Internal server error during audio processing.", "user_id": user_id}

def get_fallback_response(text_query):
    """Provides simple fallback responses if AI model is unavailable."""
    text_query = text_query.lower()
    if 'hello' in text_query or 'hi' in text_query:
        return "Hello! How can I help you today?"
    if 'how are you' in text_query:
        return "I'm doing well, thank you for asking!"
    # Add more fallback rules as needed
    return "I'm sorry, I can't access my advanced functions right now. Please try again later."

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def on_connect():
    user_id = request.sid
    print(f"✅ User connected: {user_id}")
    if gemini_model:
        # Each user gets their own independent chat session
        chat_sessions[user_id] = gemini_model.start_chat(history=[])
        print(f"💬 New Gemini chat session created for {user_id}")
    else:
        print(f"⚠️ Gemini model not available. Chat session NOT created for {user_id}.")
    emit('connection_status', {'status': 'connected', 'user_id': user_id, 'model_ready': bool(gemini_model)})

@socketio.on('disconnect')
def on_disconnect():
    user_id = request.sid
    print(f"❌ User disconnected: {user_id}")
    if user_id in chat_sessions:
        del chat_sessions[user_id]
        print(f"🗑️ Chat session removed for {user_id}")

@socketio.on('audio_data')
def handle_audio_data(data):
    user_id = request.sid
    audio_data_base64 = data.get('audio')

    if not audio_data_base64:
        emit('audio_response', {'success': False, 'error': 'No audio data received'})
        return
    
    # Process audio in a separate thread to avoid blocking SocketIO
    def process_audio_thread_target():
        result = process_audio_with_gemini(audio_data_base64, user_id)
        socketio.emit('audio_response', result, room=user_id) # Emit only to the sender

    thread = threading.Thread(target=process_audio_thread_target)
    thread.daemon = True # Allows main program to exit even if threads are running
    thread.start()

@socketio.on('clear_conversation')
def handle_clear_conversation():
    user_id = request.sid
    if gemini_model and user_id in chat_sessions:
        chat_sessions[user_id] = gemini_model.start_chat(history=[]) # Reset session
        print(f"🔄 Conversation cleared for {user_id}")
        emit('conversation_cleared', {'status': 'success', 'message': 'Conversation history cleared.'})
    else:
        emit('conversation_cleared', {'status': 'failure', 'message': 'Could not clear (model/session issue).'})

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting Real-time Voice AI Server...")
    if not GEMINI_API_KEY:
        print("🔴 CRITICAL: GEMINI_API_KEY environment variable is NOT SET. AI features will be disabled.")
    elif not gemini_model:
        print("🟠 WARNING: Gemini model initialization FAILED. AI features disabled.")
    
    # use_reloader=False is important for multi-threading with SocketIO dev server
    # In production, use a proper WSGI server like Gunicorn or uWSGI
    socketio.run(app, host="0.0.0.0", port=8116, debug=True, use_reloader=False)