import os
import base64
import threading
import io
import numpy as np
import torch
import time
from pydub import AudioSegment

from flask import Flask, render_template, jsonify, request
# CORRECTED IMPORT: Includes emit directly for easier use in handlers
from flask_socketio import SocketIO, emit
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import whisper

# --- gTTS specific import ---
try:
    from gtts import gTTS, gTTSError
    GTTS_AVAILABLE = True
    print("✅ gTTS library imported successfully.")
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️ gTTS library not found. Server-side TTS with gTTS will be disabled.")
    print("   Install it with: pip install gTTS")

# Initialize Flask & SocketIO
app = Flask(__name__) # Use __name__ which is standard practice
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'your-default-secret-key-CHANGE-ME')
socketio_app = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Configure Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_output_tokens": 1024}
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    # ★★★ CRITICAL CHANGE: NEW, MORE POWERFUL SYSTEM INSTRUCTION ★★★
    system_instruction_text = """You are a friendly and engaging multilingual voice assistant. Your primary goal is to assist the user effectively by strictly following these language processing rules:

1.  **Identify User's Language:** First, analyze the user's input text to determine its primary language (e.g., English, Hindi, Spanish, etc.).

2.  **Apply Response Language Rule:** Based on the language you identified, you MUST choose your response language as follows:
    *   **If the user's language is English ("en"):** You MUST respond in English.
    *   **If the user's language is Hindi ("hi"):** You MUST respond in Hindi.
    *   **If the user's language is ANY OTHER language (e.g., Spanish, French, German, etc.):** You MUST formulate and provide your entire response EXCLUSIVELY in ENGLISH. Do not translate your English response back to the user's original language.

3.  **Response Style:** Regardless of the language, always keep your responses:
    *   Conversational and friendly.
    *   Concise (usually 1-3 sentences, unless more detail is clearly required by the question).
    *   Natural-sounding, as if you are speaking.
    *   Refer to previous parts of the conversation if relevant to provide context and continuity.
"""
    try:
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction_text
        )
        print("✅ Gemini API configured successfully with multilingual system instruction.")
    except Exception as e:
        print(f"❌ Error initializing Gemini Model: {e}")
        gemini_model = None
else:
    print("⚠️ GEMINI_API_KEY not found. AI features will use fallback responses.")

# --- Whisper STT Setup ---
whisper_model = None
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium")
DEBUG_SAVE_AUDIO = os.environ.get("DEBUG_SAVE_AUDIO", "False").lower() == "true"
device_whisper = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print(f"Attempting to load Whisper model '{WHISPER_MODEL_SIZE}' on device '{device_whisper}'...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device_whisper)
    print(f"✅ Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully on '{device_whisper}'.")
except Exception as e:
    print(f"❌ Error loading Whisper model '{WHISPER_MODEL_SIZE}': {e}")
    whisper_model = None


# --- gTTS Synthesis Function ---
def synthesize_speech_with_gtts(text, lang_code_iso):
    if not GTTS_AVAILABLE:
        print("gTTS library not available for synthesis.")
        return None, None

    # gTTS expects basic language codes like 'en', 'hi', not 'en-US'
    base_lang_code = lang_code_iso.split('-')[0].lower()
    print(f"Synthesizing with gTTS for lang='{base_lang_code}'...")
    try:
        tts = gTTS(text=text, lang=base_lang_code, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        # Return raw bytes and the correct MIME type
        return mp3_fp.getvalue(), "audio/mpeg"

    except gTTSError as e_gtts:
        print(f"❌ Error during gTTS synthesis for '{base_lang_code}': {e_gtts}")
        return None, None
    except Exception as e_synth:
        print(f"❌ Unexpected error during gTTS synthesis for '{base_lang_code}': {e_synth}")
        import traceback
        traceback.print_exc()
        return None, None

# --- Conversation State Management ---
chat_sessions = {}

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test-gemini", methods=["GET"])
def test_gemini_route():
    if not gemini_model:
        return jsonify({"error": "Gemini API not configured"}), 500
    try:
        # Test with a simple English prompt
        response = gemini_model.generate_content("Say hello in English for an API test!")
        ai_response_text = response.text
        ai_response_language = "en" # The response language is English

        response_audio_base64 = None
        tts_mime_type = None

        if GTTS_AVAILABLE:
            audio_bytes, mime_type = synthesize_speech_with_gtts(ai_response_text, ai_response_language)
            if audio_bytes:
                response_audio_base64 = f"data:{mime_type};base64," + base64.b64encode(audio_bytes).decode('utf-8')
                tts_mime_type = mime_type

        return jsonify({
            "success": True,
            "response_text": ai_response_text,
            "response_audio_base64": response_audio_base64,
            "audio_mime_type": tts_mime_type,
            "model_name": gemini_model.model_name if gemini_model else "N/A"
        })
    except Exception as e:
        print(f"❌ Error during /test-gemini: {e}")
        return jsonify({"error": str(e)}), 500

# --- Main Audio Processing Function ---
def process_audio_with_gemini(audio_data_base64, user_id):
    if not whisper_model:
        return {"success": False, "error": "Speech recognition model not available.", "user_id": user_id}

    try:
        # 1. Decode and Prepare Audio for Whisper
        audio_bytes = base64.b64decode(audio_data_base64.split(',')[1])
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        # Normalize audio
        max_possible_val = 2**(audio_segment.sample_width * 8 - 1)
        if max_possible_val == 0: max_possible_val = 32768
        samples /= max_possible_val

        # 2. Transcribe and Detect Language with Whisper
        transcription_result = whisper_model.transcribe(samples, fp16=(device_whisper=="cuda"))
        user_text = transcription_result["text"].strip()
        detected_language_code = transcription_result["language"]
        print(f"🗣️ Whisper STT ({user_id}): Lang='{detected_language_code}', Text='{user_text}'")

        # 3. Filter out simple, short phrases to avoid accidental triggers
        common_false_positives = { "en": ["thank you", "okay", "yes", "no", "hi", "hello", "bye"], "hi": ["ठीक है", "हाँ", "नहीं", "नमस्ते", "धन्यवाद"]}
        audio_duration_seconds = len(samples) / 16000.0
        text_word_count = len(user_text.split())
        base_detected_lang = detected_language_code.split('-')[0].lower()

        if (audio_duration_seconds < 1.8 and text_word_count <= 2 and
            base_detected_lang in common_false_positives and
            user_text.lower().strip('.?!') in common_false_positives[base_detected_lang]):
            print(f"⚠️ Filtered false positive '{user_text}' for {user_id}.")
            user_text = "" # Ignore this input

        if not user_text:
             return {"success": False, "error": "Could not understand audio (empty/filtered)", "user_id": user_id}

        # 4. Interact with Gemini
        ai_response_text = "Sorry, I am unable to process that right now."
        # ★★★ CRITICAL CHANGE: Determine the language for TTS based on our rules ★★★
        ai_response_language_for_tts = "en" # Default to English

        if gemini_model and user_id in chat_sessions:
            try:
                chat = chat_sessions[user_id]
                # Send the user's text. Gemini will use the system prompt to decide how to answer.
                gemini_response = chat.send_message(user_text)
                ai_response_text = gemini_response.text

                # Now, set the language for our TTS engine based on the *input* language
                # This logic directly implements the rules from the system prompt
                if base_detected_lang in ["en", "hi"]:
                    ai_response_language_for_tts = base_detected_lang
                else:
                    # For any other language, Gemini was instructed to reply in English.
                    ai_response_language_for_tts = "en"

                print(f"🤖 Gemini ({user_id}, detected_in: {detected_language_code}, responding_as_lang_for_tts: {ai_response_language_for_tts}): \"{ai_response_text}\"")

            except Exception as e_gemini:
                print(f"❌ Gemini API error for {user_id}: {e_gemini}")
                ai_response_text = "I encountered an issue with the AI model."
        elif not gemini_model: # Fallback if Gemini isn't configured
            ai_response_text = get_fallback_response(user_text.lower(), detected_language_code)
            ai_response_language_for_tts = detected_language_code


        # 5. Synthesize Speech with gTTS
        response_audio_base64 = None
        tts_synthesis_error = None
        audio_mime_type = None

        if GTTS_AVAILABLE and ai_response_text:
            # Use the language code we determined above for synthesis
            audio_bytes, mime_type = synthesize_speech_with_gtts(ai_response_text, ai_response_language_for_tts)
            if audio_bytes:
                response_audio_base64 = f"data:{mime_type};base64," + base64.b64encode(audio_bytes).decode('utf-8')
                audio_mime_type = mime_type
                print(f"🎤 gTTS synthesized audio for '{ai_response_language_for_tts}'. Type: {mime_type}, Length: {len(audio_bytes)} bytes")
            else:
                tts_synthesis_error = f"gTTS synthesis failed for language: {ai_response_language_for_tts}"
                print(f"⚠️ {tts_synthesis_error}")

        # 6. Prepare and return the final payload
        return_payload = {
            "success": True, "transcription": user_text, "detected_language": detected_language_code,
            "response_text": ai_response_text,
            "response_language": ai_response_language_for_tts, # Send the correct language to the client
            "user_id": user_id
        }
        if response_audio_base64:
            return_payload["response_audio_base64"] = response_audio_base64
            return_payload["audio_mime_type"] = audio_mime_type
        if tts_synthesis_error and not response_audio_base64:
            return_payload["tts_error"] = tts_synthesis_error

        return return_payload

    except Exception as e:
        print(f"❌ Unexpected error in process_audio_with_gemini for {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": "Internal server error during audio processing.", "user_id": user_id}

def get_fallback_response(text_query, lang_code="en"):
    base_lang = lang_code.split('-')[0].lower()
    if base_lang == "hi": return "क्षमा करें, मैं अभी आपकी मदद नहीं कर सकता।"
    return "I'm sorry, I can't help with that right now."

# --- SocketIO Event Handlers ---
@socketio_app.on('connect')
def on_connect(auth=None): # CORRECTED SIGNATURE for modern flask-socketio
    user_id = request.sid
    print(f"✅ User connected: {user_id}")
    if gemini_model:
        chat_sessions[user_id] = gemini_model.start_chat(history=[])
        print(f"💬 New Gemini chat session created for {user_id}")
    # Use emit imported from flask_socketio
    emit('connection_status', {'status': 'connected', 'user_id': user_id, 'model_ready': bool(gemini_model)})

@socketio_app.on('disconnect')
def on_disconnect():
    user_id = request.sid
    print(f"❌ User disconnected: {user_id}")
    if user_id in chat_sessions:
        del chat_sessions[user_id]
        print(f"🗑️ Chat session removed for {user_id}")

@socketio_app.on('audio_data')
def handle_audio_data(data):
    user_id = request.sid
    audio_data_base64 = data.get('audio')

    if not audio_data_base64:
        emit('audio_response', {'success': False, 'error': 'No audio data received'})
        return

    if not whisper_model:
        emit('audio_response', {'success': False, 'error': 'Speech recognition service is not available.'})
        return

    # Process in a background thread to keep the server responsive
    def process_audio_thread_target():
        result = process_audio_with_gemini(audio_data_base64, user_id)
        socketio_app.emit('audio_response', result, room=user_id)

    thread = threading.Thread(target=process_audio_thread_target)
    thread.daemon = True
    thread.start()

@socketio_app.on('clear_conversation')
def handle_clear_conversation():
    user_id = request.sid
    if gemini_model and user_id in chat_sessions:
        # Re-initialize the chat session to clear history
        chat_sessions[user_id] = gemini_model.start_chat(history=[])
        print(f"🔄 Conversation cleared for {user_id}")
        emit('conversation_cleared', {'status': 'success', 'message': 'Conversation history cleared.'})
    else:
        emit('conversation_cleared', {'status': 'failure', 'message': 'Could not clear conversation.'})

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting Real-time Voice AI Server...")
    if not GEMINI_API_KEY: print("🔴 CRITICAL: GEMINI_API_KEY not set.")
    elif not gemini_model: print("🟠 WARNING: Gemini model init FAILED.")
    if not whisper_model: print("🔴 CRITICAL: Whisper STT model FAILED to load.")

    if not GTTS_AVAILABLE:
        print("🔴 gTTS library is not installed. Server-side TTS is disabled.")
    else:
        print("✅ gTTS is configured for server-side synthesis.")

    # use_reloader=False is important for avoiding issues with threaded/async modes
    socketio_app.run(app, host="0.0.0.0", port=8116, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)