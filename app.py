from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from spell import analyze
from chat import PronunciationBot

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
bot = PronunciationBot()

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze_pronunciation', methods=['POST'])
def analyze_pronunciation():
    """
    Endpoint to analyze pronunciation
    Expects: 
    - audio_file: Audio file (MP3/WAV)
    - text: Text to analyze against
    """
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio_file']
        text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'Invalid file type. Only MP3 and WAV files are allowed'}), 400

        # Save the file temporarily
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)

        # Analyze pronunciation
        result = analyze(text, filepath)

        # Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        # Clean up in case of error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    Endpoint for chatbot interaction
    Expects:
    - message: User's message
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message']
        
        # Process chat using the bot
        sentence = bot.tokenize(user_message)
        X = bot.bag_of_words(sentence, bot.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(bot.device)

        output = bot.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = bot.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        response = {'bot_name': bot.bot_name}
        
        if prob.item() > 0.75:
            for intent in bot.intents['intents']:
                if tag == intent["tag"]:
                    response['message'] = random.choice(intent['responses'])
                    if tag in ["pronunciation", "check_similarity"]:
                        response['action'] = 'request_audio'
                    break
        else:
            response['message'] = "I don't understand. Could you rephrase that?"

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)