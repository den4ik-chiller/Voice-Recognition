from flask import Flask, render_template, request, jsonify
import os
import whisper
import speech_recognition as sr
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydub import AudioSegment

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Загрузка предварительно обученной модели BERT для анализа эмоций
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model.eval()

# Map labels to emotions
emotion_mapping = {
    0: 'very negative',
    1: 'negative',
    2: 'neutral',
    3: 'positive',
    4: 'very positive'
}


def analyze_emotion(text):
    # Tokenize the input text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    # Map label to emotion using the emotion_mapping dictionary
    emotion = emotion_mapping.get(predicted_label, 'unknown')

    return emotion


def convert_to_wav(audio_file_path):
    sound = AudioSegment.from_file(audio_file_path)
    wav_file_path = audio_file_path[:-4] + '.wav'  # replace .mp3 with .wav
    sound.export(wav_file_path, format='wav')
    return wav_file_path


model_tr = whisper.load_model("medium")


def recognize_and_analyze(audio_file_path):
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    _, probs = model_tr.detect_language(mel)
    print(f"Язык аудио: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model_tr, mel, options)
    transcribed_text = result.text
    # recognizer = sr.Recognizer()
    #
    # with sr.AudioFile(audio_file_path) as source:
    #     audio_data = recognizer.record(source)
    #
    # text = recognizer.recognize_google(audio_data, language="ru-RU")
    print('Recognized Text:', transcribed_text)

    emotions = analyze_emotion(transcribed_text)
    return {'text': transcribed_text, 'emotion': emotions, 'lang': max(probs, key=probs.get)}


def process_audio(request):
    audio_file = request.files['file']
    print(audio_file.filename)

    if audio_file:
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recording_audio.mp3')
        audio_file.save(audio_file_path)
        print(audio_file)
        print(audio_file_path)

        # Convert audio file to WAV
        # wav_file_path = convert_to_wav(audio_file_path)
        result = recognize_and_analyze(audio_file_path)
        print(result)

        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file format'})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    response = process_audio(request)
    return response


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
