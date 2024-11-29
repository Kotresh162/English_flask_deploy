import os
import subprocess
import tempfile
import wave
import requests
from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import speech_recognition as sr

# Initialize Flask app
app = Flask(__name__)

# Initialize InferenceClient with your Hugging Face API key
client = InferenceClient(api_key="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")

# Function to convert a Google Drive view link into a direct download link
def convert_drive_url_to_direct_link(url):
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    except IndexError:
        print("Invalid Google Drive URL format.")
        return None

# Function to download a video file from a URL
def download_video(url):
    try:
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        print(f"Downloading video from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(temp_video.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Video downloaded to temporary file: {temp_video.name}")
        return temp_video.name
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

# Function to extract audio from a video file and return its path
def extract_audio(video_path):
    try:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        command = f'ffmpeg -i "{video_path}" -ab 160k -ar 44100 -vn "{temp_audio.name}" -y'
        subprocess.call(command, shell=True)
        print(f"Audio extracted to temporary file: {temp_audio.name}")
        return temp_audio.name
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

# Function to get audio duration
def get_audio_duration(audio_path):
    try:
        with wave.open(audio_path, 'rb') as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            duration = frames / float(rate)
            return duration
    except wave.Error:
        print("Error: Invalid audio file.")
        return 0

# Function to transcribe audio in chunks
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio_duration = get_audio_duration(audio_path)
    if audio_duration == 0:
        return "[Error: Invalid audio file]"

    chunk_duration = 30  # Process in 30-second chunks
    transcript = ""

    try:
        with sr.AudioFile(audio_path) as source:
            for start_time in range(0, int(audio_duration), chunk_duration):
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                remaining_duration = min(chunk_duration, audio_duration - start_time)
                audio_chunk = recognizer.record(source, duration=remaining_duration)
                try:
                    text = recognizer.recognize_google(audio_chunk)
                    print(f"Chunk {start_time // chunk_duration + 1}: {text}")
                    transcript += text + " "
                except sr.UnknownValueError:
                    print(f"Unrecognized speech in chunk {start_time // chunk_duration + 1}")
                    transcript += "[Unrecognized speech in this chunk] "
                except sr.RequestError as e:
                    print(f"Request error in chunk {start_time // chunk_duration + 1}: {e}")
                    transcript += "[Error processing this chunk] "
    except Exception as e:
        print(f"Error processing the audio file: {e}")

    return transcript

# Function to summarize the transcript using Hugging Face InferenceClient
def summarize_transcript(transcript):
    messages = [
        {
            "role": "user",
            "content": transcript
        }
    ]

    completion = client.chat.completions.create(
        model="Qwen/QwQ-32B-Preview", 
        messages=messages, 
        max_tokens=500
    )

    summary = completion.choices[0].message['content']
    return summary

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.get_json()
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "Video URL is required"}), 400

    # Convert Google Drive URL if needed
    if "drive.google.com" in video_url:
        video_url = convert_drive_url_to_direct_link(video_url)
        if not video_url:
            return jsonify({"error": "Invalid Google Drive URL"}), 400

    # Step 1: Download video from URL
    video_path = download_video(video_url)
    if not video_path:
        return jsonify({"error": "Error downloading video"}), 500

    # Step 2: Extract audio from the video
    audio_path = extract_audio(video_path)
    if not audio_path:
        return jsonify({"error": "Error extracting audio from video"}), 500

    # Step 3: Transcribe the audio
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return jsonify({"error": "Error transcribing audio"}), 500

    # Step 4: Summarize the transcript using Hugging Face InferenceClient
    summary = summarize_transcript(transcript)

    # Cleanup temporary files
    try:
        os.remove(video_path)
        os.remove(audio_path)
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")

    return jsonify({"transcript": transcript, "summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
