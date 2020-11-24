"""
google web speech api
녹음 파일 STT

초기 세팅
pip install SpeechRecognition
"""

import speech_recognition as sr

AUDIO_FILE = "9_2 lecture video.wav"

r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source) 

# 구글 웹 음성 API로 인식하기 (하루에 제한 50회)
try:
    print(r.recognize_google(audio, language='en', show_all=False))
except Exception as e:
    print(str)
