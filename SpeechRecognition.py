"""
pip install SpeechRecognition
이건 google web speech api

끝!
"""

import speech_recognition as sr

AUDIO_FILE = "hi.wav"

# audio file을 audio source로 사용합니다
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # 전체 audio file 읽기

# 구글 웹 음성 API로 인식하기 (하루에 제한 50회)
try:
    print("Google Speech Recognition thinks you said : " + r.recognize_google(audio, language='en'))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))