"""
google web speech api
실시간 마이크 STT

초기 세팅
Visual Studio 설치하고 C++ 패키지 설치
pip install wheel
pip install pipwin
pipwin install pyaudio
"""

import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    audio = r.listen(source)

# 구글 웹 음성 API로 인식하기 (하루에 제한 50회)
try:
    print(r.recognize_google(audio, language='en', show_all=False))
except Exception as e:
    print(str)