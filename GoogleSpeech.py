"""
나한테 json 파일을 받는다  

cmd에 친다  
set GOOGLE_APPLICATION_CREDENTIALS=cs5642020f-1a22eb38664.json

google cloud sdk shell를 다운받는다  
gcloud init
프로젝트 설정

아나콘다 켜서  
pip install --upgrade google-cloud-storage  
pip install google-cloud-speech  
gcloud auth activate-service-account --key-file="cs5642020f-1a22eb386634.json"  
끝!
"""
import io
import os
from google.cloud import speech

print('1')
# Instantiates a client
client = speech.SpeechClient()

# The name of the audio file to transcribe
file_name = "hi.wav"
print('2')
# Loads the audio into memory
with io.open(file_name, "rb") as audio_file:
    content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
print('3')
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
)

print('4')
# Detects speech in the audio file
response = client.recognize(config=config, audio=audio)

print('5')
for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))
    
print('6')