from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.websocket import RecognizeCallback, AudioSource

from nltk.tokenize import TreebankWordTokenizer

class MyRecognizeCallback(RecognizeCallback):
    transcript = ""

    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_data(self, data):
        for result in data["results"]:
            sentence = result["alternatives"][0]["transcript"]
            sentence = sentence.replace('%HESITATION', '')
            self.transcript += sentence

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

def speech_to_text(file_path):
    # fixme: remove key for security
    # authenticator = IAMAuthenticator('{Your key}')
    authenticator = IAMAuthenticator('PB3kuAttBgiUuoZrdv8ZbKL13j4t7-7wl99VyB5JNUn3')
    
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )

    speech_to_text.set_service_url('https://api.kr-seo.speech-to-text.watson.cloud.ibm.com/instances/c2523ff6-bb4f-4d41-9134-a2327a107b75/v1/recognize')

    myRecognizeCallback = MyRecognizeCallback()
    
    with open(file_path, 'rb') as audio_file:
        audio_source = AudioSource(audio_file)
        speech_to_text.recognize_using_websocket(
            audio=audio_source,
            content_type='audio/wav',
            recognize_callback=myRecognizeCallback,
            model='en-US_BroadbandModel',
            max_alternatives=1
        )
        return myRecognizeCallback.transcript

def tokenize(transcript):
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(transcript)
    return words

def make_input_for_model(file_path):
    transcript = speech_to_text(audio_file_path)
    result = tokenize(transcript)
    return result