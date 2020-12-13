import os
from Speech_to_text import make_input_for_model
from Punctuation_restoration import execute_punctuator
from DisfluencyRemover.main import execute_disfluency_remover
from Text_to_sign_language.TextToSignLanguage import convert_to_video

def main():
    # Speech to text
    file_path = os.getcwd() + '/Resources/audio_sample.wav'
    input = make_input_for_model(file_path)
    print("-----STT Result-----")
    print(input)

    # Punctuation restoration model
    output = execute_punctuator(input)
    file_path_punctuator = os.getcwd() + '/Results/punctuator_result.txt'
    f = open(file_path_punctuator, 'w')
    f.write(output)
    f.close()
    print("-----Punctuator Result-----")
    print(output)

    # Disfluency detection
    os.chdir('DisfluencyRemover')
    execute_disfluency_remover()
    print("-----Disfluency Remover Execution Completed-----")
    
    # Generation of sign language video
    os.chdir(os.pardir + '/Text_to_sign_language')
    convert_to_video()
    print("-----Sign Language Video Generation Completed-----")

if __name__ == '__main__':
    main()
