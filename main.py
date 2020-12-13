import os
from Speech_to_text import make_input_for_model
from Punctuation_restoration import execute_punctuator
from DisfluencyRemover.main import execute_disfluency_remover
from Text_to_sign_language.TextToSignLanguage import convert_to_video

def main():
    # Speech to text
    file_path = os.getcwd() + '/Resources/audio_sample.wav'
    # input = make_input_for_model(file_path)
    input = ['sometimes', 'the', 'objects', 'are', 'pretty', 'I', 'mean', 'it', "'s", 'very', 'difficult', 'to', 'distinguish', 'the', 'object', 'from', 'the', 'background', 'and', 'more', 'critically', 'even', 'the', 'same', 'concept', 'something', 'like', 'a', 'chair', 'can', 'have', 'a', 'different', 'appearances', 'so', 'it', "'s", 'called', 'intra', 'class', 'very', 'rich', 'this', 'is', 'very', 'involved', 'parents', 'are', 'very', 'diverse', 'and', 'if', 'you', 'want', 'to', 'recognize', 'the', 'concept', 'robust', 'against', 'this', 'is', 'diverse', 'city', 'arborist', 'interference', 'you', 'have', 'to', 'have', 'a', 'robust', 'representational', 'image', 'that', 'are', 'robust', 'against', 'which']
    print("-----STT result-----")
    print(input)

    # Punctuation restoration model
    output = execute_punctuator(input)
    file_path_punctuator = os.getcwd() + '/Results/punctuator_result.txt'
    f = open(file_path_punctuator, 'w')
    f.write(output)
    f.close()
    print("-----Punctuator result-----")
    print(output)

    # Disfluency detection
    os.chdir('DisfluencyRemover')
    execute_disfluency_remover()
    print("-----Disfluency Remover execution completed-----")

    # os.path.join(os.path.dirname(DIR), 'path')
    # os.path('/Users/jinheeryu/Desktop/KAIST/2020 Fall/CS470-teamproject/Text_to_sign_language')

    print(os.getcwd())
    # os.chdir('/Users/jinheeryu/Desktop/KAIST/2020 Fall/CS470-teamproject/Text_to_sign_language')
    os.chdir(os.pardir + '/Text_to_sign_language')
    print(os.getcwd())

    # os.chdir('/Text_to_sign_language')
    convert_to_video()
    print("-----Conversion completed-----")

if __name__ == '__main__':
    main()