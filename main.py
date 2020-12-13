import os
from Speech_to_text import make_input_for_model
from Punctuation_restoration import execute_punctuator
from DisfluencyRemover.main import execute_disfluency_remover

def main():
    # Speech to text
    file_path = os.getcwd() + '/Resources/audio_sample.wav'
    input = make_input_for_model(file_path)
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


if __name__ == '__main__':
    main()
