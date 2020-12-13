# Packages
import nltk
nltk.download('punkt')
from nltk import word_tokenize
#import useless_words
from nltk.stem import PorterStemmer
import time
from shutil import copyfile
from difflib import SequenceMatcher
from selenium import webdriver
import shutil, sys  # for the copyfile

import shlex
import subprocess
import os
#from os import startfile


# CONSTANTS     
SIGN_PATH = "C:\\Users\\고준원\\Downloads\\Text_to_sign_language" ## Your local work folder
# SIGN_PATH = os.getcwd()
DOWN_PATH = "C:\\Users\\고준원\\Downloads" ## Your local downloads folder
# DOWN_PATH = os.pardir + '/Results'

DOWNLOAD_WAIT = 7
SIMILIARITY_RATIO = 0.9
uselesswords = ['is', 'the', 'are', 'am', 'a', 'it', 'was', 'were', 'an', ',', '.', '?', '!']

# Get words
def download_word_sign(word):
    # Download Firefox browser
    # os.path.abspath(os.getcwd())
    browser = webdriver.Chrome(os.path.abspath(os.getcwd()) + '/chromedriver') ## Downloaded Chrome driver AT YOUR WORK FOLDER
    browser.get("http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi")
    first_letter = word[0]
    letters = browser.find_elements_by_xpath('//a[@class="sideNavBarUnselectedText"]')
    for letter in letters:
        if first_letter == str(letter.text).strip().lower():
            letter.click()
            time.sleep(2)
            break

    # Show drop down menu ( Spinner )
    spinner = browser.find_elements_by_xpath("//option")
    best_score = -1.
    closest_word_item = None
    for item in spinner:
        item_text = item.text
        # if stem == str(item_text).lower()[:len(stem)]:
        s = similar(word, str(item_text).lower())
        if s > best_score:
            best_score = s
            closest_word_item = item
            print(word, " ", str(item_text).lower())
            print("Score: " + str(s))        
    if best_score < SIMILIARITY_RATIO:
        print(word + " not found in dictionary")
        ### HERE
        ## 1. Alphabet video

        ## 2. Alphabet picture
        browser.close()
        return ("_" + word.lower())
    real_name = str(closest_word_item.text).lower()

    print("Downloading " + real_name + "...")
    closest_word_item.click()
    time.sleep(DOWNLOAD_WAIT)
    in_path = DOWN_PATH + "/" + real_name + ".swf" ## your local downloads
    out_path = SIGN_PATH + "/" + real_name + ".mp4"
    convert_file_format(in_path, out_path)
    browser.close()
    return real_name

def convert_file_format(in_path, out_path):
    # Converts .swf filw to .mp4 file and saves new file at out_path
    from ffmpy import FFmpeg
    ## please download ffmpeg AT YOUR WORK FOLDER

    ff = FFmpeg(executable=SIGN_PATH + "/ffmpeg/bin/ffmpeg.exe",
    inputs = {in_path: None},
    outputs = {out_path: None})
    ff = FFmpeg(inputs = {in_path: None}, outputs= {out_path: None})
    ff.run()

def get_words_in_database():
    import os
    vids = os.listdir(SIGN_PATH)
    vid_names = [v[:-4] for v in vids]
    return vid_names

def process_text(text):
        ##print("text here ", text)
    
    # Split sentence into words
    words = word_tokenize(text)
        ##print("token words here", words)

    # Remove all meaningless words
        ##print("uselesswords : ", uselesswords)
    usefull_words = [str(w).lower() for w in words if w.lower() not in uselesswords]
        ##print("useless deleted", usefull_words)
    
    return usefull_words

def merge_signs_first(words):
    # Write a text file containing all the paths to each video
    with open(SIGN_PATH + "/vidlist.txt", 'w') as f:
        for w in words:
            global PLAY_VIDEO
            if(w != None):
                f.write("file '" + SIGN_PATH + "/" + w + ".mp4'\n")
                PLAY_VIDEO = True
            elif(w[0] == '_'):
                for i in range(1, len(w)):
                    f.write("file '" + SIGN_PATH + "/" + w[i] + ".mp4'\n")
                    PLAY_VIDEO = True
            else:
                PLAY_VIDEO = False

    # Splits the command into pieces in order to feed the command line
    # command not working, we don't have a output
    # Command looks like : ffmpeg -f concat -i videolist.txt -c copy output.mp4
    command = "ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output.mp4 -y"
    
    args = shlex.split(command)
    
    #print("@@@ shlex splited into\n", args, "\n")

    env = {'PATH': SIGN_PATH + '/ffmpeg/bin'}
    # env = {'PATH': SIGN_PATH + '\\ffmpeg\\bin' + SIGN_PATH + '\\SignToSignLang'} # ffmpeg 인식안됨
    # env = os.environ # ffmpeg 인식안됨
    # env = {'PATH': os.getenv('PATH')} # ffmpeg 인식안됨
    process = subprocess.Popen(args, shell=True, env=env)

    process.wait()

    ## copyfile(src, dst)
    shutil.copyfile(SIGN_PATH + "/output.mp4", SIGN_PATH + "/outputs.mp4")

def merge_signs(words):
    # Write a text file containing all the paths to each video
    with open(SIGN_PATH + "/vidlist.txt", 'w') as f:
        for w in words:
            global PLAY_VIDEO
            if(w != None):
                f.write("file '" + SIGN_PATH + "/" + w + ".mp4'\n")
                PLAY_VIDEO = True
            elif(w[0] == '_'):
                for i in range(1, len(w)):
                    f.write("file '" + SIGN_PATH + "/" + w[i] + ".mp4'\n")
                    PLAY_VIDEO = True
            else:
                PLAY_VIDEO = False

    # Splits the command into pieces in order to feed the command line
    # command not working, we don't have a output
    # Command looks like : ffmpeg -f concat -i videolist.txt -c copy output.mp4
    command = "ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output.mp4 -y"
    
    args = shlex.split(command)
    
    #print("@@@ shlex splited into\n", args, "\n")

    env = {'PATH': SIGN_PATH + '/ffmpeg/bin'}
    # env = {'PATH': SIGN_PATH + '\\ffmpeg\\bin' + SIGN_PATH + '\\SignToSignLang'} # ffmpeg 인식안됨
    # env = os.environ # ffmpeg 인식안됨
    # env = {'PATH': os.getenv('PATH')} # ffmpeg 인식안됨
    process = subprocess.Popen(args, shell=True, env=env)

    process.wait()
    
    # Now, concat output next to out(which will be a video for whole sentences)
    command = "ffmpeg -f concat -safe 0 -i concatoutput.txt -c copy outputs_temp.mp4 -y"
    args = shlex.split(command)
    env = {'PATH': SIGN_PATH + '/ffmpeg/bin'}
    process = subprocess.Popen(args, shell=True, env=env)
    process.wait()

    ## copyfile(src, dst)
    shutil.copyfile(SIGN_PATH + "/outputs_temp.mp4", SIGN_PATH + "/outputs.mp4")

def in_database(w):
    db_list = get_words_in_database()
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    s = ps.stem(w)
    for word in db_list:
        if s == word[:len(s)]:
            return True
    return False

def similar(a, b):
    # Returns a decimal representing the similiarity between the two strings.
    return SequenceMatcher(None, a, b).ratio()

def find_in_db(w):
    best_score = -1.
    best_vid_name = None
    for v in get_words_in_database():
        s = similar(w, v)
        if best_score < s:
            best_score =  s
            best_vid_name = v
    if best_score > SIMILIARITY_RATIO:
        return best_vid_name

def convert_to_video():
    ## MAIN STARTS
    ## text 받는 부분부터 for loop 으로 txt 파일을 읽어오면 전체 텍스트에 대해 실행 가능.
    # Get the text which is going to become ASL
    # f = open(SIGN_PATH + "\\disfluency_remover_result.txt", 'r')
    text_path = os.pardir + '/Results/disfluency_remover_result.txt'
    print("text_path")
    print(text_path)
    f = open(text_path, 'r')
    input_lines = f.readlines()
    f.close()

    print("### Whole text is ", input_lines)

    i = 0

    for text in input_lines:
        print("\n### Now we are translating [ " + text +" ]")

        # Process text
        words = process_text(text)

        print("words here", words)

        # Download words that have not been downloaded in previous sessions.
        print("\n### Check if we already have video in DB")
        real_words = []
        for w in words:
            real_name = find_in_db(w)
            if real_name:
                print("  ->" + w + " is already in db as " + real_name)
                real_words.append(real_name)
            else:
                real_words.append(download_word_sign(w))
                if real_words[-1][0] == '_':
                    word_not_exist = real_words.pop()
                    for i in range(1, len(word_not_exist)):
                        real_words.append(word_not_exist[i])

        words = real_words
        print("### DB check done\n")

        # print("@@@ What words video gonna merge?\n", words, "\n")

        # Concatenate videos and save output video to folder
        if i == 0:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            merge_signs_first(words)
            i = 1
        else:
            print("####################################")
            merge_signs(words)

    # Finally make out.mp4
    shutil.copyfile(SIGN_PATH + "/outputs.mp4", SIGN_PATH + "/out.mp4")

    # Play the video
    if(PLAY_VIDEO == True):
        os.startfile(SIGN_PATH + "/out.mp4")
