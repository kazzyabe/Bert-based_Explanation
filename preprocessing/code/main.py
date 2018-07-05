import threading
import re
import langdetect
from langdetect import detect
import os


FILEFOLDER="txt files named for studies"

# Aalk txt files except in exclude_dir
def txt_walk(dir):
    file_list=[]
    exclude_dir =dir+"\\code"
    for root, dirs, files in os.walk(dir):
        dirs[:] = [d for d in dirs if d not in exclude_dir]
        for file in files:
            if file.endswith(".txt"):
                file_list.append(os.path.join(root, file))
    return file_list

def removeNE(file_in):
    file_out=file_in.replace(FILEFOLDER,"NE")
    #make dir if not exist
    os.makedirs(os.path.dirname(file_out), exist_ok=True)

    with open(file_in,'r') as infile, open(file_out,'w') as outfile:
        for line in infile:
            try:
                if detect(line)=='en':
                    outfile.write(line)
            except langdetect.lang_detect_exception.LangDetectException:
                pass
    infile.close()
    outfile.close()

def removeHyphen(file_in):
    reg_hyphen = re.compile(r'-$')  # Only specifies the hyphen in the end of line
    output_file_path = file_in.replace("NE","NE_Hyphen");

    # make dir if not exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    file_out = open(output_file_path , "w+", encoding='utf8')  # open file to write to

    # Load file
    left_from_last_line = ""
    for line in open(file_in, 'r'):#, encoding='utf8'):
        line = left_from_last_line + line
        words = line.split()
        outputline = ""
        for word in words:
            # If hyphen detected in the end, save it to head of the next line
            if reg_hyphen.search(word):
                # print(word)
                left_from_last_line = re.sub(reg_hyphen, '', word)  # remove - in the end
            else:
                outputline = outputline + " " + word
                left_from_last_line = ""
        file_out.write(outputline + "\n")
    file_out.close()


def removeParagraphMark(file_in):
    output_file_path = file_in.replace("NE_Hyphen", "NE_Hyphen_PM");
    # make dir if not exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    file_out = open(output_file_path, "w+",encoding='utf8')  # open file to write to
    for line in open(file_in, 'r', encoding='utf8'):
        line = line.replace('\r', '').replace('\n', '')
        file_out.write(line+" ")
    file_out.close()

def main():
    ''' All codes in current folder($code = code)
        All orginal files to be manupulated are in ($file = txt files named for studies)
        $code and $ file in the same parent folder
    '''
    file_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    file_list=txt_walk(file_directory)

    print("---------------txt files found in directory:---------------")
    print('[%s]' % '\n '.join(map(str, file_list)))

    # Remove non-English characters
    print("---------------Removing non-English characters---------------")

    for file in file_list:
        thread = threading.Thread(target=removeNE(file))
        thread.start()

    # Remove hyphens
    print("---------------Removing hyphens in the end of line---------------")
    file_directory = file_directory.replace(FILEFOLDER, "NE")
    file_list = txt_walk(file_directory)
    for file in file_list:
        thread = threading.Thread(target=removeHyphen(file))  # Define a thread
        thread.start()  # start thread

    # Remove paragraph marks
    print("---------------Removing paragraph marks in the end of line---------------")
    file_directory = file_directory.replace("NE", "NE_Hyphen")
    file_list = txt_walk(file_directory)
    for file in file_list:
        thread = threading.Thread(target=removeParagraphMark(file))  # Define a thread
        thread.start()  # start thread

    '''
    file_directory = file_directory.replace("NE_Hyphen", "NE_Hyphen_PM")
    file_list = txt_walk(file_directory)
    '''


''' 

  
'''

if __name__ == '__main__':
    main()









