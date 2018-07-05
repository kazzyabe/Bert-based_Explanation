import os
import threading
from prettytable import PrettyTable


def jaccard_with_files(file_x, file_y):
    #check files are in one-line
    i=0
    for line in open(file_x, 'r', encoding="utf8"):
        if i==1:
            print('Please check files are in one-line')
            exit(0)
        x = line
        i+=1
    x=x.lower().split()

    i=0
    for line in open(file_y, 'r', encoding="utf8"):
        if i == 1:
            print('Please check files are in one-line')
            exit(0)
        y = line
    y=y.lower().split()

    z= set(x) & set(y)

    return float(len(z)) / (len(x) + len(y) - len(z))

def main():
    query_directory=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    query_directory=os.path.abspath(query_directory+"\\files")

    compare_list ='compare_list.txt'
    FILE_OUT='stats_Jaccard.txt'

    print("---------------OUTPUT---------------")
    t = PrettyTable(['Query file', 'Cite file', 'similarity'])

    # Get what files to be compared
    with open(compare_list, 'r') as infile:
        for line in infile:
            #print(line)
            value=line.split(',')

            cite_file=os.path.abspath(query_directory+"\\"+value[0].strip()+".txt")
            query_file=os.path.abspath(query_directory+"\\"+value[1].strip()+".txt")

            t.add_row([value[0], value[1], jaccard_with_files(cite_file, query_file)])
    infile.close()


    print (t)

    outfile=open(FILE_OUT,"w+")
    outfile.write(t.get_string())
    outfile.close()

if __name__ == '__main__':
    main()

