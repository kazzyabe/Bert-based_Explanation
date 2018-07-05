The folder structure should be as following:

Jaccard->code,files
code->similarity.py, README.txt, compare_list.txt, stats_Jaccard.txt
files->C0101.txt, ..., Q01-1000.txt, ...

You should NOT change the structure to make sure the program can work properly

---------------------
compare_list.txt: 
---------------------
This file describes how to compare two files.
Make sure these candidate files are all in files folder

It should look like this:
C0101 , Q01-4000
C0102 , Q01-5000
C0103 , Q01-1000
C0104 , Q01-7000
....

---------------------
stats_Jaccard.txt:
---------------------
This file is the output showing jaccard similarity between query doc and cite doc 