To run this code, you must have 
1. A folder named as "code" containing 'main.py'

2. Another folder named as "txt files named for studies"(which you can change name, but you must 
assign it to the variable FILEFOLDER in the very beginning in main.py)
All txt files should be in this folder, files under subfolders will be preprocessed as well.

3. folder "code" and folder "txt files named for studies" should be under same parent folder. 
So the folder structure should looks like:
--------------------------------------------------------------
Preprocessed->code, txt files named for studies
code -> main.py, README.txt
txt files named for studies->query, C1 1.txt,C1 2.txt... ,
query->Q1.txt, Q2.txt...
--------------------------------------------------------------

The program won't go through files in code folder.

The program will generate three folders: NE, NE_Hyphen, NE_Hyphen_PM
(That represents files in this folder has already removed Non English, Hyphens, paragraph mark)
NE_Hyphen_PM is the final folder with files we want, I kept the others to check if unexcepted errors
generated in each step.