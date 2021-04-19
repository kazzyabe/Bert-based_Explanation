# Bert based explanation
This is a project is an extension of the paper "Investigating Textual Case-Based XAI". The description of the paper is in section iccbr2018.

The code and data for bert based explanation is under the directory "bert-based_explanation"

## iccbr2018
This is repo includes code and data used in studies in paper "Investigating Textual Case-Based XAI"

The raw data from ACL used is available through this link: https://drive.google.com/open?id=1FxaDxcw_g48swVlhftRN_XPnMHC0ZH3V 
The openC corpus is linked in https://github.com/allenai/citeomatic/blob/master/README.md 

Folder "10 whole query articles txt" has preprocessed text of the 10 papers selected as queries. Analogously, folder "100 cited articles" has the 100 papers selected as cited. Preprocessing code is in the folder named "preprocessing".
Labels assigned by our colleagues are in the cases. Label 1 refers to classification background (B), 0 to substantiation (S).

Folder "Extended attributes" includes code used to learn and reuse language models, compute cosine and WMV. Code files use extensions '.py', and '.txt' show results. Files that include '-tf-' have code adapted from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py . Files named with 'GS' use the Gensim library https://radimrehurek.com/gensim/ .

Code used to learn weights with gradient descent (GD) and to estimate average accuracy, which is used to compute error in GD is in folder "weight-learning-LOOCV".
Results with variations in number of neighbors and effect of removing one feature at a time are in folder "raw-results".
Thanks for checking out our work!
Please let us know what may be still missing...
