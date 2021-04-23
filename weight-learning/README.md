
# Addition of bert cosine similarity as a new feature to Weber et al.
## Results and Comparison

### Average on 10 iterations
#### Weber et al.
* Average Accuracy: 0.647
#### Our work
* Average Accuracy: 0.689

------------------------------------------------------------------------------------------------------------------------------
# Information from Weber et al.
## Scripts for Model Accuracy Testing

Scripts load data from Excel file, trained a CBR model for prediction and compute for model accuracy.

### Reuqirements

* Python3 and relative libraries.
* Data File (Excel).

### Hyper Parameters

Hyper Parameters can be modified to use different running setting.

#### File Path

Variable `file_path` will be used as the path of load Excel file to be loaded.

If the Excel file is at the same directory as script [run.py](run.py), `file_path` can be set as `file_path = '<file name>.xlsx'`.

The relative path of script [run.py](run.py) can also been used. For example `file_path = '../data/<file name>.xlsx'`

Absolute path is also acceptable. For example, `file_path = '/project/<file name>.xlsx'`

#### Column Names

Variable `cols` should provide the columns script going to use to load data. It also works as a key to get the similarity function for that feature.

Note that the Y value should always be set as the last element for `cols`.

Just comment out the names if there are some features want to be ignored. For example:

```python
cols = [
    'W2V',
    'D2V',
    #'surface similarity',
    'paper sizein Bytes',
    'publication type pair; 1 same type;0 different types',
    #'in which percentile (1 to 8) the citation occurs',
    'final decision: 1 is B, 0 is S'
]
```

#### Max Iteration for Optimization

`itermax` provieds max iteration will be run for optimization. The default value is `50`.

#### k Cases for Prediction

The CBR model will use k most similar cases for prediction. For example, if `k = 3`, then it will use most common result from 3 most similar cases.

### New Features

If there are new features addded or the same feature with new column name, the variable `fns` in function `init(data, cols)` should be modified.

`fns` is a dictionary map the col name with the similarity function.

The similarity function should use the following standard:

* Takes 2 input argument and return their similarity.
* Return 0 if the inputs are same.

### Run

To run the program, use the following command

```bash
python run.py
```
