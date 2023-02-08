# Project for Computational Learning at Unifi
## Classifying audio

**Dependencies:**
* torch
* numpy
* matplotlib
* dagshub
* librosa

Full workflow, i.e. preprocess and load data, train, validate and assess model:
```
python run.py full_workflow
```

Only preprocess data:
(In first run set setup_dagshub=True in setup_data())
```
python setup_data.py
```

Only perform network workflow, i.e. train, validate and assess model
```
python run.py main
```

Only train model:
After building and performing model validation
```
model.Model.train_model()
```

Only 

Code has to be run in directory ```cl_unifi/``` - otherwise file paths are broken!