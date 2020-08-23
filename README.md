# 1 - Cloning the repository
```
git clone https://github.com/Tikquuss/eulascript
```

# 2 - Installing the dependencies

* [PyPDF2](https://pypi.org/project/PyPDF2/) and [PyMuPDF](https://pypi.org/project/PyMuPDF/): for reading pdf files
* [python-docx](https://pypi.org/project/python-docx/) : for reading docx files 
* [wget](https://pypi.org/project/wget/) : for model downloading 
* [pandas](https://pandas.pydata.org/) : to write the result in csv files
* [validators](https://pypi.org/project/validators/) : to check the validity of the urls
* [ktrain](https://github.com/Tikquuss/ktrain) : for loading models. It is a duplication of [amaiya/ktrain](https://github.com/amaiya/ktrain) modified to install `tensorflow-cpu` (instead of `tensorflow-2.1.0-cp36-cp36m-manylinux2010_x86_64.whl`) and `tqdm>=4.29.1`.
```
pip install -r eulascript/requirements.txt
```

# 3 - Try

* **model_folder** : directory (or url of the directory) where the model is located (must contain the following three files: `tf_model.preproc`, `config.json` and `tf_model.h5`). In the case of a url the three previous files are downloaded automatically.
* **output_dir** : folder in which the csv file(s) containing the results (in the format: `clause, label, probability`) will be stored (the name of the created file starts with the name, without extension, of the original file containing the license, followed optionally by a number to avoid file collisions)
* **path_to_eula** : comma-separated list of documents (`txt, md, pdf and docx`) containing the licenses to be analyzed
* **logistic_regression** :  this parameter can be provided at the expense of **model_folder** in order to use one of the pre-trained logistic regression models (must be obligatorily made from these three models: bag_of_word, tf_idf or bert). This parameter is ignored if it is passed at the same time as **model_folder**

```
model_folder=my/model_dir_or_url
output_dir=my/output_folder
path_to_eula=my/eula.txt,my/eula.md,my/eula.pdf,my/eula.docx

python eulascript/eula.py --model_folder $model_folder --path_to_eula $path_to_eula --output_dir $output_dir
```

```
logistic_regression=bag_of_word
output_dir=my/output_folder
path_to_eula=my/eula.txt,my/eula.md,my/eula.pdf,my/eula.docx

python eulascript/eula.py --logistic_regression $logistic_regression --path_to_eula $path_to_eula --output_dir $output_dir
```


**Note**: 
* the [samples](samples) folder contains some user licenses and a [notebook](samples/notebook.ipynb) illustrating all. 
* The associated web application is available [here](https://eulapp.herokuapp.com/). 

