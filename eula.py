import argparse
import os
import csv
import pandas as pd 
import validators
import ktrain
from threading import Thread

from utils import path_leaf, is_supported, get_content, get_ktrain_predict_method, download
from utils import convert_to_clauses

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="eula")

    # main parameters
    parser.add_argument("--model_folder", type=str, default="", 
                        help="folder (link of folder) containing the model : tf_model.preproc, config.json and tf_model.h5")
    parser.add_argument("--path_to_eula", type=str, default="", 
                        help="liste des fichiers (pdf, docx, txt et md) de licence, separé par la virgule : path_to_file1,path_to_file2,...")
    parser.add_argument("--output_dir", type=str, default="", 
                        help="folder in which the results will be stored")
    
    parser.add_argument("--cache_path", type=str, default="cache", 
                        help="folder in which the models will be stored temporarily")

    parser.add_argument("--logistic_regression", type=str, default="", 
                        help="bag_of_word or tf_idf or bert")


    
    return parser
  
def thread_target(eula_file, predict_method) :

    print("============ eula_file : ", eula_file, " ============")
          
    """
    clause_list = get_content(eula_file)
    clause_dic = convert_to_clauses(clause_list)
    """
    _, _, clause_dic, _ = get_content(eula_file)
    
    clause_list = list(clause_dic.values())
    clauses_key = list(clause_dic.keys())

    probabilities = predict_method(clause_list)
    labels = [1 if y >= 0.5 else 0 for y in probabilities]

    file_name = path_leaf(path = eula_file)
    file_name, _ = os.path.splitext(file_name) 

    csv_file = file_name+".csv"
        
    if os.path.isfile(csv_file):
        i = 1
        while os.path.isfile(file_name+'.'+str(i)+".csv"):
            i += 1
        csv_file = file_name+'.'+str(i)+'.csv'

    print("============ csv_file : ", csv_file, " ============")
          
    pd.DataFrame(zip(clauses_key, clause_list, labels, probabilities)).to_csv(csv_file, header= ["clauses_key" ,"clauses", "labels", "probabilities"])


def main(params):

    if params.a :
        predict_method = get_ktrain_predict_method(ktrain_predictor = ktrain.load_predictor(params.model_folder))  
    else :
        from logistic_regression import get_predict_method
        predict_method = get_predict_method(params.logistic_regression)
        
    
    for eula_file in params.eula_files :
      Thread(target = thread_target,  kwargs={"eula_file" : eula_file, "predict_method" : predict_method}).start()

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    #assert params.model_folder
    if params.model_folder.startswith("http:") or params.model_folder.startswith("https:") :
        assert validators.url(params.model_folder)
        url = params.model_folder.split("/")
        model_name = url[-1]
        base_url = '/'.join(url[:len(url)-1])+"/"
        to_load = { 
                model_name : ["tf_model.preproc", "config.json" , "tf_model.h5"]
        }
        download(output_path = params.cache_path, to_load = to_load, base_url = base_url)
        params.model_folder = os.path.join(params.cache_path, model_name)
        params.a = True
    else :
        a = os.path.isdir(params.model_folder)
        b = params.logistic_regression in ["bag_of_word","tf_idf","bert"]
        assert a or b, "model folder path not found"
        if a :
            assert all([os.path.isfile(os.path.join(params.model_folder, f)) for f in ["tf_model.preproc", "config.json" , "tf_model.h5"]])
        params.a = a

    eula_files = params.path_to_eula.split(",")
    assert eula_files
    assert all([os.path.isfile(eula_file) for eula_file in eula_files])
    assert all([is_supported(file_name = eula_file) for eula_file in eula_files])
    
    params.eula_files = eula_files

    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
        
    main(params)