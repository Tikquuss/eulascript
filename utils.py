import PyPDF2
import docx
import ntpath
import wget
import os
import shutil
import numpy as np

def path_leaf(path):
    # source : https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def is_supported(file_name : str):
    supported_extension = [".txt", ".md", ".pdf", ".doc", ".docx"]
    _, extension = os.path.splitext(file_name) 
    if extension in supported_extension :
        return True
    else :
        return False

def text_is_valid(text : str):
    return text.strip().rstrip().replace("\n", "").replace("\r", "").replace("\t", "") # != ""

def get_content(document):
    _, extension = os.path.splitext(document)
    
    if extension == ".pdf" :
        pdfReader = PyPDF2.PdfFileReader(open(document, "rb"))
        content = [pdfReader.getPage(page).extractText() for page in range(pdfReader.numPages)]
        content = [text for text in content if text_is_valid(text = text)]
    
    elif extension in [".doc", ".docx"] :
        doc = docx.Document(open(document, "rb"))
        content = doc.paragraphs
        content = [para.text for para in content if text_is_valid(text = para.text)][:3]
      
    else :
        content = open(document, "r").read()
        content = content if text_is_valid(text = content) else ""
        content = content.split("\n")

    return content

def get_ktrain_predict_method(ktrain_predictor):
    def predict_method(eula):
        if type(eula) == str :
            eula = [eula]
        else :
            assert type(eula) == list

        predictor = ktrain_predictor.predict_proba
        output = []
        for text in eula :
            y = predictor(text)
            if type(y) == np.ndarray :
                #output.append({"acceptability" : float(y[1]), "unacceptability" : float(y[0])})
                output.append(y[1])
            else :
                #output.append({"acceptability" : int(y), "unacceptability" : int(1-y)})
                output.append(y)
                
        return output
    
    return predict_method

def download(output_path, to_load, base_url = "", free_after_download_and_load = False):
          
    cache_path = output_path
    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)

    for model_name, files in to_load.items():
            
        model_path = os.path.join(cache_path, model_name)
            
        if not os.path.isdir(model_path):
            os.mkdir(model_path) 
            
        for file_name in files :
            file_path = os.path.join(model_path, file_name)
            file_url = os.path.join(base_url, model_name, file_name).replace("\\", "/")
                
            if not os.path.isfile(file_path):
                wget.download(
                    file_url, file_path
                )

        if free_after_download_and_load :
            try:
                shutil.rmtree(model_path)
            except OSError:
                pass                        