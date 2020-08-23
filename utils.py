import PyPDF2
import docx
import ntpath
import wget
import os
import shutil
import numpy as np
import re
import  fitz

def path_leaf(path : str):
    """
    Returns the name of a file given its path
    https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def is_supported(file_name : str):
    """Verifies if a file is supported by the application"""
    supported_extension = [".txt", ".md", ".pdf", ".doc", ".docx"]
    _, extension = os.path.splitext(file_name) 
    if extension in supported_extension :
        return True
    else :
        return False

def text_is_valid(text : str):
    """Verifies if a clause is treatable/scalable"""
    return text.strip().rstrip().replace("\n", "").replace("\r", "").replace("\t", "") # != ""


def can_be_subsubtitle1(e : str) :

    regex_subsubtitle = '^\w+\.|\(?\w+\)'
    regex_subsubtitle_no_romain = '^\w{,1}\.|\(?\w{,1}\) '
    romain_set = {"i", "v", "x", ".", ")", "("}
    e_strip = e.strip()
    m = re.match(regex_subsubtitle, e_strip)
    if m :
        if set(e).issubset(romain_set) :
            return "romain"
        else :
            if len(e) <= 3 and re.match(regex_subsubtitle_no_romain, e_strip) :
                if ('.' in e_strip or e_strip[1] == ')') and len(e_strip) == 2 :
                    return "non_romain"


def extract_subsubtitle1(dico) :

    dico_temp = {}
    """
    for k, v in dico.copy().items():
        current = len(v) 
        for index, e in enumerate(reversed(v)) :
            if can_be_subsubtitle1(e) :
                dico_temp[k + '-'+ e] = v[index:current]
                del dico[k][index : current]
                current = index

    dico.update(dico_temp)
    """

    return dico 

def convert_to_clauses(content : list):
    regex_title = '[0-9]+\.\w?'
    regex_subtitle = '[0-9]+'
    fake_string = '__fake__'
    dico = {}
    current = '0.'
    dico[current] = []

    a = " ".join(content)
    b = a.split("\n \n")

    for c in b :
        if text_is_valid(text = c) :
            if re.search(regex_title, c) :  
                v = c.split(".")
                
                k = v[0]
                v = v[1]  

                m = re.match(regex_subtitle, v.strip())
                if m :
                    span = m.span()
                    current = k + '.' + v.strip()[span[0]:span[1]] 
                    v = v[span[1]:]
                else :
                    current = k + '.' 

                if current in dico.keys():
                    i = 1
                    while k+fake_string+str(i) in dico.keys():
                        i += 1
                    current = k+fake_string+str(i)

                try :
                  dico[current].append(v)
                except KeyError :
                  dico[current] = [v]
            else:  
                dico[current].append(c)


    dico = extract_subsubtitle1(dico)

    dico2 = {}
    for k, v in dico.items():
      dico2[k] = ' '.join(v).replace("\n", "")
      
    return dico2


import re
import  fitz

def text_is_valid(text : str):
    return text.strip().rstrip().replace("\n", "").replace("\r", "").replace("\t", "") # != ""

def can_be_subtitle_or_subsubtitle(e : str) :

    regex_subsubtitle = '^\w+\.\w?|^\(?\w+\)\w?'
    regex_subsubtitle_no_romain = '^\w{,1}\.\w?|^\(?\w{,1}\)\w?'
    romain_set = {"i", "v", "x", ".", ")", "("}
    e_strip = e.replace("'", "")
    #print("e_strip",  e_strip)
    m = re.match(regex_subsubtitle, e_strip)
    if m :
        v = e_strip.split(" \n")
        id = v[0]
        if set(id).issubset(romain_set) :
            return "romain"
        else :
            if len(id) <= 3 and re.match(regex_subsubtitle_no_romain, id) :
                if ('.' in id or id[1] == ')') and len(id) == 2 :
                    return "no_romain"


def can_be_title(e : str):
  regex_title = '[0-9]+\. \\n\w?'
  if re.search(regex_title, e) :
      return " \n"
  else :
     regex_title = '[0-9]+\.\w?'
     return " "

def pyMuPDF_clauses_extraction(document):

    doc = fitz.open(document)

    fake_string = '__fake__'
    current = '0.'
    current_title = current
    current_subtitle = ""
    current_subsubtitle = ""
    dico = {}
    dico[current] = []
    for page in doc :
        blocks = page.getText("blocks")
        for line in blocks :
            line = line[4]
            if text_is_valid(line) :
                a = can_be_title(line)
                b = can_be_subtitle_or_subsubtitle(line)
                if a or b :
                    #print("==0", line)
                    v = line.split(a)
                    id = v[0]
                    text = a.join(v[1:])
                    
                    current = id
                    """
                    if a :
                        current_title = id
                        current = id
                    if b == "no_romain" :
                        current_subtitle = id
                        current = (current_title +"." if current_title else "") + id
                    if b == "romain" :
                        current_subsubtitle = id
                        current = (current_title +"." if current_title else "")+ (current_subtitle +"." if current_subtitle else "")+ id
                    """
                    
                    if current in dico.keys():
                        i = 1
                        while id+fake_string+str(i) in dico.keys():
                            i += 1
                        current = id+fake_string+str(i)

                    try :
                        dico[current].append(text)
                        #print(current, current_title, "2 === ",line)
                    except  :
                        dico[current] = [text]
                        #print(current, current_title, "1 === ",line)
                    
                else :
                    #print(repr(line))
                    dico[current].append(line)
                    #print(current, current_title, "3   === ",line)
    
    for k , v in dico.copy().items():
        dico[k] = "\n".join(v)             
    
    return dico


def get_content(document):
    """reads and separates a document file (pdf, docx, doc, txt, md) into a list of clauses"""

    _, extension = os.path.splitext(document)
    clauses = None
    join = " "
    if extension == ".pdf" :
        """
        pdfReader = PyPDF2.PdfFileReader(open(document, "rb"))
        content = [pdfReader.getPage(page).extractText() for page in range(pdfReader.numPages)]
        content = [text for text in content if text_is_valid(text = text)]
        """
        
        clauses = pyMuPDF_clauses_extraction(document)
        content = list(clauses.values())
        
    elif extension in [".doc", ".docx"] :
        doc = docx.Document(open(document, "rb"))
        content = doc.paragraphs
        content = [para.text for para in content if text_is_valid(text = para.text)]
        clauses = {i : clause for i, clause in enumerate(content)}

    else :
        content = open(document, "r").read()
        content = content if text_is_valid(text = content) else ""
        join = "\n\n"
        content = content.split(join)
        clauses = {i : clause for i, clause in enumerate(content)}
        

    return content, extension, clauses, join

def get_ktrain_predict_method(ktrain_predictor):
    """Returns the prediction method from a ktrain predictor"""

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
                #output.append({"acceptability" : float(y[0]), "unacceptability" : float(y[1])})
                output.append(y)
            else :
                #output.append({"acceptability" : int(1-y), "unacceptability" : int(y)})
                output.append([1-y, y])
                
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