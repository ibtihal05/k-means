from flask import Flask, request, render_template, redirect
import os
from sklearn.cluster import KMeans
import fonction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#clusturing spectral

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
data_path=""
data_array=()
cluster=[]
nb_cluster=0

IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route("/data", methods=["POST","GET"])
def data():
    global data_path
    path = os.path.join(APP_ROOT, 'dataset/')
    if not os.path.isdir(path):
        os.mkdir(path)
    if request.method == "POST":
        data = request.files['data']
        dataset_name = data.filename
        data_path = "/".join([path, dataset_name])
        data.save(data_path)
    
        return render_template("model.html")
    else :
        return render_template("data.html")
    

@app.route("/model", methods=["POST","GET"])
def model():
    global data_path
    global cluster
    global nb_cluster
    global data_array
    #read the dataset
    dataset = pd.read_csv(data_path)
    data=dataset.iloc[:200,1:4]
    for i in range(len(data.axes[1])):
        data[data.axes[1][i]].fillna(data[data.axes[1][i]].mode(), inplace=True)
    data_array=np.array(data)  
    data_array=data_array.tolist()
    col_num,col_nomi=fonction.is_nominal(data, data.axes[1])
    data_num=fonction.normaliser_data(data[col_num])
    data_array = np.concatenate((data_num, data[col_nomi]), axis=1)

    if request.method == "POST" :
        model=request.form["model"]
        if model=="k_automatique" :
            #execute directly the code
            seuil_distance=fonction.seuil_distance_function(data_array)
            print(seuil_distance)
            nb_cluster=fonction.k_auto(data_array,seuil_distance)
            print(nb_cluster)
            
            nb_iteration=2000
            cluster=fonction.kmeans(nb_cluster,data_array.tolist(),nb_iteration)    
            return render_template("result.html",nb_cluster=nb_cluster,cluster=cluster)
        elif model=="k_utilisateur" :
            #get the number of cluster k first
            return render_template("k_utilisateur.html")
        else :
            return redirect(request.url)
   
    else :
        return render_template("model.html")


@app.route("/k_utilisateur", methods=["POST","GET"])
def k_utilisateur():
    global cluster
    global nb_cluster
    global data_array
    if request.method == "POST" :
        #get the number of cluster k
        nb_cluster=int(request.form["k"])
        #execute the code
        nb_iteration=2000
        cluster=fonction.kmeans(nb_cluster,data_array.tolist(),nb_iteration)    
            
        return render_template("result.html",nb_cluster=nb_cluster,cluster=cluster)
    else :
        return render_template("k_utilisateur.html")

@app.route("/result")
def result():
    global cluster
    global nb_cluster
    
    return render_template("result.html",nb_cluster=nb_cluster,cluster=cluster)
if __name__ == "__main__":
    app.run()