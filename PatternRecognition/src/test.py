from __init__ import main_with_path
import matplotlib.pyplot as plt
import numpy as np
import json
import os

if __name__ == '__main__':
    try:
        with os.scandir("Images") as files:
            for file in files:
                if file.is_file():
                    name, ext = str(file.name).split('.')
                    if name[:3] == 'img' and ext == 'dcm':
                        main_with_path(f"{str(file.name)}","pat02.dcm")
    except:
        with open('test_data.json', 'r') as json_file:
            data = json.load(json_file)
        
        test_result = data["test_data"]
        prec, rec = [], []
        tp, tn, fp, fn = 0, 0, 0, 0
        for xi in test_result:
            tp += xi[0]
            tn += xi[1]
            fp += xi[2]
            fn += xi[3]
            prec.append(tp/(tp + fp) if tp+fp>0 else (1 if tp>0 else 0))
            rec .append(tp/(tp + fn) if tp+fn>0 else (1 if tp>0 else 0))
        x = np.linspace(0,len(test_result)-1,len(test_result))
        plt.plot(x,prec,color="red",label="precision")
        plt.scatter(x,prec,color="red")
        plt.plot(x,rec,color="blue",label="recobrado")
        plt.scatter(x,rec,color="blue")
        plt.grid("dotted")
        plt.legend()
        plt.xlabel("Cantidad de casos")
        plt.ylabel("Valor de precision y recobrado")
        plt.show()