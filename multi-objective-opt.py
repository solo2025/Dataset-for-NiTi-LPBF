import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 14
matplotlib.rcParams["ytick.labelsize"] = 14
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams['legend.fontsize'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def find_pareto(data : pd.DataFrame):
    num = len(data)
    indexs = []
    for i in range(num):
        for j in range(num):
            if i !=j:
                if (data.loc[i, "UTS/MPa-pred"] <= data.loc[j, "UTS/MPa-pred"]) & (data.loc[i, "EL/%-pred"] <= data.loc[j, "EL/%-pred"]):
                    indexs.append(i)
                    break
    # print(len(indexs))
    for each in indexs:
        data.drop([each], axis=0, inplace=True)
    data.sort_values("Hysteresis/K-pred", inplace=True)
    
    data.to_csv("pareto_pvh_1.csv")
    

    return data["Hysteresis/K-pred"], data["UTS/MPa-pred"], data["EL/%-pred"]


if __name__=="__main__":

    base = os.getcwd()
    data = pd.read_csv(os.path.join(base, "infer_fixed_pvh_dataset.csv"))

    hys, uts, el = data["Hysteresis/K-pred"], data["UTS/MPa-pred"], data["EL/%-pred"]
    hys_pareto, uts_pareto, el_pareto = find_pareto(data)
    # uts_pareto, el_pareto, hys_pareto = pd.read_csv("new_pareto_front.csv")["UTS/MPa-pred"], pd.read_csv("new_pareto_front.csv")["EL/%-pred"], pd.read_csv("new_pareto_front.csv")["Hysteresis/K-pred"]
    # rd_pareto_low, uts_pareto_low, el_pareto_low = find_pareto_low(data)
    # rd_pareto_up, uts_pareto_up, el_pareto_up = find_pareto_up(data)

    # print(rd_pareto, uts_pareto, elpareto)
    # exit()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.scatter( uts, el, hys, marker= "o", label="All data", color="blue", alpha=0.5)
    ax.scatter( uts_pareto, el_pareto, hys_pareto, marker="^", label="Pareto front", color="red", alpha=0.8)
    ax.set_xlabel("Predicted UTS/MPa")
    ax.set_ylabel("Predicted EL/%")
    ax.set_zlabel("Predicted hysteresis/K")
    # ax.grid(False)
    plt.legend()
    plt.show()
    