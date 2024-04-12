import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import os

plt.rcParams['font.family'] = 'Times New Roman'

os.environ["OMP_NUM_THREADS"] = '10'

def plot_original_data(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['UTS/MPa-pred'], data['EL/%-pred'], data['Hysteresis/K-pred'], c='b', marker='o')
    ax.set_xlabel('UTS/MPa-pred')
    ax.set_ylabel('EL/%-pred')
    ax.set_zlabel('Hysteresis/K-pred')
    plt.show()


def num_of_clusters(origin_dataset, target_labels, n_min, n_max):
    
    if len(origin_dataset) > 1e5:
      dataset = origin_dataset.sample(frac=0.0025, random_state=42)
    else:
      dataset = origin_dataset

    target_labels = target_labels.split(",")
    X = dataset[target_labels]

    for i in range(n_min, n_max):
      # 标准化数据
      scaler = MinMaxScaler()
      X_scaled = scaler.fit_transform(X)
      
      #应用 K-means 算法
      kmeans = KMeans(n_clusters=i, random_state=42, n_init=20, max_iter=300)
      kmeans.fit(X_scaled)
      labels = kmeans.labels_
      centers = kmeans.cluster_centers_
      
      #聚类分析的轮廓系数
      # silhouette_avg = silhouette_score(X_scaled, labels)
      # print("轮廓系数:", silhouette_avg)
      
      db_score = davies_bouldin_score(X_scaled, labels)
      print("Davies-Bouldin 指数:", db_score)

      #记录每次循环中的i和db_score
      if not os.path.exists('num_clusters'):
            os.makedirs('num_clusters', exist_ok=True)
      with open('num_clusters/case_3.csv', 'a') as f:
              f.write(str(i) + ',' + str(db_score) + '\n')



def clustering(origin_dataset, target_labels, n_cluster):
    
    if len(origin_dataset) > 1e5:
      dataset = origin_dataset.sample(frac=0.0025, random_state=42)
    else:
      dataset = origin_dataset
    
#     print(len(dataset))
#     exit()
    target_labels = target_labels.split(",")
    df = dataset[target_labels]

    # 标准化数据
    scaler = MinMaxScaler()
    df_scalered = scaler.fit_transform(df)

    #应用 K-means 算法
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=20, max_iter=300)
    kmeans.fit(df_scalered)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_labels = kmeans.fit_predict(df_scalered)



    df = scaler.inverse_transform(df_scalered)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_cluster):
            cluster_data = df[cluster_labels == i]
            ax.scatter(cluster_data[:, 0], 
                       cluster_data[:, 1], 
                       cluster_data[:, 2],
                       cmap='viridis', 
                       alpha=0.5, 
                       s=80,
                       label=f'Cluster {i+1}')
    
    
    ax.set_xlabel(target_labels[0], fontsize=16, fontweight="bold")
    ax.set_ylabel(target_labels[1], fontsize=16, fontweight="bold")
    ax.set_zlabel(target_labels[2], fontsize=16, fontweight="bold")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc='upper right', fontsize=16)
    
    ax.view_init(20, 240)
    
    plt.tight_layout()
    plt.show()
#     plt.savefig("clustered_data/clustered_0.0025/target_cluster.png", dpi=300, bbox_inches='tight')

    cluster_labels = cluster_labels + 1
    dataset['cluster_labels'] = cluster_labels

    dataset.to_csv('clustered_data/clustered_0.0025_case3.csv', index=False)


def plot_num_clusters():
    # 以db_score.csv中第一列为x轴，第二列为y轴绘图
    selected_n = pd.read_csv('num_clusters/case_3.csv', header=None)
    selected_n.columns = ['n', 'db_score']

    plt.figure(figsize=(10, 6))
    plt.plot(selected_n['n'], selected_n['db_score'], 'o-', color='b')
    plt.xlabel('n')
    plt.ylabel('db_score')
    plt.show()


if __name__ == "__main__":
     
      # origin_dataset = pd.read_csv("data_source/infer_fixed_xtd_dataset.csv")
      origin_dataset = pd.read_csv("data_source/infered_full_dataset_0726.csv")
      # target_labels = "UTS/MPa-pred,EL/%-pred,Hysteresis/K-pred"
      # target_labels = "Ms/K-pred,EL/%-pred,Hysteresis/K-pred"
      target_labels = "SIM-SMR/MPa-pred,Ms/K-pred,EL/%-pred"

      n_min = 3
      n_max = 21

      # plot_original_data(origin_dataset)

      # num_of_clusters(origin_dataset, target_labels, n_min, n_max)

      n_cluster = 9
      clustering(origin_dataset, target_labels, n_cluster)
      
      # plot_num_clusters()