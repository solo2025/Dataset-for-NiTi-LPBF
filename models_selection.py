from matplotlib import axis
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import  LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor  

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json

# set the front and front size of the plots is Times New Roman
# front bold
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 20

# 1. Read in data set

data = pd.read_csv("../Dataset_transformations/original_data/v1.csv")
data = data.dropna()

# split the data set into training and validation set
train, valid = train_test_split(data, test_size=0.3, random_state=42)

# labels = "xNi,P/W,v/mm/s,t/um,h/um,Ev/J.mm-3,Es/J.mm-2,El/J.mm-1,RA,D/um,Oxygen/ppm,Ms,type".split(",")
# labels = "xNi,P/W,v/mm/s,t/um,h/um,Ev/J.mm-3,Es/J.mm-2,El/J.mm-1,RA,D/um,Oxygen/ppm,RelativeDensity/%,type".split(",")
labels = "xNi,P/W,v/mm/s,t/um,h/um,Ev/J.mm-3,Es/J.mm-2,El/J.mm-1,RA,D/um,Oxygen/ppm,UTS/MPa,EL/%,type".split(",")
# labels = "xNi,P/W,v/mm/s,t/um,h/um,Ev/J.mm-3,Es/J.mm-2,El/J.mm-1,RA,D/um,Oxygen/ppm,SIM-SMR/Mpa,type".split(",")

feature_labels = labels[0:11]
# choose the subset of features
# feature_labels = labels[0:3] + labels[4:6]
target_labels = labels[11:-2]

num_features = len(feature_labels)

train_features = train[feature_labels].values
train_targets = train[target_labels].values

valid_features = valid[feature_labels].values
valid_targets = valid[target_labels].values

# print(train_features)
# exit()

# preprocessing 
train_f_scaler = StandardScaler()
train_t_scaler = StandardScaler()
test_f_scaler = StandardScaler()
test_t_scaler = StandardScaler()

train_features = train_f_scaler.fit_transform(train_features)
train_targets = train_t_scaler.fit_transform(train_targets)

valid_features = test_f_scaler.fit_transform(valid_features)
valid_targets = test_t_scaler.fit_transform(valid_targets)

# 2. Training with different models with training set and score with validation set

scores = []
# cv_scores = []
models = []
training_predictions = []
valid_predictions = []

# 2.1. linear regression
lr = LinearRegression()
models.append(lr)
lr.fit(train_features, train_targets)
# predict the training targets and validation targets
lr_train_predicted = lr.predict(train_features).reshape(-1,1)
lr_valid_predicted = lr.predict(valid_features).reshape(-1,1)

lr_score = lr.score(valid_features, valid_targets)
scores.append(lr_score)
# cv_scores.append(cross_val_score(lr, train_features, train_targets, cv=5))
training_predictions.append(lr_train_predicted)
valid_predictions.append(lr_valid_predicted)

# 2.2. Gaussian process regression
gauss_kernel = ConstantKernel()* Matern(length_scale=1.0e-5, nu=0.5)+ WhiteKernel(noise_level=0.05)
gpr = GaussianProcessRegressor(kernel=gauss_kernel, alpha=1.0e-1,
                                optimizer="fmin_l_bfgs_b",
                                n_restarts_optimizer=50)
models.append(gpr)
gpr.fit(train_features, train_targets)
# predict the training targets and validation targets
gpr_train_predicted = gpr.predict(train_features).reshape(-1,1)
gpr_valid_predicted = gpr.predict(valid_features).reshape(-1,1)

gpr_score = gpr.score(valid_features, valid_targets)
scores.append(gpr_score)
# cv_scores.append(cross_val_score(gpr, train_features, train_targets, cv=5))
training_predictions.append(gpr_train_predicted)
valid_predictions.append(gpr_valid_predicted)

# 2.3. MLP regression
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), 
                activation='relu', 
                solver='adam', 
                alpha=0.0001,
                batch_size='auto')
models.append(mlp)
mlp.fit(train_features, train_targets)
# predict the training targets and validation targets
mlp_train_predicted = mlp.predict(train_features).reshape(-1,1)
mlp_valid_predicted = mlp.predict(valid_features).reshape(-1,1)

mlp_score = mlp.score(valid_features, valid_targets)
scores.append(mlp_score)
# cv_scores.append(cross_val_score(mlp, train_features, train_targets, cv=5))
training_predictions.append(mlp_train_predicted)
valid_predictions.append(mlp_valid_predicted)

# 2.4. SVR
svr = SVR(kernel='rbf', C=1e1, gamma=0.1, epsilon=0.1, max_iter=100)
models.append(svr)
svr.fit(train_features, train_targets)
# predict the training targets and validation targets
svr_train_predicted = svr.predict(train_features).reshape(-1,1)
svr_valid_predicted = svr.predict(valid_features).reshape(-1,1)

svr_score = svr.score(valid_features, valid_targets)
scores.append(svr_score)
# cv_scores.append(cross_val_score(svr, train_features, train_targets, cv=5))
training_predictions.append(svr_train_predicted)
valid_predictions.append(svr_valid_predicted)

# 2.5. Gradient Boosting regression
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
models.append(gbr)
gbr.fit(train_features, train_targets)
# predict the training targets and validation targets
gbr_train_predicted = gbr.predict(train_features).reshape(-1,1)
gbr_valid_predicted = gbr.predict(valid_features).reshape(-1,1)

gbr_score = gbr.score(valid_features, valid_targets)
scores.append(gbr_score)
# cv_scores.append(cross_val_score(gbr, train_features, train_targets, cv=5))
training_predictions.append(gbr_train_predicted)
valid_predictions.append(gbr_valid_predicted)


# 3. inverse transform the predicted targets and validation targets to the original scale
for i in range(len(training_predictions)):
    training_predictions[i] = train_t_scaler.inverse_transform(training_predictions[i])
    valid_predictions[i] = test_t_scaler.inverse_transform(valid_predictions[i])
# inverse transform the training targets and validation targets to the original scale
train_targets = train_t_scaler.inverse_transform(train_targets)
valid_targets = test_t_scaler.inverse_transform(valid_targets)


# 4. plot the training predictions with training targets and validation predictions with validation targets
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
for i in range(len(training_predictions)):
    # make each subplot have the same x-y range and the same aspect ratio
    # axs[i].set_xlim([250, 350])
    # axs[i].set_ylim([250, 350])
    # axs[i].set_xlim([92, 100])
    # axs[i].set_ylim([92, 100])
    # axs[i].set_ylim([100, 450])
    # axs[i].set_xlim([100, 450])
    # axs[i].set_ylim([5, 20])
    # axs[i].set_xlim([5, 20])
    axs[i].set_ylim([300, 850])
    axs[i].set_xlim([300, 850])

    ax = plt.gca()
    ax.set_aspect(1)
    
    # axs[i].set_aspect('equal')
    axs[i].plot([0, 1], [0, 1], transform=axs[i].transAxes, ls="--", c="#c32f27", linewidth=3)

    axs[i].scatter(train_targets, training_predictions[i], c='#023e8a', label='Training', alpha=0.5, s=60)
    axs[i].scatter(valid_targets, valid_predictions[i], c='#058c42', label='Validation', marker='D',alpha=0.5,s=60)

    axs[i].set_title(models[i].__class__.__name__, fontsize=20)
    axs[i].set_xlabel('Predicted UTS, MPa', fontsize=16)
    axs[i].set_ylabel('Measured UTS, MPa', fontsize=16)
    # set legend size smaller
    axs[i].legend(loc='upper left', prop={'size': 14})
    # set the tick label size
    axs[i].tick_params(axis='both', which='major', labelsize=16)
    # set the tick size
    axs[i].tick_params(axis='both', which='major', length=5, width=1)

# plt.tight_layout()
# plt.show()


# plt.savefig('figures/UTS.png', dpi=600,
#         bbox_inches='tight',
#         transparent=True)

# 5. plot the scores of the models
fig, ax = plt.subplots(figsize=(10, 10))
ax.bar(range(len(models)), scores,color='#023e8a', alpha=0.5, align='center',hatch='/')
ax.set_xticks(range(len(models)))
# set a shorter name for the models
ax.set_xticklabels([type(m).__name__ for m in models], rotation=45, fontsize=16)

# Display specific values on bar graph
for i, v in enumerate(scores):
    ax.text(i, v + 0.01, str(round(v, 4)), color='black', fontweight='bold', fontsize=20, ha='center')
# draw a line chart for the scores
ax.plot(scores, color='#ff6b6b', linewidth=3, marker='o', markersize=14, markerfacecolor='r', markeredgecolor='#023e8a')

ax.set_title("Scores of models on the validation set", fontsize=22, fontweight='bold')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.tick_params(axis='both', which='major', labelsize=16, width=5)
ax.set_ylabel("R2 scores", fontsize=18)

# #设置轴线宽度
# ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
plt.show()

# plt.savefig('figures/UTS_scores.png', dpi=600, bbox_inches='tight', transparent=True)