from matplotlib import axis
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import optuna
import logging
import sys
import pickle

from sklearn.linear_model import  LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
# joblib is used to save the model
import joblib
import json

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'

class Model_Train(object):

    def __init__(self, features_labels) -> None:
        
        self.features_labels = features_labels

        with open("config.json", "r") as f:
            config = json.load(f)

        self.datasets = config["datasets"]
        self.datasets_labels = config["datasets_labels"]
        self.full_dataset_dir = config["full_dataset_dir"]

        self.mean = config["mean"]
        self.std = config["std"]

        self.ranges = config["ranges"]

    def show_train_result(self, train_targets, train_predictions, valid_targets, valid_predictions, target_label,
                           train_score, valid_score, dest="figure.png"):
         # plot the figure
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(train_targets, train_predictions,  c='#023e8a', label='Training, R2=%.2f' % train_score , alpha=0.6, s=60)
        ax.scatter(valid_targets, valid_predictions,  c='#058c42', label='Testing, R2=%.2f'%valid_score, marker='D',alpha=0.4,s=40)

        ax.plot([train_targets.min(), train_targets.max()], [train_targets.min(), train_targets.max()], ls="--", c="#c32f27", linewidth=3)
        # ax.plot([0, 1], [0, 1], transform=ax.transAxes,  ls="--", c="#c32f27",linewidth=3)
        ax.plot([train_targets.min(), train_targets.max()], [train_targets.min(), train_targets.max()], ls="--", c="#c32f27", linewidth=3)

        # # plot the train scores and valid scores
        # ax.text(0.05, 0.9, "Train Score: %.3f" % train_score, fontsize=16, transform=ax.transAxes)
        # ax.text(0.05, 0.85, "Test Score: %.3f" % valid_score, fontsize=16, transform=ax.transAxes)

        ax.set_xlabel("Measured, %s" % target_label, fontsize=20,fontweight='bold')
        ax.set_ylabel("Predicted, %s" % target_label, fontsize=20,fontweight='bold')
        

        # ax.set_xlim(93.5,100.5)
        # ax.set_ylim(93.5,100.5)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5, width=3)
        ax.legend(loc='upper left', prop={'size': 16})

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        fig.savefig(dest, dpi=600, bbox_inches='tight', pad_inches=0.1,transparent=True)

        plt.close(fig)

    def train(self, features_labels, target_label, table, dest="figure.png", model_dest="model.pkl"):

        labels = features_labels + [target_label]

        data = table[labels]

        data_copy = data.copy()

        mean = [self.mean[each] for each in labels]
        # transform mean to pandas series
        mean = pd.Series(mean, index=labels)
        std = [self.std[each] for each in labels]
        # transform std to pandas series
        std = pd.Series(std, index=labels)

        # normalize the data
        data = (data - mean) / std

        # split the data set into training and validation setshow_train_result
        train, valid = train_test_split(data, test_size=0.25, random_state=42)

        train_features = train[features_labels].values
        train_targets = train[target_label].values

        # train with gpr 
        kernel = ConstantKernel() + Matern(length_scale=2,  
                                           nu=0.5) + WhiteKernel(noise_level=1)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gpr.fit(train_features, train_targets)

        # scores on the training set and validation set
        train_score = gpr.score(train_features, train_targets)
        valid_score = gpr.score(valid[features_labels].values, valid[target_label].values)

        # save the model
        if not os.path.exists("models"):
            os.mkdir("models")
        
        joblib.dump(gpr, model_dest)

        # make prediction
        valid_features = valid[features_labels].values
        valid_targets = valid[target_label].values
        valid_predictions = gpr.predict(valid_features)
        valid_predictions = valid_predictions * std[-1] + mean[-1]

        # for train
        train_predictions = gpr.predict(train_features)
        train_predictions = train_predictions * std[-1] + mean[-1]

        # denomarlize the data: valid_targets and train_targets
        valid_targets = valid_targets * std[-1] + mean[-1]
        train_targets = train_targets * std[-1] + mean[-1]

        self.show_train_result(train_targets, train_predictions, valid_targets, valid_predictions, target_label, 
                               train_score, valid_score, dest=dest)

    def run_training(self):
        target_labels = [each for each in self.datasets_labels]

        for target_label in target_labels:

            table = pd.read_csv(self.datasets[target_label])

            self.train(self.features_labels, self.datasets_labels[target_label], table, dest="figures/prediction/%s.png" % target_label, model_dest="models/%s.pkl" % target_label)

    def make_prediction_table(self, features_labels, target_label, mean, std, table, model_src, dest=None):
        # load the model
        model = joblib.load(model_src)

        labels = features_labels + [target_label]

        data = table[labels]
        data = (data - mean) / std

        features = data[features_labels].values

        #  model.predict(features, return_std=True)
        predictions, std_values = model.predict(features, return_std=True)

        predictions = predictions * std[-1] + mean[-1]
        std_values = std_values * std[-1]

        table[target_label + "-pred"] = predictions
        table[target_label + "-std"] = std_values
        # lower bound
        table[target_label + "-low"] = table[target_label + "-pred"] - 1.96 * table[target_label + "-std"]
        # upper bound
        table[target_label + "-up"] = table[target_label + "-pred"] + 1.96 * table[target_label + "-std"]

        if dest is not None:
            table.to_csv(dest, index=False)

        return table

    def make_predictions(self):
        target_labels = [each for each in self.datasets_labels]

        for target_label in target_labels:

            print(self.datasets)
            table = pd.read_csv(self.datasets[target_label])

            current_labels = [each for each in self.features_labels] + [self.datasets_labels[target_label]]

            mean = [self.mean[each] for each in current_labels]
            # transform mean to pandas series
            mean = pd.Series(mean, index=current_labels)
            std = [self.std[each] for each in current_labels]
            # transform std to pandas series
            std = pd.Series(std, index=current_labels)

            self.make_prediction_table(self.features_labels, self.datasets_labels[target_label], mean, std, table, model_src="models/%s.pkl" % target_label, dest="predictions/%s.csv" % target_label)

    def fill_table(self):

        table = pd.read_csv(self.full_dataset_dir)
        
        target_labels = [each for each in self.datasets_labels]

        for target_label in target_labels:
            current_labels = [each for each in self.features_labels] + [self.datasets_labels[target_label]]
            # print(current_labels)
            data = table[current_labels]

            mean = [self.mean[each] for each in current_labels]
            # transform mean to pandas series
            mean = pd.Series(mean, index=current_labels)
            std = [self.std[each] for each in current_labels]
            # transform std to pandas series
            std = pd.Series(std, index=current_labels)

            # load the model
            model_src = "models/%s.pkl" % target_label

            table = self.make_prediction_table(self.features_labels, self.datasets_labels[target_label], mean, std, table, model_src=model_src)

        table.to_csv("full_dataset.csv", index=False)

    def make_inference(self):
        features_labels = self.features_labels

        fixed = {
            "xNi": 50.0,
            "Oxygen/ppm": 200.0,
            "D/um": 100.0
        }

        mean = [self.mean[each] for each in features_labels]
        std = [self.std[each] for each in features_labels]

        changing_features = [each for each in features_labels if each not in fixed]
        lower_bounds_for_changing_features = [self.ranges[each][0] for each in changing_features]
        upper_bounds_for_changing_features = [self.ranges[each][1] for each in changing_features]

        # print(changing_features)
        # print(lower_bounds_for_changing_features)
        # print(upper_bounds_for_changing_features)

        # make a grid for changing features
        grid = np.mgrid[
            lower_bounds_for_changing_features[0]:upper_bounds_for_changing_features[0]:10, 
            lower_bounds_for_changing_features[1]:upper_bounds_for_changing_features[1]:100,
            lower_bounds_for_changing_features[2]:upper_bounds_for_changing_features[2]:5]
        
        grid = grid.reshape(3, -1).T

        print(grid.shape)
        # turn it into a dataframe
        grid = pd.DataFrame(grid, columns=changing_features)

        # add fixed features
        for each in fixed:
            grid[each] = fixed[each]

        table = grid

        target_labels = [each for each in self.datasets_labels]
        for target_label in target_labels:

            table[self.datasets_labels[target_label]] = 0.0

            current_labels = [each for each in self.features_labels] + [self.datasets_labels[target_label]]

            mean = [self.mean[each] for each in current_labels]
            # transform mean to pandas series
            mean = pd.Series(mean, index=current_labels)
            std = [self.std[each] for each in current_labels]
            # transform std to pandas series
            std = pd.Series(std, index=current_labels)

            # load the model
            model_src = "models/%s.pkl" % target_label

            table = self.make_prediction_table(self.features_labels, self.datasets_labels[target_label], mean, std, table, model_src=model_src)

        table.to_csv("infered_small_dataset.csv", index=False)
    
    def make_inference2(self):
        from calphadmesh.makegrid import makemultigrid

        changing_features = "xNi,P/W,v/mm/s,h/um,t/um,RA,D/um,Oxygen/ppm".split(",")
    

        gridnumbers = [6, 5, 11, 3, 2, 2, 3, 5]

        variables = []
        for each in makemultigrid(gridnumbers):

            for index, label in enumerate(changing_features):
                lower = self.ranges[label][0]
                upper = self.ranges[label][1]
                each[index] = lower + (upper - lower) * each[index]
            variables.append(each)
        # print(variables)
        
        grid = pd.DataFrame(variables, columns=changing_features)

        # print the first 10 rows
        # print(grid.head(10))
        # exit()
        grid["Ev/J.mm-3"] = grid["P/W"]/grid["v/mm/s"]/grid["h/um"]/grid["t/um"]*1e6
        

        table = grid

        target_labels = [each for each in self.datasets_labels]

        for target_label in target_labels:

            table[self.datasets_labels[target_label]] = 0.0

            current_labels = [each for each in self.features_labels] + [self.datasets_labels[target_label]]

            mean = [self.mean[each] for each in current_labels]
            # transform mean to pandas series
            mean = pd.Series(mean, index=current_labels)
            std = [self.std[each] for each in current_labels]
            # transform std to pandas series
            std = pd.Series(std, index=current_labels)

            # load the model
            model_src = "models/%s.pkl" % target_label

            table = self.make_prediction_table(self.features_labels, self.datasets_labels[target_label], mean, std, table, model_src=model_src)
        
        table.to_csv("infered_full_dataset_0726.csv", index=False)

    def make_inference3(self):
        from calphadmesh.makegrid import makemultigrid

        changing_features = "P/W,v/mm/s,h/um".split(",")

        fixed = {
            "xNi": 50.6,
            "t/um": 30,
            "RA": 67,
            "Oxygen/ppm": 300.0,
            "D/um": 100.0
        }

        # gridnumbers = [5, 9, 3]
        gridnumbers = [10,11,6]

        variables = []
        for each in makemultigrid(gridnumbers):

            for index, label in enumerate(changing_features):
                lower = self.ranges[label][0]
                upper = self.ranges[label][1]
                each[index] = lower + (upper - lower) * each[index]
            variables.append(each)
        # print(variables)
        
        grid = pd.DataFrame(variables, columns=changing_features)

        # add fixed features
        for each in fixed:
            grid[each] = fixed[each]

        # print the first 10 rows
        # print(grid.head(10))
        # exit()
        grid["Ev/J.mm-3"] = grid["P/W"]/grid["v/mm/s"]/grid["h/um"]/grid["t/um"]*1e6
        
        

        table = grid

        target_labels = [each for each in self.datasets_labels]

        for target_label in target_labels:

            table[self.datasets_labels[target_label]] = 0.0

            current_labels = [each for each in self.features_labels] + [self.datasets_labels[target_label]]

            mean = [self.mean[each] for each in current_labels]
            # transform mean to pandas series
            mean = pd.Series(mean, index=current_labels)
            std = [self.std[each] for each in current_labels]
            # transform std to pandas series
            std = pd.Series(std, index=current_labels)

            # load the model
            model_src = "models/%s.pkl" % target_label

            table = self.make_prediction_table(self.features_labels, self.datasets_labels[target_label], mean, std, table, model_src=model_src)
        
        table.to_csv("fixed_pvh.csv", index=False)

    def make_single_var_predictions(self):

        features_labels = "xNi,P/W,v/mm/s,h/um,t/um,RA,Ev/J.mm-3,D/um,Oxygen/ppm".split(",")
        # features_labels = self.features_labels

        # set the fixed features and their values
        fixed = {
            "xNi": 50.6,
            "P/W": 120.0,
            "v/mm/s": 900,
            "h/um": 70,
            "t/um": 30,
            "Ev/J.mm-3": 63.5,
            "RA": 67,
            # "Oxygen/ppm": 300.0,
            "D/um": 100.0
        }

        mean = [self.mean[each] for each in features_labels]
        std = [self.std[each] for each in features_labels]

        changing_features = [each for each in features_labels if each not in fixed]
        lower_bounds = [self.ranges[each][0] for each in changing_features]
        upper_bounds = [self.ranges[each][1] for each in changing_features]

        print(lower_bounds, upper_bounds)

        # 对于changing_features生成一列变化特征
        grid = np.linspace(lower_bounds[0], upper_bounds[0], 20)

        print(grid.shape)
        print(changing_features)

        grid = grid.reshape(1, -1).T

        grid = pd.DataFrame(grid, columns=changing_features)

        # grid["Ev/J.mm-3"] = grid["P/W"]/grid["v/mm/s"]/grid["h/um"]/grid["t/um"]*1e6

        # add fixed features
        for each in fixed:
            grid[each] = fixed[each]
        
        table = grid

        target_labels = [each for each in self.datasets_labels]
        for target_label in target_labels:

            table[self.datasets_labels[target_label]] = 0.0

            current_labels = [each for each in self.features_labels] + [self.datasets_labels[target_label]]

            mean = [self.mean[each] for each in current_labels]
            # transform mean to pandas series
            mean = pd.Series(mean, index=current_labels)
            std = [self.std[each] for each in current_labels]
            # transform std to pandas series
            std = pd.Series(std, index=current_labels)

            # load the model
            model_src = "models/%s.pkl" % target_label

            table = self.make_prediction_table(self.features_labels, self.datasets_labels[target_label], mean, std, table, model_src=model_src)
        
        os.makedirs("predictions/single_infered", exist_ok=True)

        # table.to_csv("predictions/single_infered/%s.csv" % changing_features, index=False)
        table.to_csv("predictions/single_infered/O2.csv", index=False)


        # extract one feature from the features_labels and make predictions
        # for each in features_labels:

        #     # make a grid for changing features
        #     grid = np.mgrid[
        #         self.ranges[each][0]:self.ranges[each][1]:10
        #     ]
            
        #     grid = grid.reshape(1, -1).T

        

        #     # turn it into a dataframe
        #     grid = pd.DataFrame(grid, columns=[each])

            

        #     # add fixed features
        #     for each in features_labels:
        #         if each != each:
        #             grid[each] = 0.0

        #     table = grid

        #     target_labels = [each for each in self.datasets_labels]
        #     for target_label in target_labels:

        #         table[self.datasets_labels[target_label]] = 0.0

        #         current_labels = [each for each in self.features_labels] + [self.datasets_labels[target_label]]

        #         mean = [self.mean[each] for each in current_labels]
        #         # transform mean to pandas series
        #         mean = pd.Series(mean, index=current_labels)
        #         std = [self.std[each] for each in current_labels]
        #         # transform std to pandas series
        #         std = pd.Series(std, index=current_labels)

        #         # load the model
        #         model_src = "models/%s.pkl" % target_label

        #         table = self.make_prediction_table(self.features_labels, self.datasets_labels[target_label], mean, std, table, model_src=model_src)
            
        #     os.makedirs("/predictions/infered", exist_ok=True)
        

        #     table.to_csv("preditions/infered/infered_%s.csv" % each, index=False)



    def new_study(study_name):

        # add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())

        # create a study
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name,
                                    load_if_exists=True,
                                    sampler= optuna.samplers.TPESampler(seed=42),
                                    directions = ["maximize","maximize","maximize"])

        def objective(trail,features_labels,self):

            # give the range of features
            fixed = {
                "xNi": 50.0,
                "Oxygen/ppm": 200.0,
                "D/um": 100.0
            }
            
            a1 = trail.suggest_float("P/W", 50.0, 200.0, step = 5)
            a2 = trail.suggest_float("v/mm/s", 50.0, 1000.0, step = 10)
            a3 = trail.suggest_float("h/um", 70.0, 100.0,step= 5)
            
            pred_density = self.make_predictions(features_labels, fixed, a1, a2, a3, model_src="models/density.pkl")
            pred_uts = self.make_predictions(features_labels, fixed, a1, a2, a3, model_src="models/uts.pkl")
            pred_el = self.make_predictions(features_labels, fixed, a1, a2, a3, model_src="models/el.pkl")

            print(pred_density, pred_uts, pred_el)

            
            return pred_density, pred_uts, pred_el


        study.optimize(objective, n_trials=100)

        # print and plot the pareto front
        print(study.best_trials)

        # save the study
        study_name = "studies/%s.pkl" % study_name
        with open(study_name, "wb") as f:
            pickle.dump(study, f)

        # plot the study
        optuna.visualization.plot_pareto_front(study).write_html("studies/%s.html" % study_name)

if __name__ == "__main__":

    features_labels = "xNi,P/W,v/mm/s,h/um,t/um,Ev/J.mm-3,RA,D/um,Oxygen/ppm".split(",")
    # features_labels = "xNi,P/W,v/mm/s,h/um,D/um,Oxygen/ppm".split(",")

    trainer = Model_Train(features_labels)

    trainer.run_training()
    # trainer.make_predictions()
    # trainer.fill_table()
    # trainer.make_inference()

    # inference 2 : change all the features
    # trainer.make_inference2()

    # inference 3: fix composition and the equipment parameters
    trainer.make_inference3()
    # trainer.make_single_var_predictions()
    trainer.new_study()