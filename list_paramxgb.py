import json
import os
n_estimators_list = [100, 150, 200]
max_depth_list = [3, 4, 5, 6, 7]
learning_rate_list = [0.01, 0.05, 0.075, 0.1, 0.2]
subsample_list = [0.5, 1]
gamma_list = [0]
min_child_weight_list = [1] #default 1
lambda_list = [1] #default 1
alpha_list = [0] #default 0

sub_file_template = ""

counter = 0

os.makedirs("hps/xgb", exist_ok=True)
for n_estimator in n_estimators_list:
  for learning_rate in learning_rate_list:
    for max_depth in max_depth_list:
      for subsample in subsample_list:
        for gamma in gamma_list:
          for min_child_weight in min_child_weight_list:
            for lambda_val in lambda_list:
              for alpha_val in alpha_list:
                with open(f"hps/xgb/XGB_hp_{counter}.json", "w") as f:
                  json.dump({"n_estimators": n_estimator, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample, "gamma": gamma, "min_child_weight": min_child_weight, "lambda": lambda_val, "alpha": alpha_val}, f)

                sub_file_template += f"/home/hep/tlt26/T2K_Rw/hps/xgb/XGB_hp_{counter}.json\n"
                counter += 1

sub_file_template += ")"

with open("hps/xgb/XGB_job.sub", "w") as f:
  f.write(sub_file_template)