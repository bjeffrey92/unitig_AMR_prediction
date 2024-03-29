import glob
import os
import pickle
import re
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def load_lasso_data(data_dir):
    if not data_dir.endswith("/"):
        data_dir += "/"
    input_files = glob.glob(data_dir + "log2*pkl")
    Abs = [os.path.split(i)[-1].split("_mic")[0] + "_mic" for i in input_files]
    data_dict = {}
    for i in range(len(Abs)):
        with open(input_files[i], "rb") as a:
            data_dict[Abs[i]] = pickle.load(a)
    return data_dict


def convert_to_dataframe_lasso(CV_results):
    """
    For formatting the output of the lasso model
    """

    df_dictionary = {
        "left_out_clade": [],
        "training_accuracy": [],
        "testing_accuracy": [],
        "validation_accuracy": [],
    }  # will be converted to df
    for left_out_clade, d in CV_results.items():
        d_val = {i: d[i]["validation_accuracy"] for i in d.keys()}
        k = max(d_val, key=d_val.get)  # key with max validation data accuracy
        df_dictionary["left_out_clade"].append(left_out_clade)
        df_dictionary["training_accuracy"].append(d[k]["training_accuracy"])
        df_dictionary["testing_accuracy"].append(d[k]["testing_accuracy"])
        df_dictionary["validation_accuracy"].append(d[k]["validation_accuracy"])

    return pd.DataFrame(df_dictionary)


def convert_to_dataframe(CV_results, SVR_data=False):
    df_dictionary = {
        "left_out_clade": [],
        "training_accuracy": [],
        "testing_accuracy": [],
        "validation_accuracy": [],
    }  # will be converted to df
    for left_out_clade, d in CV_results.items():
        df_dictionary["left_out_clade"].append(left_out_clade)
        if isinstance(d, Dict):
            if SVR_data:
                df_dictionary["training_accuracy"].append(
                    d["accuracies"]["training_accuracy"]
                )
                df_dictionary["testing_accuracy"].append(
                    d["accuracies"]["testing_accuracy"]
                )
                df_dictionary["validation_accuracy"].append(
                    d["accuracies"]["validation_accuracy"]
                )
            else:
                df_dictionary["training_accuracy"].append(
                    d["accuracies"][0]["training_accuracy"]
                )
                df_dictionary["testing_accuracy"].append(
                    d["accuracies"][0]["testing_accuracy"]
                )
                df_dictionary["validation_accuracy"].append(
                    d["accuracies"][0]["validation_accuracy"]
                )
        else:
            df_dictionary["training_accuracy"].append(d.training_accuracy)
            df_dictionary["testing_accuracy"].append(d.testing_accuracy)
            df_dictionary["validation_accuracy"].append(d.validation_accuracy)

    return pd.DataFrame(df_dictionary)


def _get_NN_results(data_dir, file_suffix=".tsv"):
    """
    Returns data from NN fitting metrics in form to be plotted with plot_results
    """
    if not data_dir.endswith("/"):
        data_dir += "/"

    input_files = glob.glob(data_dir + f"*{file_suffix}")
    Abs = [os.path.split(i)[-1].split("_mic")[0] + "_mic" for i in input_files]

    data_dict = {}
    for Ab in set(Abs):
        Ab_files = [i for i in input_files if i.startswith(data_dir + Ab)]
        Ab_accuracies = [None] * len(Ab_files)
        i = 0
        for f in Ab_files:
            left_out_clade = re.search(r"clade_(.*?)_left", f).group(1)
            last_line = pd.read_csv(Ab_files[0], sep="\t").iloc[-1]
            accuracies = last_line[
                [
                    "training_data_acc",
                    "testing_data_acc",
                    "validation_data_acc",
                ]
            ]
            accuracies["left_out_clade"] = int(left_out_clade)
            Ab_accuracies[i] = accuracies
            i += 1
        df = pd.concat(Ab_accuracies, axis=1).transpose()
        df.set_index("left_out_clade", inplace=True)
        df.columns = [
            "training_accuracy",
            "testing_accuracy",
            "validation_accuracy",
        ]
        data_dict[Ab] = df

    return data_dict


def get_SVR_results(dir_pattern):
    Abs = ["log2_azm_mic", "log2_cip_mic", "log2_cro_mic", "log2_cfx_mic"]

    input_files = [os.path.join(dir_pattern[0], Ab, dir_pattern[1]) for Ab in Abs]

    data_dict = {}
    for i in range(len(Abs)):
        with open(input_files[i], "rb") as a:
            data_dict[Abs[i]] = pickle.load(a)
    return data_dict


def density_plot(preds, labels, axs=None, axs_0=None, axs_1=None):
    df = pd.DataFrame({"Predictions": preds, "Truth": labels})
    if axs is not None:
        df.plot(kind="kde", ax=axs[axs_0, axs_1], legend=False)
        return axs
    else:
        df.plot(kind="kde")


def plot_train_preds_and_labels(data, Ab):
    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)
    n = 0
    m = 0
    for left_out_clade, d in data[Ab].items():
        axs = density_plot(
            d["model_outputs"]["training_predictions"],
            d["model_outputs"]["training_labels"],
            axs=axs,
            axs_0=n,
            axs_1=m,
        )

        # axs = density_plot(d['model_outputs']['testing_predictions'],
        #                     d['model_outputs']['testing_labels'],
        #                     axs = axs,
        #                     axs_0 = n,
        #                     axs_1 = 1)

        # axs = density_plot(d['model_outputs']['validation_predictions'],
        #                     d['model_outputs']['validation_labels'],
        #                     axs = axs,
        #                     axs_0 = n,
        #                     axs_1 = 2)

        if n > 0 and n % 3 == 0:
            n = 0
            m += 1
        else:
            n += 1
    fig.tight_layout()


def bar_plot_of_results(data, filename):
    fig, axs = plt.subplots(1, 4, sharey=True)
    width = 0.2
    n = 0
    for Ab in data.keys():
        df = data[Ab]
        axs[n].bar(
            df.index - width,
            df.training_accuracy,
            width,
            label="Training Data",
        )
        axs[n].bar(df.index, df.testing_accuracy, width, label="Testing Data")
        axs[n].bar(
            df.index + width,
            df.validation_accuracy,
            width,
            label="Validation Data",
        )
        axs[n].set_xticks(df.index)
        axs[n].set_title(Ab.upper().split("_")[1])
        n += 1

    axs[n - 1].legend(loc="lower right")
    fig.text(0.5, 0.04, "Left Out Clade", ha="center")
    fig.text(0.04, 0.5, "Prediction Accuracy (%)", va="center", rotation="vertical")

    fig.savefig(filename)


def rename_test_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "validation_accuracy": "testing_accuracy",
            "testing_accuracy": "validation_accuracy",
        }
    )


def box_plot_of_results(data, filename):
    fig, axs = plt.subplots(1, 4, sharey=True)
    n = 0
    for Ab in data.keys():
        df = data[Ab]
        df.rename(
            columns={
                "training_accuracy": "Train",
                "testing_accuracy": "Test",
                "validation_accuracy": "Validate",
            },
            inplace=True,
        )
        df = df[["Train", "Validate", "Test"]]
        axs[n].boxplot(df, notch=False, labels=df.columns)
        axs[n].set_title(Ab.upper().split("_")[1])
        axs[n].set(ylim=(0, 100))
        axs[n].tick_params(labelrotation=90)
        n += 1

    fig.text(0, 0.5, "Mean Accuracy per MIC Bin", va="center", rotation="vertical")
    fig.tight_layout()

    fig.savefig(filename)


if __name__ == "__main__":
    data_dir = "lasso_model/results/linear_model_results/cross_validation_results/"  # noqa: E501
    data = load_lasso_data(data_dir)

    # svr results are stored within subdir with name of Ab,
    # unlike rest of results which are all stored in one dir
    # dir_pattern = (
    #     "kernel_learning",
    #     "cluster_wise_cross_validation/SVR_results/SVR_CV_results_mean_acc_per_bin.pkl",
    # )
    # data_dir = "."
    # data = get_SVR_results(dir_pattern)

    Abs = list(data.keys())
    Abs.sort()  # to maintain order
    accuracy_data = {Ab: convert_to_dataframe(data[Ab]) for Ab in Abs}
    accuracy_data = {
        Ab: df.set_index("left_out_clade") for Ab, df in accuracy_data.items()
    }
    accuracy_data = {k: rename_test_and_validate(v) for k, v in accuracy_data.items()}

    box_plot_of_results(accuracy_data, os.path.join(data_dir, "CV_accuracy.png"))
