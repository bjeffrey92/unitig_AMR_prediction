import glob
import os
import pickle
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ingot_dr.utils import ResultsContainer


def plot_data(data: pd.DataFrame, score: str, title: str, filename: str):
    plt.clf()
    sns.set_theme(style="whitegrid")
    sns.catplot(
        data=data,
        kind="bar",
        x="Antibiotic",
        y=score,
        hue="Data",
    )
    plt.subplots_adjust(bottom=0.1, top=1.1)
    plt.title(title)
    plt.savefig(f"ingot_dr/results/{filename}")
    plt.clf()


def format_data(
    all_results: Dict[str, ResultsContainer], score: str
) -> pd.DataFrame:
    data_dict = {
        "Antibiotic": [],
        "Data": [],
        "Score": [],
    }
    for ab, data in all_results.items():
        data_dict["Antibiotic"].extend([ab] * 2)
        data_dict["Data"].extend(["Training Data", "Testing Data"])
        if score == "Accuracy":
            data_dict["Score"].extend(
                [data.training_accuracy, data.testing_accuracy]
            )
        elif score == "Balanced Accuracy":
            data_dict["Score"].extend(
                [
                    data.training_balanced_accuracy,
                    data.testing_balanced_accuracy,
                ]
            )
        else:
            raise ValueError(score)

    df = pd.DataFrame(data_dict)
    df.rename(columns={"Score": score}, inplace=True)
    df.sort_values(by="Antibiotic", inplace=True)
    return df


def load_data(data_file: str) -> pd.DataFrame:
    with open(data_file, "rb") as a:
        return pickle.load(a)


def main(results_files: List[str], data_type: str):
    all_results = {
        re.search(r"log2_(.*?)_mic", i).group(1).upper(): load_data(i)  # type: ignore # noqa: E501
        for i in results_files
    }

    data = format_data(all_results, "Accuracy")
    plot_data(
        data, "Accuracy", f"{data_type} Data", f"{data_type}_data_accuracy.png"
    )

    data = format_data(all_results, "Balanced Accuracy")
    plot_data(
        data,
        "Balanced Accuracy",
        f"{data_type} Data",
        f"{data_type}_data_balanced_accuracy.png",
    )


if __name__ == "__main__":
    main(glob.glob("ingot_dr/results/convolved*"), "Convolved")
    main(glob.glob("ingot_dr/results/reduced*"), "Reduced")
