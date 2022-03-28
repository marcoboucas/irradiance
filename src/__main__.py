"""Main file."""

import os
import logging
from typing import List, Dict, Tuple
import yaml
import warnings

from fire import Fire

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models import get_model, BasePredictionModel
from src.evaluation import get_evaluator, BaseEvaluator
from src.data import load_data
from src.preprocessing.dataset import generate_dataset

DatasetTuple = Tuple[np.ndarray, np.ndarray]


class CLI:
    """CLI class."""

    def __init__(self):
        """Initialize CLI."""
        self.logger = logging.getLogger("CLI")
        os.system("cls" if os.name == "nt" else "clear")

    def __read_config(self, path: str) -> Dict:
        """Read the model config"""
        with open(path, "r", encoding="utf-8") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def __run_one_model(
        self,
        model_name: str,
        train_sets: DatasetTuple,
        val_sets: DatasetTuple,
        config: Dict,
    ) -> BasePredictionModel:
        """Run just one model."""
        self.logger.info("Running model : %s", model_name)
        model = get_model(**config)

        # Train the model
        model.fit(
            X_train=train_sets[0],
            y_train=train_sets[1],
            X_val=val_sets[0],
            y_val=val_sets[1],
        )

        return model

    def __evaluate_one_model(
        self,
        model: BasePredictionModel,
        test_sets: DatasetTuple,
        evaluators: List[BaseEvaluator],
        how_many: int = 100,
        max_i: int = 10,
    ) -> None:
        """Evaluate one model."""
        self.logger.info("Evaluating the model %s...", model.__class__.__name__)
        X_test, y_test = test_sets
        results = []
        for i, x in enumerate(X_test):
            y_pred = model.predict(x, how_many=how_many, verbose=True)

            # Change the data aspect to enable the comparison
            indexes = np.cumsum(np.ones_like(y_pred))
            input_size = x.shape[0]
            input_indexes, output_indexes = indexes[:input_size], indexes[input_size:]


            input_values = x
            prediction_values = y_pred[input_size:]
            true_values = y_test[:prediction_values.shape[0]]
            if True:
                plt.figure()
                plt.title("Comparison between prediction and reality (diff level)")
                indexes = np.cumsum(np.ones_like(y_pred))
                input_size = x.shape[0]
                input_indexes, output_indexes = indexes[:input_size], indexes[input_size:]
                

                plt.plot(input_indexes, input_values, label="Input value")
                plt.plot(output_indexes, prediction_values, label="Prediction")
                plt.plot(output_indexes, true_values, label="Real")
                plt.legend()


            # Make cumsums
            input_values = np.cumsum(input_values)
            prediction_values = input_values[-1] + np.cumsum(prediction_values)
            true_values =  input_values[-1] + np.cumsum(true_values)

            # Plot the prediction
            if True:
                plt.figure()
                plt.title("Comparison between prediction and reality")
                indexes = np.cumsum(np.ones_like(y_pred))
                input_size = x.shape[0]
                input_indexes, output_indexes = indexes[:input_size], indexes[input_size:]
                

                plt.plot(input_indexes, input_values, label="Input value")
                plt.plot(output_indexes, prediction_values, label="Prediction")
                plt.plot(output_indexes, true_values, label="Real")
                plt.legend()
                plt.show()
            assert y_pred.shape[0] == x.shape[0] + how_many
            results.append(
                {
                    f"{model.__class__.__name__} | {evaluator.__class__.__name__}": evaluator(
                        pred=y_pred, truth=y_test
                    )
                    for evaluator in evaluators
                }
            )
            break  # TODO: Change the y_test by shifting it to enable several predictions
        return pd.DataFrame.from_records(results)

    def __load_datasets(
        self, dataset_config: Dict
    ) -> Tuple[DatasetTuple, DatasetTuple, DatasetTuple]:
        """Load the datasets."""
        datasets = {}
        for dataset_name in ["train", "val", "test"]:
            location = dataset_config["locations"][0]
            self.logger.info("Loading the %s dataset...", dataset_name)
            data = load_data(
                data_path=dataset_config["data_path"],
                year=dataset_config[f"{dataset_name}_year"],
            )[:, location["x"], location["y"]]
            datasets[dataset_name] = generate_dataset(
                data=data,  # TODO: We can reduce the size of the dataset here
                shift=dataset_config["shift"],
                make_diff=dataset_config["make_diff"],
                add_is_day=dataset_config["add_is_day"],
            )
        return datasets["train"], datasets["val"], datasets["test"]

    def __load_evaluators(self, evaluators_config: List[str]) -> List[BaseEvaluator]:
        """Load the evaluators."""
        return [get_evaluator(class_path) for class_path in evaluators_config]

    def run(self, display_graph: bool = True):
        """Run the pipeline."""
        self.logger.info("Running the pipeline...")
        warnings.filterwarnings("ignore")  # TODO: Not good, but pretty and efficient

        # Load the yaml file with all the configuration
        global_config = self.__read_config("./config.yml")

        # Load the dataset
        self.logger.info("Loading the dataset...")
        train_sets, val_sets, test_sets = self.__load_datasets(global_config["dataset"])

        # Load the evaluators
        evaluators = self.__load_evaluators(global_config["evaluation"]["evaluators"])

        # Train each model and make the evaluation
        global_results: List[pd.DataFrame] = []
        for model_name, config in global_config["models"].items():
            model = self.__run_one_model(
                model_name=model_name,
                config=config,
                train_sets=train_sets,
                val_sets=val_sets,
            )
            global_results.append(
                self.__evaluate_one_model(
                    model=model,
                    test_sets=test_sets,
                    evaluators=evaluators,
                    how_many=global_config["evaluation"]["how_many"],
                    max_i=global_config["evaluation"]["max_per_model"],
                )
            )

        results = pd.concat(global_results, axis=1)
        print(results.head())
        results.to_csv("./results.csv")
        results_stats = results.describe().drop(index=["count"])
        print(results_stats.head(10))
        results_stats.to_csv("./results_stats.csv")

        _, ax = plt.subplots()
        results = pd.concat(
            [
                pd.DataFrame({"model": column, "score": results[column]})
                for column in results.columns
            ],
            ignore_index=True,
        )
        sns.histplot(
            data=results,
            x="score",
            hue="model",
            bins=10,
            kde=True,
            ax=ax,
            stat="percent",
            common_bins=False,
            common_norm=False,
        )
        plt.savefig("./results_hist.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(CLI)
