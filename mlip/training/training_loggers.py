# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from copy import deepcopy
from typing import Any

import numpy as np
from rich import print as table_print
from rich.table import Table

from mlip.training.training_io_handler import LogCategory

logger = logging.getLogger("mlip")


def convert_mse_to_rmse_in_logs(to_log: dict[str, Any]) -> dict[str, Any]:
    """Simple helper function to convert all MSE values to RMSE values in a given
    metrics object to log. To compute a correct RMSE, we need to take the square root
    at the very end and not before any averaging happens, hence, our logged metrics
    objects only contain MSE as a metric, which needs to be converted with
    this function.

    Args:
        to_log: The metrics dictionary.

    Returns:
        The metrics dictionary with any MSE entries converted to RMSE.
    """

    def _convert_value(key: str, value: Any) -> Any:
        return np.sqrt(value) if "mse_" in key else value

    return {
        key.replace("mse_", "rmse_"): _convert_value(key, value)
        for key, value in to_log.items()
    }


def _build_rich_table(
    metrics: dict[str, int | float], title: str, epoch_number: int, metric_color: str
) -> Table:
    table = Table(
        title=f"{title} at epoch {epoch_number}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    keys = [k.replace("_", " ").capitalize() for k in metrics.keys()]
    values = [v if isinstance(v, int) else f"{float(v):.3f}" for v in metrics.values()]

    for k, v in zip(keys, values):
        _k = f"[{metric_color}]{k}[/{metric_color}]"
        _v = f"[white]{v}[/white]"
        table.add_row(_k, _v)

    return table


def log_metrics_to_table(
    category: LogCategory, to_log: dict[str, Any], epoch_number: int
) -> None:
    """Logging function for the training loop which logs the metrics to a nice table.

    The table will be printed to the command line.

    This function also converts MSE metrics to RMSE before logging them.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        epoch_number: The current epoch number.
    """
    _to_log = convert_mse_to_rmse_in_logs(to_log)

    table = None
    if category == LogCategory.BEST_MODEL:
        table = _build_rich_table(_to_log, "Best model", epoch_number, "white")
    elif category == LogCategory.TRAIN_METRICS:
        table = _build_rich_table(_to_log, "Training set metrics", epoch_number, "cyan")
    elif category == LogCategory.EVAL_METRICS:
        table = _build_rich_table(
            _to_log, "Validation set metrics", epoch_number, "green"
        )
    elif category == LogCategory.TEST_METRICS:
        table = _build_rich_table(
            _to_log, "Test set metrics", epoch_number, "blue_violet"
        )

    if table is not None:
        table_print(table)


def _metrics_to_str(metrics: dict[str, Any]) -> str:
    result = ""
    for k, v in metrics.items():
        result += k.replace("_", " ").capitalize()
        result += " = "
        result += str(v if isinstance(v, int) else f"{float(v):.3f}")
        result += " | "

    if result == "":
        return result
    return result[:-3]  # remove final separator


def log_metrics_to_line(
    category: LogCategory, to_log: dict[str, Any], epoch_number: int
) -> None:
    """Logging function for the training loop which logs the metrics to a single line.

    This function also converts MSE metrics to RMSE before logging them.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        epoch_number: The current epoch number.
    """
    _to_log = convert_mse_to_rmse_in_logs(to_log)

    if category == LogCategory.BEST_MODEL:
        logger.info(
            "Best model: Loss = %.3f | Best epoch = %s",
            _to_log["best_loss"],
            _to_log["best_epoch"],
        )

    elif category == LogCategory.TRAIN_METRICS:
        # Make sure loss is first
        to_log_copy = deepcopy(_to_log)
        to_log_modified = {"loss": to_log_copy.pop("loss")}
        to_log_modified.update(to_log_copy)

        logger.info("------------ Epoch %s ------------", epoch_number)
        logger.info("%-11s %s", "Training:", _metrics_to_str(to_log_modified))

    elif category == LogCategory.EVAL_METRICS:
        logger.info("Validation: %s", _metrics_to_str(_to_log))

    elif category == LogCategory.TEST_METRICS:
        logger.info("Testing: %s", _metrics_to_str(_to_log))
