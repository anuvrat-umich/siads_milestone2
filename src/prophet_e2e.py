"""
    prophet_e2e.py
    Author: Anuvrat Chaturvedi
    Date: 1st June 2024
    Purpose: End-to-end model fitting and plotting for Prophet model for the given stock
"""

# Import libraries
import pandas as pd
from numpy import sqrt
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    r2_score,
    mean_absolute_percentage_error,
)
from prophet.utilities import regressor_coefficients
import logging


def prophet_e2e(
    close_prices_adj_top: pd.DataFrame,
    selected_stock: str,
    model: Prophet = None,
    test_size: int = 30,
    save_charts: bool = True,
    save_metrics: bool = True,
    output_dir: str = "../prophet_charts/",
    batch_id: str = pd.Timestamp.now(),
    verbose: bool = True,
):
    # Print the selected stock
    # print(f"Now predicting for: {selected_stock}")

    # Disabling the logging for the Prophet model
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)

    # Adding the backslash to the output directory if not present to make the location name consistent
    if output_dir[-1] != "/":
        output_dir += "/"

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist. Creating one at {output_dir}")
        os.makedirs(output_dir)

    # Select stock
    df_train_test = (
        close_prices_adj_top[selected_stock]
        .reset_index()
        .rename(columns={"index": "ds", selected_stock: "y"})
    )
    df_train_test["ds"] = pd.to_datetime(df_train_test["ds"])

    # Split the data into training and test sets
    test_indices = [_ for _ in df_train_test.ds.sort_values(ascending=False)][
        :test_size
    ]
    train = df_train_test[~df_train_test.ds.isin(test_indices)]
    test = df_train_test[df_train_test.ds.isin(test_indices)]

    # Define the Prophet model if none is passed
    if model is None:
        print("Model not passed. Using default Prophet model")
        model = Prophet()

    # Fit the defined model
    model.fit(train)

    # Make predictions on in sample data
    forecast_test = model.predict(test)

    # Save the chart if required
    if save_charts:
        insample_test = forecast_test.merge(test, on="ds")[["yhat", "y"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(insample_test.index, insample_test.yhat, label="Predicted")
        ax.plot(insample_test.index, insample_test.y, label="Actual")
        plt.title(f"{selected_stock} stock over time - in-sample prediction")
        ax.legend()
        _ = plt.savefig(output_dir + f"prophet_e2e_insample_{selected_stock}.png")
        plt.close(fig)

    # Save the chart if required
    if save_charts:
        # Showing the complete trend of the stock using the extrapolated generative model including historical data
        test_future = model.make_future_dataframe(periods=test_size * 2)
        test_forecast = model.predict(test_future)

        fig, ax = plt.subplots(figsize=(10, 5))
        model.plot(test_forecast, ax=ax)
        plt.title(
            f"{selected_stock} stock over time - simulating trend using the extrapolated generative model"
        )
        _ = plt.savefig(output_dir + f"prophet_e2e_simulation_{selected_stock}.png")
        plt.close(fig)

    # Define the out of sample period for which we want a prediction
    df_future = pd.DataFrame(
        train.ds.max()
        + pd.timedelta_range(start="1 days", periods=5 * test_size, freq="D"),
        columns=["ds"],
    )

    # Save the chart if required
    if save_charts:
        # Make predictions using the trained model
        forecast_outofsample = model.predict(df_future)

        # Plot the forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        model.plot(forecast_outofsample, ax=ax)
        ax.plot(test.ds, test.y, "r.", label="Test data")
        plt.title(
            f"{selected_stock} stock over time - forecast for the next 60 days including {test_size} days of test data"
        )
        plt.savefig(output_dir + f"prophet_e2e_forecast_{selected_stock}.png")
        plt.close(fig)

    # Create or update the output files with metrics
    if save_metrics:
        if os.path.exists(output_dir + "prophet_e2e_metrics.pkl"):
            metrics = pd.read_pickle(output_dir + "prophet_e2e_metrics.pkl")
        else:
            metrics = pd.DataFrame(
                columns=[
                    "stock",
                    "mae",
                    "mse",
                    "rmse",
                    "evs",
                    "r2",
                    "mape",
                    "batch_id",
                ]
            )

        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    {
                        "stock": [selected_stock],
                        "mae": [mean_absolute_error(test.y, forecast_test.yhat)],
                        "mse": [mean_squared_error(test.y, forecast_test.yhat)],
                        "rmse": [sqrt(mean_squared_error(test.y, forecast_test.yhat))],
                        "evs": [explained_variance_score(test.y, forecast_test.yhat)],
                        "r2": [r2_score(test.y, forecast_test.yhat)],
                        "mape": [
                            mean_absolute_percentage_error(test.y, forecast_test.yhat)
                        ],
                        "batch_id": batch_id,
                    }
                ),
            ],
            ignore_index=True,
        )

        # Save the regressor coefficients to the output directory
        # regressor_coef = regressor_coefficients(model)
        # regressor_coef.sort_values(  # [["regressor", "regressor_mode", "coef"]]
        #    "coef"
        # ).to_csv(output_dir + f"prophet_e2e_regressor_coef_{selected_stock}.csv")

        # Print the results if required
        if verbose:
            print(
                "*" * 100,
                f"\n {' '*10} Metrics for the model - ",
                selected_stock,
                "\n",
                "*" * 100,
            )
            print(metrics)

        # Save the metrics to the output directory
        metrics.to_pickle(output_dir + "prophet_e2e_metrics.pkl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prophet model end-to-end")
    parser.add_argument(
        "close_prices_adj_top", type=str, help="Pickled dataframe of prices"
    )
    parser.add_argument("selected_stock", type=str, help="Stock to fit the model on")
    parser.add_argument(
        "-m",
        "--model",
        type=Prophet,
        default=Prophet(),
        help="Estimator to use for the model fitting",
    )
    parser.add_argument(
        "-t",
        "--test_size",
        type=int,
        default=30,
        help="Number of days to predict",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        help="Output directory location",
        type=str,
        default="../prophet_charts/",
    )
    parser.add_argument(
        "-c",
        "--save_charts",
        action="store_true",
        help="Save charts?",
    )
    parser.add_argument(
        "-m",
        "--save_metrics",
        action="store_true",
        help="Save metrics?",
    )
    parser.add_argument(
        "-b",
        "--batch_id",
        type=str,
        default=str(pd.Timestamp.now()),
        help="Batch ID for the run",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode?")
    args = parser.parse_args()

    # print(args.close_prices_adj_top)

    # Load the data
    close_prices_adj_top_df = pd.read_pickle(args.close_prices_adj_top)

    # Run the model
    prophet_e2e(
        close_prices_adj_top=close_prices_adj_top_df,
        selected_stock=args.selected_stock,
        model=args.model,
        test_size=args.test_size,
        save_charts=args.save_charts,
        save_metrics=args.save_metrics,
        output_dir=args.output_directory,
        batch_id=args.batch_id,
        verbose=args.verbose,
    )
