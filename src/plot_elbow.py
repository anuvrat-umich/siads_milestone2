"""
    plot_elbow.py
    Author: Anuvrat Chaturvedi
    Date: 11th June 2024
    Purpose: Plot the elbow curve for KMeans clustering to find the optimal number of clusters
    Input: inpdf (input dataframe)
    Input: seed (random seed for reproducibility, defaults to 42)
    Input: title (title for the plot, defaults to 'Elbow Curve')
    Input: show (show the plot, defaults to True)
    Output: plt, sse, silhouette_scores
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def plot_elbow(
    inpdf: pd.DataFrame, seed: int = 42, title="Elbow Curve", plot: bool = True
) -> plt:
    """
    Plot the elbow curve for KMeans clustering to find the optimal number of clusters
    param inpdf: input dataframe with features for clustering
    param seed: random seed for reproducibility. Defaults to 42
    param title: title for the plot. Defaults to 'Elbow Curve'
    return plt, sse, silhouette_scores
    """
    sse = []
    silhouette_scores = []
    X = inpdf.values

    for k in range(2, 15):

        kmeans = KMeans(n_clusters=k, random_state=seed)
        preds = kmeans.fit_predict(inpdf)

        sse.append(kmeans.inertia_)  # SSE for each n_clusters
        silhouette_scores.append(
            silhouette_score(inpdf, preds)
        )  # Silhouette score for each n_clusters

    # Plot the elbow curve for SSE for each number of clusters
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5)
    )  # Create a figure with 2 subplots
    # Plot the SSE (inertia) for each number of clusters
    ax[0].plot(range(2, 15), sse)
    ax[0].scatter(range(2, 15), sse)
    ax[0].set_title("Elbow Curve for SSE (Inertia)")
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("Inertia (SSE)")  # Lower inertia is better
    ax[0].grid()  # Show the grid
    # Plot the Silhouette Score for each number of clusters
    ax[1].plot(range(2, 15), silhouette_scores)
    ax[1].scatter(range(2, 15), silhouette_scores)
    ax[1].set_title("Elbow Curve for Silhouette Score")
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Silhouette Score")  # Higher Silhouette Score is better
    ax[1].grid()  # Show the grid
    # Add x and y axis labels
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    if plot:
        plt.show()
    else:
        plt.close()

    return fig, sse, silhouette_scores, preds


# Write the code to run the function from the command line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot the elbow curve for KMeans clustering to find the optimal number of clusters"
    )
    parser.add_argument(
        "-i",
        "-inpdf",
        type=str,
        help="Pickled dataframe with features for clustering",
        required=True,
    )
    parser.add_argument(
        "-s",
        "-seed",
        type=str,
        help="Random seed for reproducibility. Defaults to 42",
        required=False,
        default=42,
    )
    parser.add_argument(
        "-t",
        "-title",
        type=str,
        help="Title for the plot. Defaults to 'Elbow Curve'",
        required=False,
        default="Elbow Curve",
    )
    parser.add_argument(
        "-p",
        "-plot",
        type=bool,
        help="Plot the chart? Defaults to True",
        required=False,
        default=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Plot the elbow curve
    plot_elbow(
        inpdf=pd.read_pickle(args.close_prices_adj_top),
        seed=args.seed,
        title=args.title,
        plot=args.plot,
    )
