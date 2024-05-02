import os
import pandas as pd
import matplotlib.pyplot as plt

labels_dir = "../../data/input/raw/labels"  #Change the path to where you save all raw token networks( download from gg drive)
labels_stats_filename = "../../data/output/labels/"
for filename in os.listdir(labels_dir):
    labels_df = pd.read_csv("{}/{}".format(labels_dir,filename))
    plt.figure(figsize=(15, 6))  # Set the figure size
    points = labels_df.iloc[:, 0]
    plt.plot(range(len(points)), points,linestyle='-', color='b')  # Plot the points
    plt.xlabel('snapshot')  # Set the x-axis label
    plt.ylabel('label')  # Set the y-axis label
    plt.yticks([0, 1])
    plt.title("Snapshots' labels")  # Set the chart title

    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(
        "{}/{}.png".format(labels_stats_filename,filename.split('.')[0]))
