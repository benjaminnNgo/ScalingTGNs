import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plotting the training losses
def plot_results(path, filename, y_label, data_list, train=True):
        if train:
            columns = ["epoch", "train_loss", "train_auc", "train_ap"]
            sublplot_rows = 3
        else:
            columns = ["epoch", "test_auc", "test_ap"]
            sublplot_rows = 2

        avg_path = '{}/average/{}_epochlist.csv'.format(path, filename)
        epoch = range(0, y_axis_value["epoch"])

        average_data = pd.read_csv(avg_path)
        fig, axs = plt.subplot(sublplot_rows, len(data_list)+1)

        for r in sublplot_rows:
            axs[r, 0].plot(epoch, average_data[r+1])
            axs[r, 0].set_title(columns[r+1])
            for i, data in enumerate(data_list):
                data_path = '{}/data/{}/{}_epochlist.csv'.format(path, data, filename)
                y_axis_value = pd.read_csv(data_path)
                axs[r, i].plot(epoch, y_axis_value, label='Training Loss')
                axs[r, i].set_title('Data {} {}'.format(data, columns[r+1]))
                axs.set(xlabel='Epochs', ylabel=columns[r+1])

        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.xticks(np.arange(0, epoch, 50))
        # plt.legend(loc='best')
        # plt.show()
        plt.savefig(f'../data/output/figures/train_loss.png')


if __name__ == '__main__':
    train_path = "../data/output/training_test"
    eval_path = "../data/output/epoch_result"
    file_date = ""
    plot_results(path=train_path, filename=file_date, data=["ALBT", "POLS"])