import matplotlib.pyplot as plt

# Dataset stats
nodes = [92877, 246131, 488376, 800390, 1533879, 2483726]
transactions = [473997, 985600, 2026700, 3792994, 8283209, 12634517]
snapshots = [3126, 6414, 8995, 16138, 32935, 49646]

# AUCs
gclstm_auc = [0.617196, 0.618736, 0.573311, 0.627615, 0.658960, 0.662549]
htgn_auc = [0.615221, 0.667577, 0.675932, 0.704038, 0.713838, 0.726769]

# Data pack labels
pack_labels = ['pack 2', 'pack 4', 'pack 8', 'pack 16', 'pack 32', 'pack 64']

# Function to plot and save
def plot_with_pack_labels(x, x_label, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, gclstm_auc, marker='o', label='MiNT-GCLSTM')
    plt.plot(x, htgn_auc, marker='s', label='MiNT-HTGN')

    for i in range(len(x)):
        if i == 0:
            # Label only once at first point
            avg_y = (gclstm_auc[i] + htgn_auc[i]) / 2
            plt.text(x[i], avg_y + 0.005, pack_labels[i], ha='center', fontsize=9)
        else:
            plt.text(x[i], gclstm_auc[i] + 0.005, pack_labels[i], ha='center', fontsize=9)
            plt.text(x[i], htgn_auc[i] + 0.005, pack_labels[i], ha='center', fontsize=9)

    plt.xlabel(x_label)
    plt.ylabel('AUC')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Plot and save
plot_with_pack_labels(nodes, '#Nodes', 'AUC vs #Nodes', 'pic/auc_vs_nodes.png')
plot_with_pack_labels(transactions, '#Transactions', 'AUC vs #Transactions', 'pic/auc_vs_transactions.png')
plot_with_pack_labels(snapshots, '#Snapshots', 'AUC vs #Snapshots', 'pic/auc_vs_snapshots.png')
