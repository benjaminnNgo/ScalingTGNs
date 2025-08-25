import seaborn as sns
import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))



def log_tick_formatter(x,pos):
    return f'{int(x)}'



x = [2,4,8,16,32,64]
mint_htgn = [0.615, 0.667, 0.676, 0.704, 0.7138, 0.7267]
error = [0.116950624, 0.111100689, 0.099041845, 0.115632369, 0.106874049, 0.113716785]
htgn_single = 0.68

df = pd.DataFrame({'Number of Networks': x, 'AUC': mint_htgn, 'error': error})

mint_gclstm = [0.617,0.619,0.573, 0.628,0.659,0.663]
gclstm_error = [0.091,0.119,0.125,0.147,0.106,0.098]

df_gclstm = pd.DataFrame({'x': x, 'y': mint_gclstm, 'error': gclstm_error})
gclstm_single = 0.608765236

sns.lineplot(x='Number of Networks', y='AUC', data=df, color='blue', label="MiNT Model", ax=ax1)
ax1.scatter(df['Number of Networks'], df['AUC'], color='blue', s=100, zorder=5)
ax1.axhline(y=htgn_single, color='orange', linestyle='--', label="Single Model")
ax1.fill_between(df['Number of Networks'], 
                 df['AUC'] - df['error'],
                 df['AUC'] + df['error'],
                 alpha=0.1)
ax1.set_xlabel("Number of Training Networks", size=14)
ax1.set_ylabel("AUC", size=14)
ax1.set_ylim(0.5, 0.8)

ax1.set_xscale('log', base=2)
ax1.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
ax1.tick_params(axis='both', which='major', labelsize=14)

ax1.text(0.5, -0.2, '(a). HTGN', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
ax1.legend(fontsize=14)


sns.lineplot(x='x', y='y', data=df_gclstm, color='blue', label="MiNT Model", ax=ax2)
ax2.scatter(df_gclstm['x'], df_gclstm['y'], color='blue', s=100, zorder=5)
ax2.axhline(y=gclstm_single, color='orange', linestyle='--', label="Single Model")
ax2.fill_between(df_gclstm['x'], 
                 df_gclstm['y'] - df_gclstm['error'],
                 df_gclstm['y'] + df_gclstm['error'],
                 alpha=0.1)
ax2.set_xlabel("Number of Training Networks", size=14)
ax2.set_ylabel("AUC", size=14)
ax2.set_ylim(0.5, 0.8)

ax2.set_xscale('log', base=2)
ax2.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
ax2.text(0.5, -0.2, '(b). GCLSTM', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
ax2.legend(fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)


sns.set_style("whitegrid")
sns.set_palette("deep")
plt.tight_layout()
plt.savefig("fig1.pdf")
# plt.ion()
# plt.show()

