"""Plot fixed-instance replay sensitivity from the preserved summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
data=json.loads((root/'reference/sensitivity.json').read_text())
fig,ax=plt.subplots(figsize=(7,4.4),layout='constrained')
labels=['Layered: 1K elements, 32 base iterations',
        'Layered: 4K elements, 256 base iterations',
        'Branched: 4K elements, 1,024 base iterations']
for row,label,color in zip(data,labels,['#177d65','#c7732c','#4d65ab']):
    values=[row['measurements'][n]['6'] for n in ('1','8','64')]
    y=np.array([v['improvement_pct'] for v in values]);bounds=np.array([v['ci95_pct'] for v in values])
    ax.errorbar(np.arange(3),y,yerr=np.array([y-bounds[:,0],bounds[:,1]-y]),
                label=label,color=color,marker='o',capsize=4,linewidth=1.8)
ax.axhline(0,color='#555555',linewidth=.8)
ax.set(xticks=range(3),xticklabels=['1','8','64'],xlabel='Graph replays per timing interval',
       ylabel='Runtime reduction relative to original graph (%)',
       title='Same added-edge rule; different workloads and replay counts')
ax.grid(axis='y',alpha=.18);ax.legend(loc='lower left',fontsize=8.5,frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'reference/replay-sensitivity.png',dpi=180)
fig.savefig(root/'reference/replay-sensitivity.pdf')
