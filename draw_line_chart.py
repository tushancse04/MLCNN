import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
x = ['20k','50k','100k','200k','300k']
y = [68,72,80,88,87]
fmri = sns.load_dataset("fmri")
ax = sns.lineplot(x="timepoint", y="signal", data=fmri)