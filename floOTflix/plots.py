from numpy import iterable
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(matrices, title=None, titles=[], size=7, lambdas=[]):
    """Plot a chip-distribution pie chart.

    Args:
        pi: probabilities to plot
        title: Title of the plot
    """
    fig, ax = plt.subplots(1, len(matrices),
                           figsize=(len(matrices) * size, size*.8))
    if len(titles) < len(matrices):
        titles += [None] * (len(matrices) - len(titles))
    if len(lambdas) < len(matrices):
        lambdas += [lambda x: x] * (len(matrices) - len(lambdas))
    if not iterable(ax):
        ax = [ax]
    for matrix, ax, m_title, fct in zip(matrices, ax, titles, lambdas):
        s = sns.heatmap(matrix, ax=ax)
        s.set_title(m_title)
        fct(s)
    if title:
        fig.suptitle(title)
