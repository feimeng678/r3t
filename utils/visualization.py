from collections import deque
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np

def visualize_node_tree_2D(rrt, fig=None, ax=None):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    node_queue = deque([rrt.root_node])
    lines = []
    while node_queue:
        node = node_queue.popleft()
        if node.children is not None:
            node_queue.extendleft(node.children)
            for child in node.children:
                lines.append([np.ndarray.flatten(node.state), np.ndarray.flatten(child.state)])
        ax.scatter(*np.ndarray.flatten(node.state), c='gray', s=2)
    lc = mc.LineCollection(lines, linewidths=0.5, colors='gray')
    ax.add_collection(lc)
    return fig, ax
