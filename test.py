import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import argparse
import torch

#import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
RS =123
from sklearn.manifold import TSNE
import time

from data import MNIST_Test
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = 10
    #palette = np.array(sns.color_palette("hls", num_classes))
    palette = np.array(["#67E568","#257F27","#08420D","#FFF000","#FFB62B","#E56124","#E53E30","#7F2353","#F911FF","#9F8CA6"])

    # create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=14)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()
    return f, ax, sc, txts


def test(args):
	no_of_test = 2000
	model = Model()
	model.load_state_dict(torch.load("epoch12_20"))
	model.eval()
	mnist = MNIST_Test(args.list_domainS, args.list_domainT, args.list_root)
	time_start = time.time()
	x_subset = np.empty((no_of_test*2, 128))
	y = np.empty((no_of_test*2))
	for i in range(no_of_test):
		print(i)
		d = mnist[i]
		Xs, Xt, _, _, _, _, _, _, _, _ = model(d[0], d[1], d[1])
		x_subset[2*i] = Xs.detach().numpy()
		x_subset[2*i + 1] = Xt.detach().numpy()
		y[2*i] = d[2]
		y[2*i + 1] = d[3]

	fashion_tsne = TSNE(random_state=RS).fit_transform(x_subset)
	fashion_scatter(fashion_tsne, y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--list_domainS',
                        default='mnist_m/mnist_m_test_labels.txt')
    parser.add_argument('--list_domainT',
                        default='mnistTest.txt')
    parser.add_argument('--list_root',
                        default='mnist_m/mnist_m_train')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    test(args)
