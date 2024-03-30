import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if len(X) > 120:
            break
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    logdir = os.path.join(args.logdir, "*")
    sub_folders = glob.glob(logdir)
    print(sub_folders)
    for f in sub_folders:
        logdir = os.path.join(f, 'events*')
        eventfiles = glob.glob(logdir)

        X,Y = get_section_results(eventfiles[0])
        X_data = np.zeros((len(X),len(eventfiles)))
        Y_data = np.zeros((len(Y),len(eventfiles)))
        for i in range(len(eventfiles)):
            X, Y = get_section_results(eventfiles[i])
            X_data[:,i] = X
            Y_data[:,i] = Y

        X = np.mean(X_data[1:], axis=1)
        Y = np.mean(Y_data, axis=1)
        Y_std = np.std(Y_data, axis=1)
    
        plt.plot(X,Y, '-', label=f[9:])
        plt.fill_between(X, Y-Y_std, Y+Y_std, alpha=0.6)
    plt.legend()
    plt.show()