import matplotlib.pyplot as plt
import os
import numpy as np

def plot_loss_curve(train_loss, test_loss, img_path, eval_freq):
    """
    plot loss curve and save the image to path provided in img_path

    """
    steps = np.arange(1, len(train_loss) + 1)
    steps *= eval_freq
    plt.plot(steps, train_loss, label = "train loss")
    plt.plot(steps, test_loss, label = "test loss")
    plt.xlabel("Steps")
    plt.ylabel('Loss')
    plt.title(os.path.basename(img_path))
    
    dir_name = os.path.dirname(img_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 
    plt.legend(loc="best", frameon=False)
    plt.savefig(img_path)
    plt.close()


def plot_acc_curve(accs, img_path, eval_freq):
    """
    plot accuracy curve and save the image to path provided in img_path

    """
    steps = np.arange(1, len(accs) + 1)
    steps *= eval_freq
    plt.plot(steps, accs)
    plt.xlabel("Steps")
    plt.ylabel('Accuracy')
    plt.title(os.path.basename(img_path))

    dir_name = os.path.dirname(img_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) 
    plt.savefig(img_path)
    plt.close()

if __name__ == '__main__':
    train_loss = [0.02, 0.001, 0.0001]
    test_loss = [0.032, 0.0031, 0.00031]

    plot_loss_curve(train_loss, test_loss, "loss_curves/hello", 100) 
    accs = [0.23, 0.25, 0.3]
    plot_acc_curve(accs, "loss_curves/hello1", 100)
