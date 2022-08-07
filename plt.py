import matplotlib.pyplot as plt
import pandas as pd

def plt_metrics(x, y1, y2):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    
    ax.plot(x, y1, 'b', label="dropout")
    ax.plot(x, y2, 'r', label="no_dropout")
    ax.set_title("acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("acc")
    plt.legend()
    #plt.savefig(col[1] + ".pdf", format="pdf", dpi=1200)
    plt.savefig("dropout_vs_nodropout.png", format="png", dpi=1200)
    plt.show()
if __name__ == "__main__":
    data = pd.read_csv("./metrics_05.csv", sep='\t')
    col = ['loss', 'train_acc', 'test_acc']
    y1 = data['test_acc']
    data = pd.read_csv("./metrics_0.csv", sep='\t')
    y2 = data['test_acc']
    x = list(range(len(y1)))
    plt_metrics(x, y1, y2)