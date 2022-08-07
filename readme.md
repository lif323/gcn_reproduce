# Graph Convolutional Networks
This is pytorch implementation of Graph Convolutional Networks for task of(semi-supervised) classification of nodes in a graph, as described in paper:
Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
# Requirements
- pytorch 1.12.1 cpu
- python 3.9.7
- # Quickstart
```
python gcn.py
```
# Bug
There is a bug in the officially implemented GCN, the test nodes are sorted. The discussion on this bug is as follows: https://github.com/tkipf/gcn/issues/76
# Result
![](./dropout_vs_nodropout.png)  
