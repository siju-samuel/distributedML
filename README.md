# distributedML

This project aims to develop distributed algorithms for machine learning.

The first project is based on parameter server implementation of the 
distributed Stochastic Gradient Descent method (D-SGD).

## Dependencies
This code has been tested to work with the Anaconda 5.2 (Python 2.7) 
package manager. 
Following packages have to be additionally installed from Anaconda:
- grpcio v1.12.0
- protobuf v3.5.2
- autograd v1.2

Currently, the code can be run locally in three different terminals as:
```bash
$ python worker.py --id 1
$ python worker.py --id 2
$ python worker.py --id 3

```

It uses gRPC for communication between the nodes. The three nodes initially run the Paxos algorithm for leader election. The 
elected node is designated as the Parameter Server. Each worker node has a
copy of the [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
image dataset from which it independently chooses batches of images to train
its local model copy. It then sends the parameters from the trained model to 
the parameter server. The parameter server aggregates the parameters from each
worker and updates its model. The workers then fetch the updated parameters 
from the server and continue training.
