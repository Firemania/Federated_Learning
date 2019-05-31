# Federated_Learning
A Federated Learning computing framework for deep learning-based model development by both server and edge devices 

This is a Python package for the training/testing of deep learning-based models in server-edge collaborative computing framework, called Federated Learning.

Our initial version as of 05/30/2019 enables feature evaluation and training of CNN models by edge devices, as well as the training of base model in a server using the features extracted from edge devices, which enables data privacy. Our framework is not concerned with bandwith problem at the current stage, since the main applications here do not involve a very short time period with large data.

A training cycle typically operates like this:

First, the edge devices (clients) send a request to the server for model training.

Second, the server send a base model to the edges. The edges then extract features from their data using the deep learning model, which is showcased in Client_stage_2_modelling.py. Then the edges send the extracted feature data to the server. The server will train a new base model on the feature data (with labels), which is showcased in Server_stage_2_modelling.py.

Next, the server send the new base model to the edges. The edges can use the new base model and do transfer learning on each's own local data, as showcased in Client_stage_3_modelling.py. Now, each client has its own model. The model can be deployed.

The Federated Learning process can be in various forms. For flexible adjustment, you can use the modules in lib folder. For convenience of package development, the edge devices and the server share the same modules and functions in lib folder.


