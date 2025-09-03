I have done this project for the final exam of 'Development and life cycle of artificial interrigence software'.

Dataset: FashionMNIST\
    dataset provided by torchvision. Consists in all differents 28x28 images of different clothes classified in 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

Neural network model used: SimpleCNN\
    I have built the model based on the neural network base class SimpleCNN.
    the 'lr' value of the optimizer has been found with 'optuna_search.py', that uses Optuna, an automatic
    hyperparameter optimization framework.

EDA:\
    the EDA rappresents the basics data, such as classes distribution and sample images.
    the plot confusion matrix (for the guesses) is in the train_FashionMNSIT.py if needed
    ![classes distribuiton](ClassesDistribution.jpg)
    ![sample imgaes](image-2.png)

tests/:\
    Folder containing the files needed to verify that everything works without errors.

requiremenst:\
    Required libraries to work on this project.

Docker:\
    Published the repository on DockerHub

Accuracy:

Epoch [1/5], Loss: 0.4896\
Epoch [2/5], Loss: 0.3159\
Epoch [3/5], Loss: 0.2742\
Epoch [4/5], Loss: 0.2432\
Epoch [5/5], Loss: 0.2216\
Training time: 1622.30 seconds\
100%|██████████| 5/5 [27:02<00:00, 324.46s/it]\

Test accuracy: 0.9234\

Classification Report\
              precision    recall  f1-score   support

           0       0.90      0.89      0.90      6000
           1       0.99      0.99      0.99      6000
           2       0.85      0.90      0.88      6000
           3       0.93      0.94      0.94      6000
           4       0.94      0.75      0.83      6000
           5       0.97      1.00      0.98      6000
           6       0.74      0.84      0.79      6000
           7       0.97      0.96      0.97      6000
           8       0.97      1.00      0.98      6000
           9       0.99      0.97      0.98      6000

    accuracy                           0.92     60000
   macro avg       0.93      0.92      0.92     60000\
weighted avg       0.93      0.92      0.92     60000

![confusion matrix](image.png)
