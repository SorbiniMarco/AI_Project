import os
from src.train_FashionMNIST import train_FashionMNIST, evaluate_FashionMNIST

def test_FashionMNIST_training():
    test_model_path = "test_FashionMNIST_model.pth"
    # Train for 1 epoch, just to check that the training works
    train_FashionMNIST(1, save_path=test_model_path)
    # Check if the model file is created
    assert os.path.exists(test_model_path), "Model file not found after training."
    # Check if the model file is not empty, and can be loaded
    assert os.path.getsize(test_model_path) > 0, "Model file is empty."
    
    # Load the model to ensure it can be loaded without errors
    import torch
    from src.train_FashionMNIST import SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(test_model_path)) #load parameters from the epoch previously made
    
    # Do a simple inference to check if the model is working
    xin = torch.randn(1, 1, 28, 28) #ranodm input tensor with the same FashionMNIST image dimension
    model.eval()
    with torch.no_grad(): #no gradients calculation
        output = model(xin) #gives the random images to the model
    assert output.shape == (1, 10), "Model output shape is incorrect."
    
    # Clean up the model file after the test
    os.remove(test_model_path)
    os.remove("confusion_matrix.png")
    print("Test passed: Model trained and saved successfully.")
    
def test_FashionMNIST_evaluation():
    # Train the model first
    test_model_path = "test_FashionMNIST_model.pth"
    train_FashionMNIST(1, save_path=test_model_path) #train the model
    evaluate_FashionMNIST(test_model_path) #evaluate the model just trained
    os.remove(test_model_path)