import os
from src.eda import eda

def test_eda():
    eda() # call to the function
    #remove temp file
    os.remove("class_distribution.png")
    os.remove("sample_images.png")
    print("EDA test passed: EDA function executed successfully.")