# GPT Model for Marathi Language (10m) 
# Character level
This repository contains a Marathi GPT model designed to perform text generation and autocomplete tasks for the Marathi language. The model is based on the GPT architecture and has been trained on a sample dataset to understand and generate fluent Marathi text.  

## Files in the Repository  
### 1. `Marathi_GPT.py`  
This file contains the complete training code for the Marathi GPT model. It includes:  
- Data preprocessing steps for Marathi text.  
- Implementation of the GPT architecture using a deep learning framework.  
- Training loop, optimizer configuration, and model saving for generating high-quality text. 

### 2. `asker.py`  
This file allows you to interact with the trained model. It includes:  
- Code to load the trained Marathi GPT model.  
- Functionality to input partial text and receive generated text completions in Marathi.  
- Needs Trained Weights https://drive.google.com/file/d/1eaKKawpUeXeFZCGK8Vvim3TG4QtQ6axx/view?usp=drive_link
### 3. `mastered.txt`  
This file contains the example dataset used to train the model. It includes:  
- Marathi text samples for training.  
- Properly formatted data to ensure the model learns the structure and style of Marathi text effectively.  

## Requirements  
- Python 3.8 or above  
- PyTorch with CUDA support  
- Transformers library (Hugging Face)  
- CUDA-enabled GPU (Required for training and efficient text generation)  
- Other dependencies listed in `requirements.txt`  

## How to Use  

1. **Train the Model:**  
   Run `Marathi_GPT.py` to train the model on your dataset. Ensure your system has a CUDA-enabled GPU for efficient training.  

2. **Generate Text Completions:**  
   Use `asker.py` to interact with the model. Provide a partial sentence or phrase in Marathi, and the model will generate a continuation based on its training. A CUDA-enabled GPU is recommended for faster inference.  

3. **Dataset Preparation:**  
   Update `mastered.txt` with your Marathi text data to retrain or fine-tune the model for improved performance.  

## Future Scope  
- Expand the dataset to cover diverse Marathi contexts and styles for enhanced fluency.  
- Optimize the model for more efficient text generation on larger datasets.  
- Extend the project to include tasks like text summarization or translation in Marathi.  

Feel free to contribute or share feedback to improve the model and repository.  
