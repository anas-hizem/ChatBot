
# Chatbot using PyTorch and NLTK

This repository contains a simple chatbot developed using PyTorch and NLTK. The chatbot uses a neural network to classify user input based on defined intents and generate appropriate responses.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Running the Chatbot](#running-the-chatbot)
- [Customization](#customization)
- [License](#license)

## Overview
This chatbot is trained on a set of predefined intents and responses, allowing it to identify user input and respond accordingly. The neural network model is created using PyTorch, and text processing is handled by NLTK.

## Features
- **Intent Classification**: Uses a neural network to classify input messages into intents.
- **Customizable Intents**: Add new intents in `intents.json`.
- **Bag of Words Model**: Employs a bag-of-words model for text vectorization.

## Requirements
- Python 3.x
- [PyTorch](https://pytorch.org/)
- [NLTK](https://www.nltk.org/)
- Additional libraries in `requirements.txt` (if included)

## Project Structure
- `intents.json`: JSON file containing predefined intents, patterns, and responses.
- `model.py`: Defines the neural network model.
- `nltk_utils.py`: Utility functions for tokenizing, stemming, and creating a bag of words.
- `train.py`: Script to train the model on the dataset defined in `intents.json`.

## Setup and Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
   Make sure that `PyTorch` and `NLTK` are installed. For PyTorch installation, follow [PyTorch’s official guide](https://pytorch.org/get-started/locally/).

3. **Download NLTK data**:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Training the Model
To train the model, run:
```bash
python train.py
```
This will load data from `intents.json`, preprocess it, and train the neural network model. The trained model parameters are saved to `model.pth`.

### Example output during training:
```
Input size: 45, All words: 45
Output size: 6, Tags: ['greeting', 'goodbye', 'age', 'name', 'shop', 'hours']
Epoch [50/256], Loss: 1.2234
...
Training complete.
Model saved to model.pth
```

## Running the Chatbot
Once the model is trained, you can integrate it into a script to handle real-time input. Create a `chat.py` script (sample provided below) to interact with the model:

```python
import torch
import json
import random
import nltk_utils
from model import NeuralNet

# Load intents and model
with open('intents.json', 'r') as f:
    intents = json.load(f)

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Chat loop
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # Preprocess and predict
    words = nltk_utils.tokenize(sentence)
    X = nltk_utils.bag_of_words(words, all_words)
    output = model(torch.tensor(X).float())
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]

    # Find and print response
    for intent in intents['intents']:
        if tag == intent["tag"]:
            print("Bot:", random.choice(intent['responses']))
```

## Customization
You can add new intents and responses by editing the `intents.json` file. Structure your file like this:
```json
{
    "tag": "tag_name",
    "patterns": ["example question", "another query"],
    "responses": ["response 1", "response 2"]
}
```
After updating `intents.json`, retrain the model with `python train.py` to include new intents.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
