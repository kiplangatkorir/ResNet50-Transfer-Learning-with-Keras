# ResNet50-Transfer-Learning-with-Keras

## Overview

This repository contains code and resources for performing transfer learning using the ResNet50 architecture with the Keras deep learning library. Transfer learning leverages the pre-trained weights of a model trained on a large dataset (such as ImageNet) to adapt it to a new, smaller dataset. This technique is useful for training deep neural networks on datasets where labeled data is limited.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

ResNet50 is a 50-layer deep convolutional neural network that has demonstrated state-of-the-art performance in various computer vision tasks. This project demonstrates how to use ResNet50 for transfer learning, adapting the network to classify images from a new dataset.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- OpenCV (optional, for image preprocessing)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/ResNet50-Transfer-Learning-with-Keras.git
   cd ResNet50-Transfer-Learning-with-Keras

2. Create a virtual environment and activate it:
```sh
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```sh
pip install -r requirements.txt
```

## Dataset Preparation

1. Prepare your dataset in the following structure:
```markdown
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
└── validation/
    ├── class1/
    ├── class2/
    └── ...
```
2. Update the data_dir variable in the train.py script to point to your dataset directory.

## Usage
1. ## Training the model:
Run the train.py script to start training the model:
```sh
python train.py
```
The script will:

-Load the ResNet50 model pre-trained on ImageNet.

-Replace the top layers with new layers suitable for your dataset.

-Train the new layers while freezing the pre-trained layers.

-Optionally, fine-tune the pre-trained layers for better performance.

2. ## Evaluating the model:
After training, the model can be evaluated using the validation set. The train.py script will automatically save the best model based on validation accuracy.

3. ## Making predictions:
Use the predict.py script to make predictions on new images:
```sh
python predict.py --image_path path/to/your/image.jpg
```

## Results
The training process will output the model's performance metrics, such as accuracy and loss, at each epoch. The best model will be saved to the saved_models directory. You can visualize the training history using the plot_history.py script to see how the model's performance improves over time.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or improvements.

-Fork the repository.
-Create your feature branch (git checkout -b feature/your-feature).
-Commit your changes (git commit -am 'Add your feature').
-Push to the branch (git push origin feature/your-feature).
-Create a new Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.






