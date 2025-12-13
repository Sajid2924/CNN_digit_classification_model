# MNIST CNN Classification with Hyperparameter Tuning

Project Overview
----------------
This project implements a Convolutional Neural Network (CNN) for handwritten digit
classification on the MNIST dataset using PyTorch.

The project is intentionally divided into two parts:
1) A manually designed baseline CNN
2) A CNN optimized using Optuna for hyperparameter tuning

The main objective is to demonstrate a complete deep learning workflow rather than
chasing marginal accuracy improvements.

Project Structure
-----------------
MNIST-CNN/
|
|-- baseline_CNN.py        -> Manually designed CNN (baseline model)
|-- optuna_tuning.py       -> CNN hyperparameter tuning using Optuna
|-- data/                 -> MNIST dataset (auto-downloaded)
|-- README.md

Files Description
-----------------

baseline_CNN.py
---------------
This file contains a manually designed CNN architecture with fixed hyperparameters.
The model is trained using SGD optimizer with weight decay.

Test Accuracy Achieved: ~99.3%

Purpose:
Demonstrates the ability to design an effective CNN without using automated tuning tools.

optuna_tuning.py
----------------
This file performs hyperparameter tuning using Optuna.
The following parameters are explored:
- Number of convolution blocks
- Number of convolution filters
- Number of fully connected layers
- Learning rate
- Optimizer type (Adam, SGD, RMSprop)
- Dropout rate
- Weight decay
- Batch size

Best Test Accuracy Achieved: ~99.25%

Purpose:
Demonstrates understanding of automated hyperparameter optimization and search-space design.

Results Summary
---------------
Baseline CNN Accuracy       : 99.3%
Optuna Tuned CNN Accuracy   : 99.25%

Why Tuned Model Did Not Outperform Baseline
-------------------------------------------
- MNIST is a simple and saturated dataset
- Many reasonable CNN architectures already exceed 99% accuracy
- Small differences are due to randomness and sampling variance
- Hyperparameter tuning improves robustness, not guaranteed peak accuracy

Important Insight:
Hyperparameter tuning does not always outperform a well-designed baseline on simple datasets.

Experimental Notes
------------------
- Dataset split used: 80% training, 20% testing
- Test set is used during tuning for simplicity
- In production systems, a separate validation set should be used
- This choice is documented intentionally for learning clarity

Installation Requirements
-------------------------
Python version: 3.8 or higher

Required packages:
- torch
- torchvision
- numpy
- scikit-learn
- optuna

Installation Command:
pip install torch torchvision numpy scikit-learn optuna

How to Run
----------
To run the baseline CNN:
python baseline_CNN.py

Expected Output:
accuracy : ~99.3 %

To run hyperparameter tuning:
python optuna_tuning.py

Expected Output:
accuracy : 0.9925
best accuracy achieved for parameters : {...}

Note:
Hyperparameter tuning may take several minutes depending on system performance.

Key Learnings
-------------
- Implemented a full CNN training pipeline using PyTorch
- Designed CNN architectures manually
- Built custom Dataset and DataLoader
- Implemented training and evaluation loops
- Applied Optuna for hyperparameter tuning
- Understood why tuning may not always outperform a strong baseline
- Learned that accuracy alone is not the goal in ML engineering

Future Improvements
-------------------
- Add a validation split
- Plot training and validation curves
- Add data augmentation
- Save and reload trained models
- Extend the project to CIFAR-10 or Fashion-MNIST

Final Conclusion
----------------
This project demonstrates a professional deep learning workflow combining
manual model design, automated hyperparameter tuning, and honest result analysis.

The focus is on methodology, robustness, and understanding rather than
chasing marginal accuracy gains on saturated datasets.
