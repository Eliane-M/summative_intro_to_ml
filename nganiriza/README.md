# Intro to Machine Learning Summative

## Project Title: NGANIRIZA

This project aims to tackel the issue of teenage pregancies in Rwanda by providing a safe space for girls to learn more about their bodies and how to prevent teenage pregancies in general. It will do this by analyisng the data given by a preson and determining if they are at either high, medium or low risk of getting pregnant prematurely. After analysing the data, it will recomend resources to read based on the risk, or even reccomend talking to a specialist in some cases.

The dataset used in this ML project is synthetic due to the lack of collected data on teenage pregancies in Rwanda. Though, the dataset used was synthesised using Rwandan standard parameters like the socio-economic classes ie the Ubudehe categories, education levels and the expected ages to be in certain classes, etc.


## Model Statistics

| Training Instance | Optimizer Used | Regularizer Used (L1 and L2) | Epochs | Early Stopping (Yes or No) | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|---------------|-------------------------------|--------|----------------------------|------------------|--------------|----------|----------|--------|-----------|
| Instance 1      |               |                               | 100    | No                         | 4                | 0.01         | 0.99     | Low: 0.0, Mid: 0.99, High: 0.99 | Low: 0.0, Mid: 0.99, High: 1.00 | Low: 0.0, Mid: 1.00, High: 0.97 |
| Instance 2      | Adam          | L1                            | 100    | Yes                        | 4                | 0.001        | 0.985    | Low: 0.0, Mid: 0.98, High: 0.99 | Low: 0.0, Mid: 0.96, High: 1.00 | Low: 0.0, Mid: 1.00, High: 0.97 |
| Instance 3      | Adam          | L2                            | 100    | Yes                        | 4                | 0.001        | 0.965    | Low: 0.0, Mid: 0.98, High: 0.99 | Low: 0.0, Mid: 0.96, High: 1.00 | Low: 0.0, Mid: 1.00, High: 0.97 |
| Instance 4      | RMSProp       | L1                            | 100    | Yes                        | 4                | 0.001        | 0.965    | Low: 0.0, Mid: 0.98, High: 0.99 | Low: 0.0, Mid: 0.96, High: 1.00 | Low: 0.0, Mid: 1.00, High: 0.97 |
| Instance 5      | SGD           | L1_L2                         | 100    | Yes                        | 4                | 0.001        | 0.96     | Low: 0.0, Mid: 0.98, High: 0.99 | Low: 0.0, Mid: 0.96, High: 1.00 | Low: 0.0, Mid: 1.00, High: 0.97 |



## Running the notebook and the best saved model

Follow these codes to clone the repo

```
git clone https://github.com/Eliane-M/summative_intro_to_ml.git
```

```
cd summative_intro_to_ml
```

create a virtual environment and activate it

```
python -m venv venv
```

```
source venv/Scripts/activate
```

install dependencies

```
pip install requirements.txt
```

Then open the notebook and run the best saved model following the code below

```
import tensorflow as tf

# Load the best saved model
model = tf.keras.models.load_model('saved_models/nn_model_4.h5')

# Make predictions
import numpy as np
sample_input = np.array([[...]])
prediction = model.predict(sample_input)
print("Prediction:", prediction)
```


## Summary of findings from the models

### ML models VS Traditional ML algorithm

In this project, I used Supervised Vector MAchine as the traditional ml algorithm due to its ability to find an optimal hyperplane that best separates the data into different classes in a high-dimensional space. I compared it to Neural network models, which I tuned repeatedly to achieve the best model that could fit my data. The best saved neural network algorith was nn_model_4.h5 with a final training accuracy of 0.8633, and a validation accuracy of 0.95 outperfomed the traditional algorithm which had an accuracy of 0.94. These findings show that the neural network was able to learn more complex patterns in the data, leading to better generalization.
Though the neural network may have benefited from the regularization techniques used and early stopping which would explain the slightly lower validation accuracy, the final accuracies also suggest that deep learning techniques may be more effective for this particular complex dataset.


## Video

[video presentation](https://drive.google.com/file/d/1SpVXw11w0jNoLHsmoM__SueiRbY80ZyK/view?usp=sharing)
