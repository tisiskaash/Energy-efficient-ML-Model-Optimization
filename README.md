Energy-Efficient Machine Learning: Maintaining High Accuracy through Hyperparameter Optimisation
About the Project

This project looks at whether machine learning models can reduce their energy consumption through hyperparameter optimisation without losing too much accuracy.

Five machine learning models are tested:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbours (KNN)
Neural Network

The models are tested on five different datasets. The normal (baseline) models are compared with versions where the hyperparameters have been optimised using NSGA-II. The optimisation considers both model accuracy and energy consumption.

main.py runs the machine learning experiments, while visual.py is used to process the results and generate the graphs used for analysis.

Files

The main files in the project are:

main.py
visual.py
README.md

The project also produces result files and graphs:

results/
    heart_results.csv
    spambase_results.csv
    car_evaluation_results.csv
    Dry_Bean_Dataset_results.csv
    UCI_Credit_Card_results.csv

visual/
    generated graphs

The datasets themselves are not included in the submission because they can be downloaded from their original public sources.

Requirements

The project uses Python 3 and the following Python libraries:

pandas
numpy
scipy
scikit-learn
pyRAPL
pymoo
matplotlib
seaborn

They can be installed using:

pip install pandas numpy scipy scikit-learn pyRAPL pymoo matplotlib seaborn

It is recommended to use a virtual environment.

On Linux:

python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn pyRAPL pymoo matplotlib seaborn
Energy Measurement

Energy consumption is measured using pyRAPL.

The energy measurement requires a Linux system with Intel RAPL support and suitable permissions to access the CPU energy counters. Therefore, the energy measurement part of the project may not work on every computer.

The program measures CPU package energy while the models are being trained. The values are converted from microjoules to joules.

Datasets

Five datasets are used in the experiments.

Heart Disease

File:

heart.csv

Source:

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

The target column used by the program is:

HeartDisease
Spambase

File:

spambase.csv

Source:

https://archive.ics.uci.edu/dataset/94/spambase

The target column is:

spam
Car Evaluation

File:

car_evaluation.csv

Source:

https://archive.ics.uci.edu/dataset/19/car+evaluation

The target column is:

class
Dry Bean

File:

Dry_Bean_Dataset.arff

Source:

https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The target column is:

Class
Default of Credit Card Clients

File:

UCI_Credit_Card.csv

Source:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The target column is:

default.payment.next.month

The ID column is removed by the program before training.

Where to Put the Datasets

Create a datasets folder in the same directory as the Python files:

project/
│
├── main.py
├── visual.py
├── README.md
│
└── datasets/
    ├── heart.csv
    ├── spambase.csv
    ├── car_evaluation.csv
    ├── Dry_Bean_Dataset.arff
    └── UCI_Credit_Card.csv

The dataset files are loaded from this folder by main.py.

Data Preprocessing

The program automatically separates the input features from the target variable.

Numerical features are scaled using StandardScaler, while categorical features are converted using OneHotEncoder.

The data is split into training and testing sets using:

Test size: 20%
Random state: 42
Stratification: enabled

The target values are also encoded before the models are trained.

Machine Learning Models

The project tests five models.

Logistic Regression

The optimisation searches different values for:

C
solver
penalty
Random Forest

The optimisation searches:

n_estimators
max_depth
min_samples_split
max_features
Support Vector Machine

The optimisation searches:

C
kernel
gamma
K-Nearest Neighbours

The optimisation searches:

n_neighbors
weights
algorithm
Neural Network

The Neural Network uses scikit-learn's MLPClassifier.

The optimisation searches:

number of neurons
learning rate
solver
activation function
alpha
Hyperparameter Optimisation

NSGA-II is used as a multi-objective optimisation algorithm.

The two objectives are:

Increase accuracy.
Reduce energy consumption.

The optimisation is configured with:

Population size: 10
Generations: 10
Random seed: 42

After NSGA-II produces the possible solutions, the program selects a balanced solution using the normalised accuracy and energy values.

The selected solution is then used as the optimised model for comparison with the baseline model.

Baseline Runs

Each baseline model is trained five times.

For every run, the following are recorded:

Accuracy
Training time
Energy consumption

The median of the five runs is used as the baseline result.

The optimised model is then trained and its accuracy, training time and energy consumption are recorded.

This information is saved to CSV files.

Running the Project

First, make sure the datasets are in the datasets folder.

Run the main experiment with:

python main.py

or:

python3 main.py

This will run the experiments for all five datasets and five models.

The result files produced are:

heart_results.csv
spambase_results.csv
car_evaluation_results.csv
Dry_Bean_Dataset_results.csv
UCI_Credit_Card_results.csv
Generating the Graphs

visual.py uses the result CSV files to create the graphs.

The script expects the CSV files to be inside a results folder:

results/
    heart_results.csv
    spambase_results.csv
    car_evaluation_results.csv
    Dry_Bean_Dataset_results.csv
    UCI_Credit_Card_results.csv

After placing the files there, run:

python visual.py

The script combines the results into final_data.csv and creates a visual folder containing the generated graphs.

The graphs cover areas such as:

accuracy comparisons
energy consumption
energy efficiency
optimisation outcomes
accuracy versus energy
model rankings
total energy footprint
Reproducing the Results

The main random seeds used in the project are set to 42.

The train/test split also uses random_state=42.

The baseline models use five runs, and the median result is reported.

The optimisation uses a population size of 10 and 10 generations.

Energy and training-time results may still vary slightly between runs because they depend on the computer, background processes and system load.

Code and Dataset Provenance

The project uses publicly available Python libraries, including scikit-learn, pymoo and pyRAPL. These libraries provide the machine learning algorithms, optimisation functionality and energy measurement functionality used by the project.

The datasets are also publicly available and their sources are listed above.

The project code was developed for this dissertation using these libraries and their documented APIs. Any third-party code or code snippets used separately from these libraries should be acknowledged in the relevant project documentation.

Author

Karan Karthikeyan

MSc Dissertation Project
University of Stirling
2026
