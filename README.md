# 10-class-genre-classification-model

## Overview
The Model is a 10-class music genre classifier using a Neural Network that predicts the genre of an audio clip among ten genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. The model is built using TensorFlow and trained using supervised learning technique, transfer learning using YAMNet model from Google for embeddings, and GTZAN Dataset(Balanced). 

### Technologies Used:
Python, TensorFlow Keras, YAMNet, NumPy, Pandas, Scikit-Learn, Librosa, and TFLite.

### Techniques Used:
Supervised Learning, Audio Segmentation, Transfer Learning.

### Activation Function:
ReLU, Softmax.

### Model Architecture:
```text
YAMNet Embeddings (1024)
           │
           ▼
    Dense (256, ReLU)
           │
           ▼
   Batch Normalization
           │
           ▼
      Dropout (0.3)
           │
           ▼
    Dense (128, ReLU)
           │
           ▼
 Dense (10, Softmax)
           │
           ▼
 Genre Prediction
```

### Training Result:

#### Results

| Metric | Score |
|----------|----------|
| Accuracy | 75% |
| Macro Precision | 0.75 |
| Macro Recall | 0.75 |
| Macro F1-Score | 0.75 |

#### Best-Performing Genres
| Genre | F1 Score |
|---------|----------|
| Classical | 0.93 |
| Jazz | 0.86 |
| Metal | 0.83 |
| Hip Hop | 0.79 |

#### Challenging Genres
| Genre | F1 Score |
|---------|----------|
| Rock | 0.49 |
| Disco | 0.69 |
| Pop | 0.70 |

#### Confusion Matrix
```text
           blues  classical  country  disco  hiphop  jazz  metal  pop  reggae  
blues        287          0       14     14       0    14      9    0      48   
classical      1        383        2      4       0     7      1    1       1   
country       35          1      296      0       0     9      2   18       1   
disco          0          3        6    250      29     1     10   69       9   
hiphop         2          0       10      6     316     6      3   35      16   
jazz           8         26        8      0       0   355      0    0       0   
metal          2          0        6      5       5     1    337   14       0   
pop            3          0        3     21       8     0     20  312       3   
reggae         6          1        8     21      34     8     13   10     294   
rock          43          7       53      5      13    25     18   37      20 
```

## Usage

### Prerequisites

* Python 3.11 or earlier (Python 3.12+ is not recommended due to compatibility issues with TensorFlow Hub)
* Git

### Clone the Repository

```bash
git clone https://github.com/Austine0829/10-class-genre-classification-model.git
cd 10-class-genre-classification-model
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebooks

#### Inference with the TFLite Model

Open `testing_notebook.ipynb` and run all cells to perform music genre prediction using the exported TensorFlow Lite (TFLite) model.

#### Training the Model

Open `notebook.ipynb` and run all cells to:

* Load and preprocess the GTZAN dataset
* Perform audio segmentation
* Extract YAMNet embeddings
* Train the neural network classifier
* Evaluate model performance
* Generate classification reports and confusion matrices
* Export the trained model to TensorFlow Lite (TFLite) format
