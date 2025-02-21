# Forecasting Basics

## Overview
This repository provides an introduction to **time series forecasting** using machine learning and deep learning techniques. It covers essential concepts, preprocessing steps, and model implementations.

## Features
- Data preprocessing (handling missing values, scaling, and feature engineering)
- Time series forecasting using statistical and ML models
- Implementation of LSTM and Transformer-based models
- Model evaluation and visualization

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch / TensorFlow
- Matplotlib / Seaborn
- Streamlit (for interactive visualization)

## Installation
Clone this repository:
```sh
git clone https://github.com/harshdhiman7/Transformers.git
cd forecasting-basics
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the main script for data preprocessing and model training:
```sh
python pipeline/train.py
```

For interactive visualization, run:
```sh
streamlit run pipeline/app.py
```

## Dataset
Ensure your dataset is placed in the `data/` directory. The expected format:
```
Date, Open Price, High, Low, Close, Volume
2024-01-01, 100, 105, 98, 103, 10000
...
```

## Model Training
Modify `config.yaml` to set hyperparameters, then train the model:
```sh
python train.py --config config.yaml
```

## Results
- Model performance metrics (RMSE, MAE, MDA)
- Visualized predictions vs. actual values

## Contributing
Feel free to open issues or submit pull requests!

## License
This project is licensed under the MIT License.

