# Chess Opening Classification

Access the GCP cloud run deployed live classification model via API at https://chess-opening-classification-api-akkxliqoya-uc.a.run.app/classify 

See [test_classify_api_cloud.py](https://github.com/mar1-k/chess_opening_classifier/blob/main/test_classify_api_cloud.py) or [test.ipynb](https://github.com/mar1-k/chess_opening_classifier/blob/main/test.ipynb) for examples on how to use this API

## Problem Description
This project is a machine learning neural network solution to classify chess positions into their corresponding openings based on the board state (represented as a FEN codes). The model analyzes the current board position and classifies it into the Chess opening it thinks that this board most likely had. This is a multi-class classification problem where each board position can potentially match multiple standard chess openings, especially in cases of transpositions. The neural networks were trained via Pytorch. 

## Why work on this?
- Transposition Detection: Helps identify when different move orders lead to the same opening structure
- Game Analysis: Enables quick classification of games by opening type
- Learning Value: I figured this would be a fun and unique way to learn neural networks and use my RTX 3070 for machine learning

## Dataset

The original data used for this project is this 6,000,000 chess games datase from Kaggle https://www.kaggle.com/datasets/arevel/chess-game - These games were processed in order to generate ~2,000,000 board states to train and validate the model with. See the scripts in data/preprocessing folder for more information on how boards were extracted. 

### Key Dataset Features
- ECO (Encyclopedia of Chess Openings) codes
- Opening names
- Board positions (FEN notation)
- Turn numbers

Each record in the dataset maps a specific board position to its corresponding chess opening classification.

## Model Details

### Final Model

The final model implementation is a Dense neural network trained via Pytorch

I selected it because:
   - Optimal hyperparameter configuration
   - Balanced architecture size
   - Effective dropout rate
   - Appropriate learning rate
   - Best overall stability

See notebook.ipynb for full analysis

## Model Performance and conclusions

### Model Performance Metrics

| Metric                | Model 1 (Dense) | Model 2 (Dense) | Model 3 (CNN) | Model 4 (CNN) |
|--------------------- |------------------|------------------|------------------|------------------|
| Final Val Accuracy   | ~13%             | ~13%             | ~13%             | ~12%             |
| Peak Val Accuracy    | ~13%             | ~13%             | ~18%             | ~12%             |
| Final Train Loss     | ~4.1             | ~4.8             | ~1.9             | ~4.9             |
| Final Val Loss       | ~5.1             | ~4.8             | ~9.2             | ~4.8             |
| Training Epochs      | 500              | 500              | 100              | 100              |
| Convergence Speed    | Medium (~150 epochs)| Medium (~100 epochs)| Fast (~10 epochs)| Fast (~20 epochs)|


### Model 1 Results (Dense Neural Network No dropout) 

![image](https://github.com/user-attachments/assets/95dfc9e8-7841-4005-81f6-570dc27a1028)

### Model 2 Results (Dense Neural Network 20% dropout) - Best

![image](https://github.com/user-attachments/assets/43c03bd1-75ab-4917-9a26-9ecbd08616a9)

### Model 3 Results (CNN No dropout) - Worst 

![image](https://github.com/user-attachments/assets/8ae23e00-39f1-4adc-8c5f-dedd51e68876)

### Model 4 Results (CNN 30% dropout)

![image](https://github.com/user-attachments/assets/951e1856-9912-4f41-af2c-3de37340d2fd)


## Model Parameters

Here are the parameters I used for training and tuning the models. 

| Parameter          | Model 1 (Dense)    | Model 2 (Dense)    | Model 3 (CNN)      | Model 4 (CNN)      |
|-------------------|-------------------|-------------------|-------------------|-------------------|
| Architecture      | Dense             | Dense             | CNN               | CNN               |
| Inner Layers      | [512, 256, 128]   | [256, 128, 64]    | Default           | Default           |
| Dropout Rate      | 0.0               | 0.2               | 0.0               | 0.3               |
| Learning Rate     | 0.0001            | 0.001             | 0.001             | 0.01              |
| Batch Size        | 4096              | 4096              | 4096              | 4096              |
| Epochs            | 500               | 500               | 100               | 100               |

## Model Characteristics

### Model 1 (Dense Neural Network)
- Large dense architecture with deeper layers [512, 256, 128]
- No dropout (0.0) which might explain minor overfitting
- Very conservative learning rate (0.0001)
- Shows minor signs of overfitting after epoch 200
- Stable validation accuracy
- Slower but steady learning due to small learning rate

### Model 2 (Dense Neural Network)
- Moderate dense architecture [256, 128, 64]
- Moderate dropout (0.2) helping prevent overfitting
- Balanced learning rate (0.001)
- Best overall stability and performance
- Optimal balance of regularization and model capacity

### Model 3 (Convolutional Neural Network)
- CNN architecture (layers not specified)
- No dropout (0.0) contributing to severe overfitting
- Moderate learning rate (0.001)
- Shows clear signs of overfitting
- Lack of regularization is problematic

### Model 4 (Convolutional Neural Network)
- CNN architecture (layers not specified)
- High dropout (0.3)
- Aggressive learning rate (0.01)
- Stable but suboptimal performance
- High learning rate might be preventing better convergence

## Conclusions

1. **Best Overall Model**: Model 2 (Dense)
   - Optimal hyperparameter configuration
   - Balanced architecture size
   - Effective dropout rate
   - Appropriate learning rate
   - Best overall stability

2. **Worst Overall Model**: Model 3 (CNN)
   - Lack of dropout leading to overfitting
   - CNN architecture may be unnecessarily complex
   - Would benefit from dropout and architecture specifications


## Project Structure
```
chess_opening_classifier/
├── chess_opening_classifier/
│   └── train.cypython-310.pyc
├── classify_opening_api/
│   ├── __pycache__/
│   ├── classify.py
│   ├── dense_neural_network_v2.bin #For ease of dockerizing
│   ├── dockerfile
│   ├── label_encoder.pkl #For ease of dockerizing
│   └── requirements.txt
├── cloud_deployment/terraform/
│   ├── .terraform/
│   ├── .terraform.lock.hcl
│   ├── main.tf
│   ├── terraform.tfstate
│   ├── terraform.tfstate.backup
│   └── variables.tf
├── data/
│   ├── preprocessing/
│   │   ├── process_boards.py
│   │   └── process_raw_csv.py
│   ├── .gitattributes
│   └── chess_boards_sample.csv
├── models/
│   ├── cnn_v1.bin
│   ├── cnn_v2.bin
│   ├── dense_neural_network_v1.bin
│   ├── dense_neural_network_v2.bin
│   └── label_encoder.pkl
├── .gitignore
├── LICENSE
├── notebook.ipynb
├── Pipfile
├── Pipfile.lock
├── README.md
├── test_classify_api_cloud.py
├── test_classify_api_local.py
├── test.ipynb
└── train.py
```

##  Technology and Libraries Used

```
- python 3.10
- pipenv
- chess
- polars
- matplotlib
- seaborn
- numpy
- pytorch
- scikit-learn
- fast api
- Docker
- Google Cloud Platform
- GitLFS
- Jupyter notebook
```


# How to test it out yourself

You can directly make POST requests to the live deployed classification API at https://chess-opening-classification-api-akkxliqoya-uc.a.run.app/classify 

See [test_classify_api_cloud.py](https://github.com/mar1-k/chess_opening_classifier/blob/main/test_classify_api_cloud.py) or [test.ipynb](https://github.com/mar1-k/chess_opening_classifier/blob/main/test.ipynb) for examples on how to use this API

You may also deploy the neural network on your local machine via docker at then test via test_classify_api_local.py

Training the model locally is also possible but please see the note below.

**NOTE**: 

**You may run into issues running notebook.ipynb and train.py if you have insufficient hardware, I trained these neural networks on an RTX 3070. I ensured that at least train.py should run and train via pytorch using CPU (THIS COULD TAKE A VERY LONG TIME DEPENDING ON YOUR HARDWARE). 
**

**Ensure that you have installed git LFS otherwise the csv will have to be downloaded manually
**

## Environment Setup

Clone the repository:
```
git clone https://github.com/mar1-k/chess-opening-classifier.git
cd chess-opening-classifier
```

Make sure you have python 3.10 installed on your system and install pipenv
```
pip install pipenv
```

Install dependencies
```
pipenv install
```

Activate the virtual environment
```
pipenv shell
```

Get the environment name so that you may install it via ipykernel:
```
pipenv --venv
```

Install the pipenv environment to ipykernel so that it may be used with the notebook - you need the output of step 5 for the name (example: chess_opening_classifier-Nbaf3HzU)
```
python -m ipykernel install --user --name=<YOUR-virtualenv-name->
```

## Usage and Evaluation Instructions

*Please Ensure that you have installed git LFS before cloning otherwise the csv will have to be downloaded manually*

### Running the notebook

NOTE: You may run into issues running the notebook and train.py if you have insufficient hardware, I trained these neural networks on an RTX 3070. I ensured that at least train.py should run and train via pytorch using CPU (THIS COULD TAKE A VERY LONG TIME DEPENDING ON YOUR HARDWARE). 

1. Ensure the dataset `chess_boards_sample.csv` is in the `data/` directory and is not just a Git LFS placeholder file
2. Inside the notebook ensure that you select the kernel associated with the pipenv installations
3. Open and run notebook.ipynb - ensure that you have installed all dependencies from environment setup
```
jupyter notebook notebook.ipynb
```
4. Run the notebook

### Training the Model

NOTE: You may run into issues running the notebook and train.py if you have insufficient hardware, I trained these neural networks on an RTX 3070. I ensured that at least train.py should run and train via pytorch using CPU (THIS COULD TAKE A VERY LONG TIME DEPENDING ON YOUR HARDWARE). 

1. Ensure the dataset `chess_boards_sample.csv` is in the `data/` directory
2. Run the training script:
```
python train.py
```
This will generate the model binary binaries and place them in the `models/` directory.


## Serve the model locally via docker

```
#Navigate to the classify_opening_api folder of this project
cd classify_opening_api

# Build the Docker image - this will use the binary files found locally in the directory
docker build -t chess-opening-classifier .

# Run the container
docker run -p 8000:8000 chess-opening-classifier
```

Once the container is running you can simply make POST requests to localhost:8000/classify - Alternatively, just run test.ipynb or test_classify_api_local.py

## Cloud Deployment

The application is deployed using Docker and Google Cloud Run. Access the live API at https://chess-opening-classification-api-akkxliqoya-uc.a.run.app/classify 

The application is deployed on Google Cloud Run, providing scalable and serverless execution. This is faciliated through Terraform for ease of reproducibility and maintaining IaC.

Deployment instructions:

1. Create a GCP cloud project

2. Create a Terraform Service Account and grant it appropriate permissions - Editor and Cloud Run Admin

3. Create a Service Account Key
- From the console, click on your newly created service account and navigate to the "KEYS" tab
- Click on "Add Key" to Create a key file for this service account
- Save the key file somewhere safe and accessible on the system that you will be using Terraform from

4. Enable Necessary APIs
Terraform will need the following GCP APIs enabled for this project, please enable them in your project
https://console.developers.google.com/apis/api/run.googleapis.com
https://console.cloud.google.com/apis/library/cloudresourcemanager.googleapis.com

6. Setup Terraform Variables file 
- Navigate to the Terraform folder of this project and ensure that the Terraform variables file `variables.tf` has the correct project name and GCP key file path information

6. Push Docker containers to GCR
```
gcloud config set project <YOUR PROJECT NAME>
gcloud auth login
gcloud auth configure-docker

#In the /classify_opening_api directory
docker build -t chess-opening-classifier .
docker tag chess-opening-classifier:latest gcr.io/chess-opening-classifier/chess-opening-classification-api:latest
docker push gcr.io/chess-opening-classifier/chess-opening-classification-api:latest
```

7. Terraform init and apply
- While in the Terraform folder of this project run `terraform init` and then `terraform apply`
- Review the Terraform plan and type `yes` if everything looks good, you should see `Plan: 2 to add, 0 to change, 0 to destroy.

8. Navigate to the frontend url provided by Terraform

9. Enjoy your deployment! Don't forget to `terraform destroy` when done

## Acknowledgments

This has been a capstone project for the 2024-2025 cohort of DataTalks.Club Machine Learning Zoomcamp. And yet again deeply grateful to Alexey Grigorev and the team for making this quality course available completely free and painstaklingly going through the effort of making it all possible.

I am also thankful to
- Lichess.org for the original raw dataset
- Raw Kaggle dataset curator A.revel 
