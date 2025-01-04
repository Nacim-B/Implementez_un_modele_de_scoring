# Credit Scoring Model API

## Project Overview
This project implements a credit scoring API that helps predict the probability of credit default for clients. The API is built using FastAPI and serves a machine learning model trained to assess credit risk.

## Project Structure
```
├── .github/            # GitHub configuration files
├── fast_api.py         # Main FastAPI application file
├── test_api.py         # API testing module
├── requirements.txt    # Project dependencies
├── new_client_model.pkl    # Trained model for new clients
├── old_client_model.pkl    # Trained model for existing clients
└── data_test_for_dashboard.csv    # Test dataset for dashboard
```

## Key Components

### API (fast_api.py)
The main API implementation that provides endpoints for:
- Test connection to the API
- Credit scoring predictions for existing clients
- Credit scoring predictions for new clients

### Models
- `new_client_model.pkl`: Machine learning model optimized for new client predictions
- `old_client_model.pkl`: Machine learning model optimized for existing client predictions

### Test Suite (test_api.py)
Contains test cases to ensure API reliability and accuracy.

### Data
The `data_test_for_dashboard.csv` file contains test data to use the models with


## Installation & Deployment

### Local Development

To install the project locally, follow these steps:
- git clone https://github.com/Nacim-B/Implementez_un_modele_de_scoring.git
- cd [repository-name]
- pip install -r requirements.txt

To run the API locally:
-uvicorn fast_api:app --reload

Once the API is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`

### Continuous Integration/Continuous Deployment (CI/CD)

The project implements automated deployment using GitHub Actions and Render.com:

#### GitHub Actions
- Automated testing and deployment pipeline
- Configuration available in `.github/workflows/deploy.yml`

#### Deployment on Render.com
The API is automatically deployed to Render.com through our CI/CD pipeline:
- Create a new Web Service on Render.com
- Connect it to the GitHub repository
- To deploy a new version, go to the GitHub repository
- Navigate to 'Actions' tab
- Select the 'Deploy API to Render' workflow
- Click 'Run workflow' to trigger a manual deployment
- The API will be automatically deployed to Render.com once the workflow completes

The deployed API will be accessible through Render.com's provided URL.
