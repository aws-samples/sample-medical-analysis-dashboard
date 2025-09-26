# Medical Reports Analysis Dashboard

An interactive dashboard that analyzes medical reports using AWS Bedrock's AI capabilities, LangChain's document processing, and Streamlit's visualization features.

## Prerequisites

- AWS account with Bedrock access enabled and appropriate IAM permissions
- AWS Access Key and Secret Access Key
- S3 bucket for storing medical reports
- Python 3.9 or higher with pip

We'll use Python virtual environment (venv) for this project to ensure a clean, isolated environment. While we're using Python's built-in venv, alternatives like conda are also available.

## Deployment

1. Clone the repository:
```bash
git clone gitlab.aws.dev:my-group-aditranj/medical-analysis-dashboard.git
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv

# For Mac/Linux
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

3. Update pip and install dependencies:
```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Configure AWS credentials:
```bash
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
```

5. Our repository contains two sample files:
- `blood_test.csv`: Complete blood work with 15 parameters
- `basic_test.csv` with basic parameters:
```csv
Parameter,Value,Reference_Range,Unit
Hemoglobin,13.8,13.5-17.5,g/dL
RBC,4.8,4.5-5.9,million/µL
WBC,8500,4000-11000,cells/µL
Glucose,92,70-100,mg/dL
Creatinine,1.0,0.7-1.3,mg/dL
```

6. Upload sample files to S3:
```bash
aws s3 cp blood_test_2024.csv s3://BUCKET_NAME/
aws s3 cp basic_test_2024.csv s3://BUCKET_NAME/
```

7. Update S3 bucket name in `app.py` Line 68:
```python
BUCKET_NAME = "YOUR_S3_BUCKET_NAME"  # Line 68
```

8. Run the application:
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## Dependencies

```
boto3
streamlit
unstructured
langchain-aws
langchain-community
pandas
plotly
numpy
docarray
```

These packages handle AWS integration, web interface, data processing, and visualizations. They'll be installed in the virtual environment during deployment.
