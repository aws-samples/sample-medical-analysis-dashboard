# Medical Reports Analysis Dashboard - Architecture Diagram

## Solution Overview

The Medical Reports Analysis Dashboard is a comprehensive solution designed to analyze medical reports using AWS services and AI capabilities. The system allows healthcare professionals to upload medical reports, analyze them using natural language processing, and visualize the results through an interactive dashboard.

The solution leverages AWS Bedrock's AI capabilities for natural language understanding, LangChain for document processing, and Streamlit for creating an interactive web interface. The architecture is designed to be scalable, secure, and user-friendly.

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     Medical Reports Analysis Dashboard                                             │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              User Interface Layer                                                  │
│                                                                                                                   │
│  ┌─────────────────────────────┐          ┌─────────────────────────────┐          ┌─────────────────────────────┐│
│  │                             │          │                             │          │                             ││
│  │      Streamlit Web App      │◄────────►│      Chat Interface         │◄────────►│    Data Visualizations      ││
│  │                             │          │                             │          │    (Plotly Charts)          ││
│  │  - Interactive Dashboard    │          │  - User Query Input         │          │  - Bar Charts               ││
│  │  - Report Selection         │          │  - AI Response Display      │          │  - Range Charts             ││
│  │  - Visualization Controls   │          │  - Conversation History     │          │  - Trend Lines              ││
│  │                             │          │                             │          │                             ││
│  └─────────────────────────────┘          └─────────────────────────────┘          └─────────────────────────────┘│
│                                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              Processing Layer                                                      │
│                                                                                                                   │
│  ┌─────────────────────────────┐          ┌─────────────────────────────┐          ┌─────────────────────────────┐│
│  │                             │          │                             │          │                             ││
│  │    LangChain Document       │◄────────►│     Conversational          │◄────────►│     Data Parsing &          ││
│  │    Processing               │          │     Retrieval Chain         │          │     Transformation          ││
│  │                             │          │                             │          │                             ││
│  │  - Document Loading         │          │  - Query Processing         │          │  - CSV Parsing              ││
│  │  - Text Extraction          │          │  - Context Retrieval        │          │  - Reference Range Analysis ││
│  │  - Document Chunking        │          │  - Response Generation      │          │  - Data Normalization       ││
│  │                             │          │                             │          │                             ││
│  └─────────────────────────────┘          └─────────────────────────────┘          └─────────────────────────────┘│
│                                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                AI/ML Layer                                                         │
│                                                                                                                   │
│  ┌─────────────────────────────┐          ┌─────────────────────────────┐          ┌─────────────────────────────┐│
│  │                             │          │                             │          │                             ││
│  │       AWS Bedrock           │◄────────►│     Bedrock Embeddings      │◄────────►│     DocArray In-Memory      ││
│  │      (Claude v2 LLM)        │          │                             │          │     Vector Store            ││
│  │                             │          │                             │          │                             ││
│  │  - Natural Language         │          │  - Text Vectorization       │          │  - Vector Storage           ││
│  │    Understanding            │          │  - Semantic Representation  │          │  - Similarity Search        ││
│  │  - Medical Context          │          │  - Embedding Generation     │          │  - Fast Retrieval           ││
│  │    Comprehension            │          │                             │          │                             ││
│  │                             │          │                             │          │                             ││
│  └─────────────────────────────┘          └─────────────────────────────┘          └─────────────────────────────┘│
│                                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              Storage Layer                                                         │
│                                                                                                                   │
│  ┌───────────────────────────────────────────────────────────┐          ┌───────────────────────────────────────┐ │
│  │                                                           │          │                                       │ │
│  │                     Amazon S3                             │◄────────►│     Conversation Buffer Memory        │ │
│  │               (Medical Reports Storage)                   │          │           (Chat History)              │ │
│  │                                                           │          │                                       │ │
│  │  - CSV Medical Reports                                    │          │  - User Queries                       │ │
│  │  - Blood Test Results                                     │          │  - AI Responses                       │ │
│  │  - Secure Object Storage                                  │          │  - Context Preservation               │ │
│  │  - Versioning & Access Control                            │          │                                       │ │
│  │                                                           │          │                                       │ │
│  └───────────────────────────────────────────────────────────┘          └───────────────────────────────────────┘ │
│                                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           Authentication Layer                                                     │
│                                                                                                                   │
│  ┌───────────────────────────────────────────────────────────┐          ┌───────────────────────────────────────┐ │
│  │                                                           │          │                                       │ │
│  │                      AWS IAM                              │◄────────►│      AWS Credentials Management       │ │
│  │                  (Access Control)                         │          │                                       │ │
│  │                                                           │          │                                       │ │
│  │  - Role-Based Access Control                              │          │  - Access Key Management              │ │
│  │  - Service Permissions                                    │          │  - Secret Key Security                │ │
│  │  - Resource Policies                                      │          │  - Environment Variables              │ │
│  │  - Security Best Practices                                │          │                                       │ │
│  │                                                           │          │                                       │ │
│  └───────────────────────────────────────────────────────────┘          └───────────────────────────────────────┘ │
│                                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **User Interaction**:
   - User selects a medical report from the S3 bucket through the Streamlit interface
   - User asks questions about the report via the chat interface
   - System displays visualizations of the medical data

2. **Document Processing**:
   - S3FileLoader retrieves the selected medical report from Amazon S3
   - LangChain processes the document and extracts structured data
   - Data is transformed for visualization and analysis

3. **AI Analysis**:
   - User queries are processed by the Conversational Retrieval Chain
   - Bedrock LLM (Claude v2) generates responses based on document content
   - Bedrock Embeddings create vector representations for semantic search
   - DocArray stores vector embeddings for efficient retrieval

4. **Visualization**:
   - Processed data is displayed in interactive Plotly charts
   - Bar charts show parameter values
   - Range charts compare values to reference ranges
   - Trend lines track changes over time

5. **Security**:
   - AWS IAM controls access to S3 and Bedrock services
   - AWS credentials are managed securely through environment variables

## Key Components

1. **Streamlit Web Application**:
   - Provides the user interface for the dashboard
   - Handles user interactions and displays visualizations
   - Manages the chat interface for AI interactions

2. **AWS Bedrock**:
   - Provides the Claude v2 LLM for natural language understanding
   - Generates embeddings for semantic search
   - Processes medical context and terminology

3. **LangChain**:
   - Handles document processing and retrieval
   - Manages conversational context and memory
   - Connects the LLM to the document store

4. **Amazon S3**:
   - Stores medical reports in CSV format
   - Provides secure, scalable object storage
   - Enables version control and access management

5. **Plotly**:
   - Creates interactive data visualizations
   - Displays bar charts, range charts, and trend lines
   - Provides hover information and tooltips

## Benefits

- **AI-Powered Analysis**: Leverages AWS Bedrock's advanced LLM capabilities for medical report interpretation
- **Interactive Experience**: Combines chat interface with data visualizations for comprehensive analysis
- **Scalable Architecture**: Built on AWS services for reliable performance and scalability
- **Secure Data Handling**: Implements AWS security best practices for sensitive medical data
- **Flexible Document Processing**: Handles various medical report formats through LangChain's document processing
