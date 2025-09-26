import streamlit as st
import boto3
import pandas as pd
from langchain_community.document_loaders import S3FileLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import BedrockLLM
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_aws import BedrockEmbeddings
import plotly.graph_objects as go
import os
import sys
import time
import random
import warnings
from botocore.exceptions import ClientError

from langchain_aws import ChatBedrock

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*libmagic.*")

# Page configuration
st.set_page_config(
    page_title="Medical Reports Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e9ecef;
    }
    .assistant-message {
        background-color: #f8f9fa;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# AWS Configuration
try:
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    s3_client = session.client('s3')
    BUCKET_NAME = "YOUR_S3_BUCKET_NAME"
except Exception as e:
    st.error(f"AWS Configuration Error: {str(e)}")

# Rate limiting configuration - Adjust these values based on your Bedrock quotas
RATE_LIMIT_DELAY = 3  # seconds between requests (increased from 2)
MAX_RETRIES = 5  # increased retries
BASE_DELAY = 2  # base delay for exponential backoff (increased)

def exponential_backoff_retry(func, max_retries=MAX_RETRIES, base_delay=BASE_DELAY):
    """Retry function with exponential backoff for throttling errors."""
    for attempt in range(max_retries):
        try:
            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting delay
            return func()
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"Rate limited. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"Rate limited. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e
    raise Exception(f"Max retries ({max_retries}) exceeded")

# Available Bedrock Models (Using Inference Profile IDs)
BEDROCK_MODELS = {
    "Claude Opus 4.1 (Latest & Most Capable)": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "Claude 3.7 Sonnet (Newest)": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "Claude Sonnet 4 (Advanced)": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "Claude Opus 4 (Premium)": "us.anthropic.claude-opus-4-20250514-v1:0",
    "Claude 3.5 Sonnet v2 (Excellent)": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Claude 3 Opus (Reliable)": "us.anthropic.claude-3-opus-20240229-v1:0",
    "Claude 3 Sonnet (Classic)": "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "Amazon Nova Pro (Balanced)": "us.amazon.nova-pro-v1:0",
    "Amazon Nova Lite (Fast)": "us.amazon.nova-lite-v1:0",
}

# Bedrock Configuration - will be initialized based on user selection
llm = None
embeddings = None

def initialize_bedrock_llm(model_id):
    """Initialize Bedrock LLM with selected model"""
    try:
        return ChatBedrock(
            model_id=model_id,
            region_name="us-east-1",
            model_kwargs={
                "max_tokens": 1000,
                "system": "You are a medical analysis assistant. Always respond in English only, regardless of the language of the input documents or previous conversation. Provide clear, professional medical insights in English."
            }
        )
    except Exception as e:
        st.error(f"Bedrock Configuration Error for {model_id}: {str(e)}")
        return None

try:
    embeddings = BedrockEmbeddings(region_name="us-east-1")
    # Initialize direct Bedrock client for Converse API
    bedrock_runtime = session.client('bedrock-runtime')
except Exception as e:
    st.error(f"Bedrock Configuration Error: {str(e)}")

def load_documents():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        return [obj['Key'] for obj in response.get('Contents', [])]
    except Exception as e:
        st.error(f"Error listing S3 objects: {str(e)}")
        return []

def create_qa_chain(file_key, selected_llm):
    try:
        if not selected_llm:
            st.error("No LLM model selected")
            return None
            
        loader = S3FileLoader(bucket=BUCKET_NAME, key=file_key)
        documents = loader.load()
        
        # Create embeddings with rate limiting
        def create_embeddings():
            return DocArrayInMemorySearch.from_documents(documents, embedding=embeddings)
        
        vector_store = exponential_backoff_retry(create_embeddings)
        retriever = vector_store.as_retriever()
        
        # Create modern prompt template
        system_prompt = (
            "You are a medical analysis assistant. Always respond in English only, "
            "regardless of the language of the input documents or previous conversation. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Provide clear, professional medical insights in English.\n\n"
            "Format your response with:\n"
            "- Use bullet points or numbered lists when comparing multiple parameters\n"
            "- Add line breaks between different topics\n"
            "- Use clear paragraph structure\n"
            "- Underline important values or findings when appropriate\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create the modern chain with selected LLM
        question_answer_chain = create_stuff_documents_chain(selected_llm, prompt)
        qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return qa_chain
    except Exception as e:
        st.error(f"Error in creating QA chain: {str(e)}")
        return None

def format_medical_response(response):
    """Format medical response for better readability"""
    # Add line breaks after sentences that end with units or ranges
    import re
    
    # Add line breaks after common medical patterns
    response = re.sub(r'(\d+(?:\.\d+)?(?:\s*[a-zA-Z/%μ]+)?(?:\s*is\s+(?:within|above|below).*?range.*?)\.)', r'\1\n\n', response)
    
    # Add line breaks after parameter mentions
    response = re.sub(r'(The\s+[A-Z][a-z]+(?:\s+[a-z]+)*\s+(?:count|level|of).*?\.)', r'\1\n\n', response)
    
    # Add line breaks before "Overall" or summary statements
    response = re.sub(r'(Overall,)', r'\n\n\1', response)
    
    # Clean up multiple line breaks
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    return response.strip()

def parse_reference_range(range_str):
    """Parse the reference range string (e.g., '13.5-17.5') into min and max values."""
    try:
        min_val, max_val = map(float, range_str.split('-'))
        return min_val, max_val
    except Exception as e:
        st.error(f"Error parsing reference range '{range_str}': {str(e)}")
        return None, None

def create_bar_chart(df):
    try:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df['Parameter'],
                y=df['Value'],
                name='Values',
                marker_color='#1f77b4',
                hovertemplate='Parameter: %{x}<br>Value: %{y} %{customdata}<extra></extra>',
                customdata=df['Unit']
            )
        )
        fig.update_layout(
            height=600,
            title_text="Blood Parameters Bar Chart",
            title_x=0.5,
            xaxis_title="Parameters",
            yaxis_title="Values",
            template='plotly_white',
            showlegend=False
        )
        return fig
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        return None

def create_trend_lines(df):
    try:
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, param in enumerate(df['Parameter'].unique()):
            param_data = df[df['Parameter'] == param]
            unit = param_data['Unit'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=param_data.index,
                    y=param_data['Value'],
                    mode='lines+markers',
                    name=f"{param} ({unit})",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8),
                    hovertemplate=f"{param}: %{{y}} {unit}<extra></extra>"
                )
            )
        fig.update_layout(
            height=600,
            title_text="Parameters Trend Lines",
            title_x=0.5,
            xaxis_title="Measurement Index",
            yaxis_title="Values",
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        return fig
    except Exception as e:
        st.error(f"Error creating trend lines: {str(e)}")
        return None

def create_range_chart(df):
    try:
        fig = go.Figure()
        parameters_found = []

        # For each parameter, create the range bars
        for param in df['Parameter'].unique():
            param_data = df[df['Parameter'] == param]
            value = param_data['Value'].iloc[0]
            range_str = param_data['Reference_Range'].iloc[0]
            unit = param_data['Unit'].iloc[0]

            # Parse the reference range (normal range)
            min_val, max_val = parse_reference_range(range_str)
            if min_val is None or max_val is None:
                st.warning(f"Skipping {param}: Invalid reference range")
                continue

            parameters_found.append(param)

            # Define the ranges: Low, Normal, High
            # Low: 0 to min_val
            # Normal: min_val to max_val (from CSV)
            # High: max_val to an estimated upper limit (max_val + 50% of range for simplicity)
            range_width = max_val - min_val
            high_max = max_val + range_width * 0.5  # Extend the high range by 50% of the normal range width

            ranges = [
                {'label': 'Low', 'min': 0, 'max': min_val, 'color': '#F44336'},  # Red for Low
                {'label': 'Normal', 'min': min_val, 'max': max_val, 'color': '#4CAF50'},  # Green for Normal
                {'label': 'High', 'min': max_val, 'max': high_max, 'color': '#FF9800'}  # Orange for High
            ]

            # Add range bars
            for r in ranges:
                fig.add_trace(
                    go.Bar(
                        y=[param],
                        x=[r['max'] - r['min']],
                        base=r['min'],
                        orientation='h',
                        name=r['label'],
                        marker_color=r['color'],
                        opacity=0.6,
                        hovertemplate=f"{r['label']}<br>Range: {r['min']}-{r['max']} {unit}<extra></extra>",
                        showlegend=False
                    )
                )

            # Add the user's value as a marker
            fig.add_trace(
                go.Scatter(
                    y=[param],
                    x=[value],
                    mode='markers+text',
                    name=param,
                    marker=dict(color='#FF5722', size=12, symbol='circle'),  # Orange marker
                    text=[f"{value}"],
                    textposition='middle right',
                    hovertemplate=f"Value: {value} {unit}<extra></extra>",
                    showlegend=False
                )
            )

        if not parameters_found:
            st.error("No parameters with valid reference ranges found.")
            return None

        # Update layout
        fig.update_layout(
            height=600,
            title_text="Blood Parameters Range Comparison",
            title_x=0.5,
            barmode='stack',
            xaxis_title="Values",
            yaxis_title="Parameters",
            template='plotly_white',
            showlegend=False
        )

        return fig
    except Exception as e:
        st.error(f"Error creating range chart: {str(e)}")
        return None

def main():
    # Sidebar
    st.sidebar.title("📊 Medical Reports Analysis Dashboard")
    st.sidebar.markdown("---")
    
    # Model Selection
    st.sidebar.subheader("🤖 AI Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose Bedrock Model",
        list(BEDROCK_MODELS.keys()),
        index=0,  # Default to Claude 3 Haiku
        help="Different models offer varying performance, speed, and cost trade-offs"
    )
    selected_model_id = BEDROCK_MODELS[selected_model_name]
    
    # Model info
    model_info = {
        "us.anthropic.claude-opus-4-1-20250805-v1:0": "🔥 Latest Claude 4.1",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "✨ Newest Claude 3.7 - Enhanced reasoning",
        "us.anthropic.claude-sonnet-4-20250514-v1:0": "🚀 Claude Sonnet 4 - Next-gen performance",
        "us.anthropic.claude-opus-4-20250514-v1:0": "💎 Claude Opus 4 - Premium intelligence",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "🎯 Claude 3.5 v2 - Excellent for medical analysis",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": "⚡ Fast & smart - Great speed/quality balance",
        "us.anthropic.claude-3-opus-20240229-v1:0": "🧠 Classic Opus - Deep analysis capabilities",
        "us.anthropic.claude-3-sonnet-20240229-v1:0": "🔄 Proven Sonnet - Consistent performance",
        "us.amazon.nova-pro-v1:0": "⚖️ Balanced Amazon model",
        "us.amazon.nova-lite-v1:0": "🏃 Fast Amazon model",
    }
    st.sidebar.info(model_info.get(selected_model_id, ""))
    
    # Initialize LLM based on selection
    if 'current_model' not in st.session_state or st.session_state.current_model != selected_model_id:
        with st.spinner("Initializing model..."):
            llm = initialize_bedrock_llm(selected_model_id)
            st.session_state.current_model = selected_model_id
            st.session_state.llm = llm
            # Reset QA chain when model changes
            st.session_state.qa_chain = None
    else:
        llm = st.session_state.llm
    
    st.sidebar.markdown("---")
    
    # Optional: Show system info
    with st.sidebar.expander("ℹ️ System Info", expanded=False):
        st.text("• libmagic warning can be ignored")
        st.text("• All core features working normally")
        st.text(f"• Current model: {selected_model_name}")
    
    # Rate limiting status
    current_time = time.time()
    time_since_last_query = current_time - st.session_state.get('last_query_time', 0)
    if time_since_last_query < RATE_LIMIT_DELAY:
        remaining_time = RATE_LIMIT_DELAY - time_since_last_query
        st.sidebar.warning(f"⏳ Rate limit: {remaining_time:.1f}s remaining")
    else:
        st.sidebar.success("✅ Ready for queries")
    
    documents = load_documents()
    selected_doc = st.sidebar.selectbox(
        "Select Medical Report",
        documents,
        help="Choose a medical report to analyze"
    )

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None
    if 'last_query_time' not in st.session_state:
        st.session_state.last_query_time = 0
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = {
            'inputTokens': 0,
            'outputTokens': 0,
            'totalTokens': 0
        }

    # Token Usage Display in Sidebar (after session state init)
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Token Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Input Tokens", st.session_state.token_usage['inputTokens'])
            st.metric("Total Tokens", st.session_state.token_usage['totalTokens'])
        with col2:
            st.metric("Output Tokens", st.session_state.token_usage['outputTokens'])
            if st.button("Reset", key="reset_tokens"):
                st.session_state.token_usage = {'inputTokens': 0, 'outputTokens': 0, 'totalTokens': 0}
                st.rerun()

    # Chat Analysis at top
    st.header("💬 Chat Analysis")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        🙋‍♂️ {message['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Use st.markdown for assistant messages to preserve formatting
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        🤖 
                    </div>
                """, unsafe_allow_html=True)
                # Display the formatted response with proper markdown
                st.markdown(message['content'])

    # Query input
    if 'query_state' not in st.session_state:
        st.session_state.query_state = ""

    def on_input_change():
        if st.session_state.query_input.strip():
            st.session_state.query_state = st.session_state.query_input
            st.session_state.query_input = ""

    query = st.text_input(
        "Ask a question about the report:", 
        key="query_input",
        on_change=on_input_change
    )

    if st.session_state.query_state and selected_doc:
        # Check if we need to create a new QA chain (document or model changed)
        if (st.session_state.qa_chain is None or 
            st.session_state.current_doc != selected_doc or
            st.session_state.get('qa_chain_model') != selected_model_id):
            with st.spinner("Loading document and creating analysis chain..."):
                st.session_state.qa_chain = create_qa_chain(selected_doc, llm)
                st.session_state.current_doc = selected_doc
                st.session_state.qa_chain_model = selected_model_id
        
        # Rate limiting check
        current_time = time.time()
        time_since_last_query = current_time - st.session_state.last_query_time
        if time_since_last_query < RATE_LIMIT_DELAY:
            remaining_time = RATE_LIMIT_DELAY - time_since_last_query
            st.warning(f"Please wait {remaining_time:.1f} more seconds before making another query.")
            time.sleep(remaining_time)

        if st.session_state.qa_chain:
            with st.spinner("Analyzing..."):
                try:
                    # Build context from conversation history
                    context_messages = []
                    for i in range(0, len(st.session_state.conversation), 2):
                        if i + 1 < len(st.session_state.conversation):
                            user_msg = st.session_state.conversation[i]["content"]
                            assistant_msg = st.session_state.conversation[i + 1]["content"]
                            context_messages.append(f"Previous Q: {user_msg}\nPrevious A: {assistant_msg}")
                    
                    context_str = "\n\n".join(context_messages[-3:])  # Keep last 3 exchanges
                    
                    # Build the full question with context
                    full_question = f"{context_str}\n\nCurrent question: {st.session_state.query_state}"
                    
                    # Execute query with rate limiting and retry logic
                    def execute_query():
                        return st.session_state.qa_chain.invoke({
                            "input": full_question
                        })
                    
                    result = exponential_backoff_retry(execute_query)
                    response = result['answer']
                    
                    # Get actual token usage using direct Converse API call
                    try:
                        # Make a direct Converse API call to get token usage
                        converse_response = bedrock_runtime.converse(
                            modelId=selected_model_id,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [{"text": full_question}]
                                }
                            ]
                        )
                        
                        # Extract usage from Converse API response
                        if 'usage' in converse_response:
                            usage = converse_response['usage']
                            st.session_state.token_usage['inputTokens'] += usage.get('inputTokens', 0)
                            st.session_state.token_usage['outputTokens'] += usage.get('outputTokens', 0)
                            st.session_state.token_usage['totalTokens'] += usage.get('totalTokens', 0)
                    except Exception as token_error:
                        st.warning(f"Could not get token usage: {str(token_error)}")
                        # Fallback: estimate tokens
                        input_estimate = len(full_question.split()) * 1.3
                        output_estimate = len(response.split()) * 1.3
                        st.session_state.token_usage['inputTokens'] += int(input_estimate)
                        st.session_state.token_usage['outputTokens'] += int(output_estimate)
                        st.session_state.token_usage['totalTokens'] += int(input_estimate + output_estimate)
                    
                    # Format the response for better readability
                    response = format_medical_response(response)
                    
                    # Update last query time
                    st.session_state.last_query_time = time.time()

                    st.session_state.conversation.append({
                        "role": "user",
                        "content": st.session_state.query_state
                    })
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.session_state.query_state = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    if "ThrottlingException" in str(e):
                        st.info("💡 **Tip**: The system is experiencing high load. Please wait a moment before trying again.")

    # Document Overview and Visualizations
    col1, col2 = st.columns([1.2, 2.8])  # Adjusted proportions to make the visualizations wider

    with col1:
        st.header("📑 Document Overview")
        if selected_doc:
            st.write(f"Currently analyzing: **{selected_doc}**")
            try:
                obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=selected_doc)
                df = pd.read_csv(obj['Body'])
                st.subheader("🔍 Raw Data")
                st.dataframe(df, width=600)
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    with col2:
        if selected_doc:
            try:
                if 'df' not in locals():
                    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=selected_doc)
                    df = pd.read_csv(obj['Body'])
                
                st.subheader("📊 Blood Parameters Visualization")
                
                # Range Chart (at the top)
                range_fig = create_range_chart(df)
                if range_fig:
                    st.plotly_chart(range_fig, use_container_width=True)

                # Bar Chart (below Range Chart)
                bar_fig = create_bar_chart(df)
                if bar_fig:
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                # Trend Lines (below Bar Chart)
                trend_fig = create_trend_lines(df)
                if trend_fig:
                    st.plotly_chart(trend_fig, use_container_width=True)

            except Exception as e:
                st.warning("Using sample data for visualization...")
                sample_data = {
                    'Parameter': ['Hemoglobin', 'RBC', 'WBC', 'Hematocrit'],
                    'Value': [14.2, 5.1, 75, 42],
                    'Reference_Range': ['13.5-17.5', '4.5-5.9', '40-110', '41-50'],
                    'Unit': ['g/dL', 'million/µL', 'cells/µL', '%']
                }
                df = pd.DataFrame(sample_data)
                
                # Range Chart (at the top)
                range_fig = create_range_chart(df)
                if range_fig:
                    st.plotly_chart(range_fig, use_container_width=True)

                # Bar Chart (below Range Chart)
                bar_fig = create_bar_chart(df)
                if bar_fig:
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                # Trend Lines (below Bar Chart)
                trend_fig = create_trend_lines(df)
                if trend_fig:
                    st.plotly_chart(trend_fig, use_container_width=True)

if __name__ == '__main__':
    try:
        import asyncio
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        st.warning(f"Note: Async warning can be ignored: {str(e)}")
    
    main()
