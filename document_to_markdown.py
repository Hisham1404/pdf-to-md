import streamlit as st
import os
from markitdown import MarkItDown
from typing import Tuple
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch
from pypdf import PdfReader
from pdf2image import convert_from_path
import numpy as np
import requests
import json
from groq import Groq
import shutil
from pathlib import Path

# Add API key configuration
GROQ_API_KEY = ""  # Replace with your actual Groq API key

def setup_document_model():
    """Initialize the LayoutLMv3 model for document analysis"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            trust_remote_code=True
        )
        model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            trust_remote_code=True
        )
        model = model.to(device)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def setup_groq_client():
    """Configure Groq API client"""
    try:
        api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error setting up Groq client: {str(e)}")
        return None

def generate_with_groq(prompt: str, client) -> str:
    """
    Generate text using Groq API
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Groq's Mixtral model
            messages=[
                {"role": "system", "content": "You are an expert at converting documents to well-formatted markdown without removing any datas just format it according to markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=2048,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        st.error("Full error details:", exc_info=True)
        return ""

def check_poppler_installation():
    """Check if Poppler is installed and accessible"""
    try:
        if os.name == 'nt':  # Windows
            poppler_path = shutil.which('pdftoppm')
            if not poppler_path:
                st.error("""
                Poppler is not found! Please install Poppler:
                1. Download from: http://blog.alivate.com.au/poppler-windows/
                2. Add the bin/ directory to your PATH
                Or use: pip install poppler-utils
                """)
                return False
        else:  # Linux/Mac
            if not shutil.which('pdftoppm'):
                st.error("""
                Poppler is not installed! Please install:
                - Linux: sudo apt-get install poppler-utils
                - Mac: brew install poppler
                """)
                return False
        return True
    except Exception as e:
        st.error(f"Error checking Poppler installation: {str(e)}")
        return False

def process_pdf(file_path, processor, model) -> str:
    """
    Process PDF document with LayoutLMv3
    """
    if not check_poppler_installation():
        return "Error: Poppler is required for PDF processing"
        
    try:
        # First try to extract text directly
        pdf_reader = PdfReader(file_path)
        text_content = []
        
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
            
        if any(text_content):  # If text extraction successful
            return "\n\n".join(text_content)
            
        # If no text found, try OCR approach
        try:
            images = convert_from_path(file_path)
        except Exception as e:
            st.error(f"Error converting PDF to images. Make sure Poppler is properly installed. Error: {str(e)}")
            return ""
            
        extracted_text = []
        device = next(model.parameters()).device
        
        for image in images:
            image_np = np.array(image)
            encoding = processor(image_np, return_tensors="pt")
            # Move tensors to same device as model
            encoding = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = model(**encoding)
            
            page_text = processor.decode(outputs.logits.argmax(-1)[0])
            extracted_text.append(page_text)
        
        return "\n\n".join(extracted_text)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def process_single_file(file, doc_model, groq_client, use_ocr=False, use_llm=False) -> Tuple[str, str]:
    """
    Convert a single file to markdown format with optional OCR and LLM processing.
    """
    if doc_model[0] is None or doc_model[1] is None:
        st.error("Document model not properly initialized")
        return file.name, ""
    
    if use_llm and groq_client is None:
        st.error("Groq client not properly initialized")
        return file.name, ""
        
    temp_path = f"temp_{file.name}"
    try:
        with open(temp_path, 'wb') as f:
            f.write(file.getbuffer())
        
        if file.type == 'application/pdf' and use_ocr:
            extracted_text = process_pdf(temp_path, *doc_model)
            
            if use_llm and extracted_text:
                prompt = (
                    "Convert the following text to well-formatted markdown. "
                    "Preserve the original layout, headings, lists, and all content. "
                    "Ensure proper markdown syntax for all elements.\n\n"
                    f"{extracted_text}"
                )
                markdown_content = generate_with_groq(prompt, groq_client)
                return file.name, markdown_content
            
            return file.name, extracted_text
        else:
            md = MarkItDown()
            result = md.convert(temp_path)
            
            if use_llm:
                prompt = f"Improve the markdown formatting of this text:\n\n{result.text_content}"
                return file.name, generate_with_groq(prompt, groq_client)
            
            return file.name, result.text_content
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def save_to_folder(content: str, filename: str, folder_path: str) -> bool:
    """Save markdown content to a specific folder"""
    try:
        # Ensure folder path is absolute
        folder_path = os.path.abspath(folder_path)
        
        # Create directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Create full file path with md extension
        full_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.md")
        
        # Force write content
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
        except Exception as write_error:
            st.error(f"Write error: {write_error}")
            return False
        
        # Verify file exists and has content
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            st.success(f"✓ File saved at: {full_path}")
            return True
        else:
            st.error("File was not saved properly")
            return False
            
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False

def save_multiple_files(results: list, folder_path: str) -> bool:
    """Save multiple files to folder"""
    success = True
    saved_files = []
    
    for filename, content in results:
        try:
            full_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.md")
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            saved_files.append(full_path)
        except Exception as e:
            st.error(f"Error saving {filename}: {str(e)}")
            success = False
    
    if success:
        st.success("✓ All files saved successfully!")
        with st.expander("Saved Files Location"):
            for file_path in saved_files:
                st.info(f"- {file_path}")
    return success

def main():
    """Main application for document to markdown conversion"""
    # Initialize session states
    if 'conversion_results' not in st.session_state:
        st.session_state.conversion_results = []
    if 'save_triggered' not in st.session_state:
        st.session_state.save_triggered = False
    if 'current_results' not in st.session_state:
        st.session_state.current_results = []

    st.title("Document to Markdown Converter with Layout Analysis")
    
    # Check Poppler installation at startup
    if not check_poppler_installation():
        st.warning("PDF processing capabilities will be limited without Poppler installation")
    
    # Initialize models with improved error handling
    with st.spinner("Loading models..."):
        doc_model = setup_document_model()
        groq_client = setup_groq_client()
        
        if groq_client is None:
            st.error("Failed to initialize Groq API client. Please check your API key.")
            st.stop()

    # Add processing options in sidebar
    with st.sidebar:
        st.header("Processing Options")
        use_ocr = st.checkbox("Use Layout Analysis for PDFs", value=True,
                             help="Use LayoutLMv3 to analyze document structure (recommended for PDFs)")
        use_llm = st.checkbox("Use LLM for Text Enhancement", value=True,
                             help="Use Groq LLM to improve text formatting")
        
        # Add Groq API Key configuration in sidebar
        current_api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        if current_api_key == "gsk_your_key_here":
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key"
            )
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
                groq_client = setup_groq_client()
        else:
            st.success("Groq API Key is configured")
        
        st.header("About")
        st.markdown("""
        ## Supported Formats:
        - PDF (.pdf)
        - Word (.docx)
        - PowerPoint (.pptx)
        - Excel (.xlsx)
        - Images (.jpg, .png)
        - HTML (.html)
        - Text (.txt)
        
        ## How to Use:
        1. Upload one or more files
        2. Choose output format
        3. Convert and download
        """)

    # Main file upload section
    uploaded_files = st.file_uploader(
        "Choose files to convert", 
        type=['pdf', 'docx', 'pptx', 'xlsx', 'jpg', 'png', 'html', 'txt'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} files for conversion")
        
        # Replace folder selection with text input
        with st.form(key='save_form'):
            col1, col2 = st.columns([3, 1])
            with col1:
                output_format = st.radio(
                    "Choose output format:",
                    ["Combined Markdown (Single File)", "Separate Markdown Files"]
                )
            with col2:
                download_path = st.text_input(
                    "Save to folder path",
                    value=st.session_state.get('download_folder', os.path.expanduser('~')),  # Default to user's home
                    help="Enter the full path to save files (e.g., C:/Users/username/Documents)"
                )
            
            convert_button = st.form_submit_button("Convert and Save")
            
            if convert_button:
                if not download_path:
                    st.error("Please specify a save location")
                else:
                    try:
                        # Validate and create folder
                        download_path = os.path.abspath(os.path.normpath(download_path))
                        os.makedirs(download_path, exist_ok=True)
                        st.session_state['download_folder'] = download_path
                        
                        # Convert files
                        with st.spinner("Converting files..."):
                            results = []
                            for file in uploaded_files:
                                filename, content = process_single_file(
                                    file, doc_model, groq_client, 
                                    use_ocr=use_ocr, use_llm=use_llm
                                )
                                results.append((filename, content))
                            
                            st.session_state.current_results = results
                            
                            # Handle combined or separate files
                            if output_format == "Combined Markdown (Single File)":
                                combined_content = "\n\n".join([
                                    f"# Content from {filename}\n\n{content}" 
                                    for filename, content in results
                                ])
                                
                                st.text_area(
                                    "Combined Output Preview", 
                                    combined_content, 
                                    height=300
                                )
                                
                                # Save combined file
                                success = save_to_folder(
                                    combined_content, 
                                    "combined_output.md", 
                                    download_path
                                )
                                
                            else:  # Separate files
                                success = save_multiple_files(results, download_path)
                            
                            if success:
                                st.success(f"Files saved to: {download_path}")
                                # Keep results for browser download
                                st.session_state.conversion_results = results
                                
                    except Exception as e:
                        st.error(f"Error during conversion/save: {str(e)}")
                        st.error("Full error details:", exc_info=True)

        # Show browser download buttons if results exist
        if st.session_state.conversion_results:
            st.subheader("Download Options")
            if output_format == "Combined Markdown (Single File)":
                combined_content = "\n\n".join([
                    f"# Content from {filename}\n\n{content}" 
                    for filename, content in st.session_state.conversion_results
                ])
                st.download_button(
                    label="Download Combined File",
                    data=combined_content,
                    file_name="combined_output.md",
                    mime="text/markdown"
                )
            else:
                for idx, (filename, content) in enumerate(st.session_state.conversion_results):
                    st.download_button(
                        label=f"Download {filename}",
                        data=content,
                        file_name=f"{os.path.splitext(filename)[0]}.md",
                        mime="text/markdown",
                        key=f"download_{idx}"
                    )

    # Show previously converted results if they exist
    elif st.session_state.conversion_results:
        st.info("Previous conversion results available")
        if st.button("Clear previous results"):
            st.session_state.conversion_results = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()