import os
import json
import time
import tempfile
import logging
from contextlib import contextmanager
from typing import Dict, Optional

from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_from_directory
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
import google.generativeai as genai
from dotenv import load_dotenv

# Choose best PDF processing library
try:
    import fitz  # PyMuPDF - much better than pdf2image
    PDF_METHOD = 'pymupdf'
except ImportError:
    try:
        from pdf2image import convert_from_path
        PDF_METHOD = 'pdf2image'
    except ImportError:
        PDF_METHOD = 'direct'  # Use Gemini's native PDF support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_PIXELS = 50_000_000  # Better than unlimited
MAX_DIMENSION = 2000
MIN_DIMENSION = 800

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Gemini - optimized for fast CAD analysis
gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    model = None
else:
    genai.configure(api_key=gemini_api_key)
    
    # Configure generation settings for optimal speed and quality
    generation_config = genai.types.GenerationConfig(
        temperature=0.1,  # Lower temperature for more consistent, faster responses
        top_p=0.8,        # Focused sampling for speed
        top_k=20,         # Limit token choices for faster generation
        max_output_tokens=2048,  # Reasonable limit for CAD analysis
    )
    
    # Use Gemini 2.0 Flash for faster processing (2.5 is slower for document analysis)
    model = genai.GenerativeModel(
        'gemini-2.0-flash',  # Faster than 2.5 for this specific task
        generation_config=generation_config
    )
    logger.info("Gemini 2.0 Flash initialized with speed-optimized settings for CAD analysis")

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('EMAIL_USER')

mail = Mail(app)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@contextmanager
def safe_image_context(image_path):
    """Context manager for safe image handling with proper cleanup"""
    img = None
    try:
        # Set reasonable limit instead of unlimited
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(image_path)
        yield img
    finally:
        if img:
            img.close()

def convert_pdf_with_pymupdf(pdf_path):
    """Convert PDF using PyMuPDF - much faster and cleaner"""
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]  # Get first page

        # Higher quality rendering
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better quality
        pix = page.get_pixmap(matrix=mat)

        # Save as temporary image inside uploads with a temp_ prefix
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = str(int(time.time()))
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{base}_{timestamp}_pymupdf.png")
        pix.save(temp_path)

        # Clean up
        pix = None
        doc.close()

        logger.info(f"PDF converted successfully with PyMuPDF: {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"PyMuPDF conversion failed: {e}")
        return None

def convert_pdf_with_pdf2image(pdf_path):
    """Fallback: Convert PDF using pdf2image"""
    try:
        # Try different poppler configurations
        poppler_configs = [
            None,  # System PATH first
            os.path.join(os.path.dirname(__file__), 'poppler-25.07.0', 'Library', 'bin'),
            os.path.join(os.path.dirname(__file__), 'poppler', 'bin'),
        ]
        
        for poppler_path in poppler_configs:
            try:
                kwargs = {
                    'dpi': 200,
                    'first_page': 1,
                    'last_page': 1,
                    'fmt': 'png'
                }
                
                if poppler_path and os.path.exists(poppler_path):
                    kwargs['poppler_path'] = poppler_path
                    logger.info(f"Trying poppler path: {poppler_path}")
                
                images = convert_from_path(pdf_path, **kwargs)
                
                if images:
                    base = os.path.splitext(os.path.basename(pdf_path))[0]
                    timestamp = str(int(time.time()))
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{base}_{timestamp}_pdf2image.png")
                    images[0].save(temp_path, 'PNG')
                    
                    # Clean up images from memory
                    for img in images:
                        img.close()
                    
                    logger.info(f"PDF converted successfully with pdf2image: {temp_path}")
                    return temp_path
                    
            except Exception as e:
                logger.debug(f"Failed with poppler config {poppler_path}: {e}")
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"pdf2image conversion failed: {e}")
        return None

def convert_pdf_to_image(pdf_path):
    """Smart PDF conversion using best available method"""
    
    if PDF_METHOD == 'pymupdf':
        return convert_pdf_with_pymupdf(pdf_path)
    elif PDF_METHOD == 'pdf2image':
        return convert_pdf_with_pdf2image(pdf_path)
    else:
        # No conversion library available - will use direct PDF analysis
        logger.info("No PDF conversion library available, will attempt direct PDF analysis")
        return None

def enhance_image(image_path):
    """Enhanced image processing with proper resource management"""
    try:
        # Place enhanced image next to uploads and mark as temp-enhanced
        base = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = str(int(time.time()))
        enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{base}_{timestamp}_enhanced.jpg")

        with safe_image_context(image_path) as img:
            # Handle different color modes smartly
            if img.mode in ['CMYK', 'LAB']:
                img = img.convert('RGB')
            elif img.mode == 'P':  # Palette mode
                img = img.convert('L')  # Grayscale is often better for CAD
            
            # Smart resizing logic
            width, height = img.size
            max_dimension = max(width, height)
            
            if max_dimension > MAX_DIMENSION:
                scale_factor = MAX_DIMENSION / max_dimension
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized from {width}x{height} to {new_width}x{new_height}")
            elif max_dimension < MIN_DIMENSION:
                scale_factor = MIN_DIMENSION / max_dimension
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Upscaled from {width}x{height} to {new_width}x{new_height}")
            
            # Apply enhancements for CAD drawings
            # Moderate contrast boost for better line definition
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            
            # Sharpness for text clarity
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            # Save enhanced image
            img.save(enhanced_path, 'JPEG', quality=95, optimize=True)
        
        return enhanced_path
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return image_path  # Return original if enhancement fails

def analyze_with_gemini_direct_pdf(pdf_path):
    """Analyze PDF directly with Gemini - optimized for speed"""
    try:
        if not model:
            return {"fields": [{"name": "Error", "value": "Gemini API not configured"}]}
        
        # Check file size - if too large, use image conversion instead
        file_size = os.path.getsize(pdf_path)
        if file_size > 5 * 1024 * 1024:  # 5MB limit for direct PDF analysis
            logger.info(f"PDF too large ({file_size/1024/1024:.1f}MB), switching to image conversion")
            return None  # Will trigger fallback to image conversion
        
        # Read PDF as binary
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        
        # Create the optimized prompt
        prompt = get_analysis_prompt()
        
        # Prepare the PDF for Gemini
        pdf_part = {
            "mime_type": "application/pdf",
            "data": pdf_data
        }
        
        # Generate analysis with timeout handling
        logger.info("Starting direct PDF analysis...")
        start_time = time.time()
        
        response = model.generate_content([prompt, pdf_part])
        
        processing_time = time.time() - start_time
        logger.info(f"Direct PDF analysis completed in {processing_time:.2f} seconds")
        
        return parse_gemini_response(response.text)
        
    except Exception as e:
        logger.error(f"Direct PDF analysis failed: {e}")
        return {"fields": [{"name": "Error", "value": f"Direct PDF analysis failed: {str(e)}"}]}

def analyze_with_gemini_image(image_path):
    """Analyze image with Gemini Vision - optimized for speed"""
    try:
        if not model:
            return {"fields": [{"name": "Error", "value": "Gemini API not configured"}]}
        
        logger.info("Starting image analysis...")
        start_time = time.time()
        
        with safe_image_context(image_path) as img:
            # Optimize image size for faster processing
            width, height = img.size
            if width > 1024 or height > 1024:
                # Resize for faster processing while maintaining quality
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {img.size} for faster processing")
            
            prompt = get_analysis_prompt()
            response = model.generate_content([prompt, img])
            
        processing_time = time.time() - start_time
        logger.info(f"Image analysis completed in {processing_time:.2f} seconds")
        
        return parse_gemini_response(response.text)
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return {"fields": [{"name": "Error", "value": f"Image analysis failed: {str(e)}"}]}

def get_analysis_prompt():
    """Optimized CAD analysis prompt for speed"""
    return """
    Extract key information from this CAD/technical drawing. Focus on the most important data.
    
    EXTRACT:
    - Drawing/part numbers, revisions
    - Dimensions, measurements, tolerances  
    - Materials, specifications
    - Manufacturing notes, processes
    - Designer info, dates, standards
    - Text annotations, symbols
    
    Return ONLY valid JSON:
    {
        "fields": [
            {"name": "Drawing Number", "value": "extracted value"},
            {"name": "Material", "value": "extracted value"}
        ]
    }
    
    Rules:
    - Extract visible text and numbers
    - Use clear field names
    - If unclear, prefix with "Partial: "
    - Focus on manufacturing-relevant data
    - Be concise but accurate
    """

def parse_gemini_response(response_text):
    """Parse Gemini response with robust JSON extraction"""
    try:
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        
        # Extract JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate structure
            if isinstance(result, dict) and 'fields' in result and isinstance(result['fields'], list):
                logger.info(f"Successfully parsed {len(result['fields'])} fields from response")
                return result
        
        # Fallback: return as single field
        logger.warning("Could not parse JSON, returning raw response")
        return {
            "fields": [
                {"name": "Analysis Result", "value": response_text[:1000] + ("..." if len(response_text) > 1000 else "")}
            ]
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return {
            "fields": [
                {"name": "Raw Analysis", "value": response_text}
            ]
        }

def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    # Remove provided file paths if they look temporary
    for file_path in file_paths:
        try:
            if not file_path:
                continue
            if os.path.exists(file_path):
                name = os.path.basename(file_path).lower()
                if ('temp_' in name) or ('_pymupdf' in name) or ('_pdf2image' in name) or ('_enhanced' in name):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.debug(f"Could not clean up {file_path}: {e}")

    # Additionally, proactively remove older temp_* files in upload folder (best-effort)
    try:
        for fname in os.listdir(app.config['UPLOAD_FOLDER']):
            if fname.startswith('temp_'):
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                try:
                    os.remove(fpath)
                    logger.debug(f"Cleaned up residual temp file: {fpath}")
                except Exception:
                    pass
    except Exception:
        pass

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    temp_files = []  # Track temp files for cleanup
    start_time = time.time()

    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PDF, PNG, or JPG files.'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        safe_filename = f"{name}_{timestamp}{ext}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)

        analysis_result = None

        if ext.lower() == '.pdf':
            # Strategy 1: Try direct PDF analysis (best for Gemini 2.5)
            logger.info('Attempting direct PDF analysis...')
            analysis_result = analyze_with_gemini_direct_pdf(filepath)

            # Strategy 2: Fallback to image conversion if direct analysis fails
            if not analysis_result or 'Error' in str(analysis_result):
                logger.info('Direct PDF failed, trying image conversion...')
                converted_image = convert_pdf_to_image(filepath)

                if converted_image:
                    temp_files.append(converted_image)
                    enhanced_image = enhance_image(converted_image)
                    temp_files.append(enhanced_image)

                    analysis_result = analyze_with_gemini_image(enhanced_image)
                else:
                    analysis_result = {
                        'fields': [
                            {'name': 'Error', 'value': 'Failed to process PDF - no conversion method available'}
                        ]
                    }

        else:
            # Image files
            logger.info('Processing image file...')
            enhanced_image = enhance_image(filepath)
            temp_files.append(enhanced_image)

            analysis_result = analyze_with_gemini_image(enhanced_image)

        # Measure processing time
        processing_time = round(time.time() - start_time, 3)

        # Attach processing metadata to analysis result
        metadata = {
            'file_name': safe_filename,
            'file_size': os.path.getsize(filepath),
            'processing_time_seconds': processing_time,
            'confidence': None,
            'language': 'English'
        }

        # Add simple confidence heuristic if fields exist
        try:
            fields_count = len(analysis_result.get('fields', [])) if isinstance(analysis_result, dict) else 0
            if fields_count > 0:
                # heuristic: base 80 + 1 * fields (capped)
                metadata['confidence'] = min(99.9, 80 + fields_count * 1.0)
            else:
                metadata['confidence'] = 50.0
        except Exception:
            metadata['confidence'] = None

        # Persist analysis JSON next to uploaded file for later retrieval
        analysis_artifact = {
            'fields': analysis_result.get('fields', []) if isinstance(analysis_result, dict) else [],
            'raw': analysis_result,
            'metadata': metadata
        }

        analysis_filename = f"{safe_filename}.analysis.json"
        analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)
        try:
            with open(analysis_path, 'w', encoding='utf-8') as af:
                json.dump(analysis_artifact, af, indent=2)
        except Exception as e:
            logger.debug(f"Could not write analysis artifact: {e}")

        # Clean up temporary files
        cleanup_temp_files(*temp_files)

        return jsonify({
            'success': True,
            'message': 'File analyzed successfully!',
            'filename': safe_filename,
            'analysis_file': analysis_filename
        }), 200

    except Exception as e:
        # Clean up on error
        cleanup_temp_files(*temp_files)
        logger.error(f"Upload processing failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/analysis/<path:filename>')
def analysis_page(filename):
    """Render analysis page for a given uploaded filename"""
    # Locate analysis artifact
    safe = secure_filename(filename)
    analysis_filename = f"{safe}.analysis.json"
    analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)

    if not os.path.exists(analysis_path):
        flash('Analysis not found for the requested file.', 'error')
        return redirect(url_for('home'))

    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_artifact = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read analysis artifact: {e}")
        flash('Could not read analysis results.', 'error')
        return redirect(url_for('home'))

    # Build document history (list of uploaded files) - newest first
    uploads = []
    try:
        # Get all files with their modification times
        files_with_times = []
        for fname in os.listdir(app.config['UPLOAD_FOLDER']):
            if fname.endswith('.analysis.json'):
                continue  # Skip analysis artifacts
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            mtime = os.path.getmtime(file_path)
            files_with_times.append((fname, mtime))
        
        # Sort by modification time (newest first)
        files_with_times.sort(key=lambda x: x[1], reverse=True)
        
        # Build uploads list
        for fname, _ in files_with_times:
            uploads.append({
                'name': fname,
                'url': url_for('uploaded_file', filename=fname)
            })
    except Exception:
        uploads = []

    # Determine simple file metadata for display
    metadata = analysis_artifact.get('metadata', {})

    # For PDFs, try to get page count
    extra_file_info = {}
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if os.path.exists(file_path):
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.pdf' and 'fitz' in globals():
                try:
                    doc = fitz.open(file_path)
                    extra_file_info['pages'] = doc.page_count
                    doc.close()
                except Exception:
                    extra_file_info['pages'] = None
            elif ext.lower() in ('.png', '.jpg', '.jpeg'):
                try:
                    with Image.open(file_path) as im:
                        extra_file_info['resolution'] = f"{im.width}x{im.height} px"
                except Exception:
                    extra_file_info['resolution'] = None

    except Exception:
        pass

    return render_template('analysis.html', analysis=analysis_artifact, uploads=uploads, metadata=metadata, extra=extra_file_info, filename=safe)


@app.route('/save-analysis/<path:filename>', methods=['POST'])
def save_analysis(filename):
    """Save updated analysis fields to JSON file"""
    try:
        # Get the updated fields from request
        data = request.get_json()
        if not data or 'fields' not in data:
            return jsonify({'error': 'No fields data provided'}), 400
        
        # Locate analysis artifact file
        safe = secure_filename(filename)
        analysis_filename = f"{safe}.analysis.json"
        analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)
        
        if not os.path.exists(analysis_path):
            return jsonify({'error': 'Analysis file not found'}), 404
        
        # Read existing analysis
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_artifact = json.load(f)
        
        # Update the fields
        analysis_artifact['fields'] = data['fields']
        
        # Add update timestamp
        analysis_artifact['metadata']['last_updated'] = time.time()
        analysis_artifact['metadata']['updated_fields_count'] = len(data['fields'])
        
        # Save updated analysis
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_artifact, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis updated for {filename}: {len(data['fields'])} fields")
        
        return jsonify({
            'success': True,
            'message': 'Analysis saved successfully',
            'fields_count': len(data['fields'])
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to save analysis for {filename}: {e}")
        return jsonify({'error': f'Failed to save analysis: {str(e)}'}), 500

@app.route('/chat/<path:filename>', methods=['POST'])
def chat_endpoint(filename):
    """Enhanced AI chat endpoint with comprehensive CAD document understanding."""
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Load comprehensive analysis context
    safe = secure_filename(filename)
    analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{safe}.analysis.json")
    document_context = load_document_context(analysis_path, safe)
    
    # If model is configured, use enhanced AI response
    if model:
        try:
            enhanced_prompt = create_enhanced_chat_prompt(document_context, user_message, safe)
            response = model.generate_content(enhanced_prompt)
            text = getattr(response, 'text', None) or str(response)
            
            # Log the interaction for debugging
            logger.info(f"Chat interaction for {filename}: User: '{user_message[:50]}...' AI: '{text[:50]}...'")
            
            return jsonify({'reply': text}), 200
        except Exception as e:
            logger.error(f"Enhanced chat model call failed: {e}")
            return jsonify({'reply': get_intelligent_fallback_response(user_message, document_context)}), 200

    # Enhanced fallback response
    return jsonify({'reply': get_intelligent_fallback_response(user_message, document_context)}), 200

def load_document_context(analysis_path, filename):
    """Load comprehensive document context for AI chat."""
    context = {
        'filename': filename,
        'fields': [],
        'categories': {'A': [], 'B': [], 'C': [], 'D': []},
        'metadata': {},
        'field_count': 0,
        'document_type': 'Unknown'
    }
    
    try:
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                
            # Extract fields and categorize them
            fields = analysis_data.get('fields', [])
            context['fields'] = fields
            context['field_count'] = len(fields)
            context['metadata'] = analysis_data.get('metadata', {})
            
            # Categorize fields for better context
            for field in fields:
                category = field.get('category', 'Uncategorized')
                if category in context['categories']:
                    context['categories'][category].append(field)
            
            # Determine document type based on content
            context['document_type'] = determine_document_type(fields)
            
    except Exception as e:
        logger.debug(f"Could not load document context: {e}")
    
    return context

def determine_document_type(fields):
    """Determine the type of CAD document based on extracted fields."""
    field_text = ' '.join([f.get('name', '') + ' ' + f.get('value', '') for f in fields]).lower()
    
    if any(keyword in field_text for keyword in ['assembly', 'exploded', 'parts list', 'bom']):
        return 'Assembly Drawing'
    elif any(keyword in field_text for keyword in ['detail', 'part', 'component']):
        return 'Detail Drawing'
    elif any(keyword in field_text for keyword in ['schematic', 'circuit', 'electrical']):
        return 'Schematic'
    elif any(keyword in field_text for keyword in ['section', 'cross-section', 'view']):
        return 'Section Drawing'
    else:
        return 'Technical Drawing'

def create_enhanced_chat_prompt(context, user_message, filename):
    """Create a comprehensive prompt for the AI assistant."""
    
    # Build detailed context with all available fields
    all_fields = []
    for category, fields in context['categories'].items():
        if fields:
            category_names = {
                'A': 'Document Information',
                'B': 'Metadata', 
                'C': 'Textual Information',
                'D': 'Drawing Details'
            }
            for field in fields:
                all_fields.append(f"- {field['name']}: {field['value']}")
    
    # Also include uncategorized fields
    for field in context['fields']:
        if field.get('category') not in ['A', 'B', 'C', 'D']:
            all_fields.append(f"- {field['name']}: {field['value']}")
    
    context_summary = '\n'.join(all_fields) if all_fields else "No specific fields extracted yet."
    
    prompt = f"""You are an expert CAD/Technical Drawing AI Assistant.

DOCUMENT: {filename} ({context['document_type']})
EXTRACTED DATA ({context['field_count']} fields):
{context_summary}

USER QUESTION: {user_message}

INSTRUCTIONS:
- Answer based on the EXTRACTED DATA above
- If the requested information IS in the data, provide it clearly
- If the requested information is NOT in the data, say so and suggest where to look
- Be helpful and specific
- Use bullet points for lists
- Keep responses concise but informative

RESPONSE:"""

    return prompt

def get_intelligent_fallback_response(user_message, context):
    """Generate intelligent fallback responses when AI model is unavailable."""
    
    message_lower = user_message.lower()
    
    # Search for specific information in extracted fields
    if 'part number' in message_lower or 'part num' in message_lower:
        part_fields = [f for f in context['fields'] if any(keyword in f.get('name', '').lower() 
                      for keyword in ['part', 'number', 'item', 'drawing', 'dwg'])]
        if part_fields:
            results = []
            for field in part_fields[:3]:
                results.append(f"• {field['name']}: {field['value']}")
            return f"**Part Number Information Found:**\n" + '\n'.join(results)
        else:
            return "**Part Numbers:** No part numbers were extracted from the document. Check the title block, drawing number field, or parts list in the original document."
    
    elif 'dimension' in message_lower or 'measurement' in message_lower:
        dimensions = [f for f in context['fields'] if any(dim_word in f.get('name', '').lower() 
                     for dim_word in ['dimension', 'length', 'width', 'height', 'diameter', 'radius', 'size'])]
        if dimensions:
            results = []
            for field in dimensions[:4]:
                results.append(f"• {field['name']}: {field['value']}")
            return f"**Dimensions Found:**\n" + '\n'.join(results)
        else:
            return "**Dimensions:** No dimensional information was extracted. Look for dimension lines, measurement callouts, or size specifications in the original drawing."
    
    elif 'material' in message_lower:
        materials = [f for f in context['fields'] if any(mat_word in f.get('name', '').lower() 
                    for mat_word in ['material', 'steel', 'aluminum', 'plastic', 'metal'])]
        if materials:
            results = []
            for field in materials[:3]:
                results.append(f"• {field['name']}: {field['value']}")
            return f"**Material Information:**\n" + '\n'.join(results)
        else:
            return "**Materials:** No material specifications were extracted. Check the notes section, title block, or material callouts in the original document."
    
    elif any(word in message_lower for word in ['tolerance', 'precision', 'accuracy']):
        tolerances = [f for f in context['fields'] if any(tol_word in f.get('name', '').lower() 
                     for tol_word in ['tolerance', 'precision', '±', 'accuracy'])]
        if tolerances:
            results = []
            for field in tolerances[:3]:
                results.append(f"• {field['name']}: {field['value']}")
            return f"**Tolerance Information:**\n" + '\n'.join(results)
        else:
            return "**Tolerances:** No tolerance specifications were extracted. Look for ± symbols, precision callouts, or geometric dimensioning and tolerancing (GD&T) symbols."
    
    # Show available information
    if context['field_count'] > 0:
        sample_fields = context['fields'][:4]
        results = []
        for field in sample_fields:
            results.append(f"• {field['name']}: {field['value']}")
        
        return f"**Available Information ({context['field_count']} fields extracted):**\n" + '\n'.join(results) + f"\n\nAsk me about specific aspects like dimensions, materials, part numbers, or manufacturing details."
    else:
        return f"**Document Analysis:** This {context['document_type']} is being processed. No specific fields have been extracted yet. Please try asking about general document information or upload a clearer image."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        company = request.form.get('company', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()
        
        # Basic validation
        if not all([name, email, subject, message]):
            flash('Please fill in all required fields.', 'error')
            return redirect(url_for('contact'))
        
        # Check email configuration
        if not (app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']):
            flash('Email service is not configured. Please contact us directly at brpuneet898@gmail.com', 'error')
            return redirect(url_for('contact'))
        
        try:
            # Send email
            msg = Message(
                subject=f'ADPA Contact Form: {subject}',
                sender=app.config['MAIL_USERNAME'],
                recipients=['brpuneet898@gmail.com'],
                reply_to=email
            )
            
            msg.body = f"""
New contact form submission from ADPA website:

Name: {name}
Email: {email}
Company: {company or 'Not specified'}
Subject: {subject}

Message:
{message}

---
Reply to: {email}
This email was sent from the ADPA contact form.
            """
            
            mail.send(msg)
            flash('Your message has been sent successfully! We\'ll get back to you soon.', 'success')
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            flash('There was an error sending your message. Please email us directly at brpuneet898@gmail.com', 'error')
        
        return redirect(url_for('contact'))
    
    return render_template("contact.html")

if __name__ == "__main__":
    logger.info(f"Starting application with PDF method: {PDF_METHOD}")
    app.run(debug=True)