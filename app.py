import os
import json
import time
import tempfile
import logging
from contextlib import contextmanager
from typing import Dict, Optional

from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
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

# Initialize Gemini - optimized for image analysis
gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    model = None
else:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Gemini 2.0 Flash initialized for image analysis")

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
    """Analyze PDF directly with Gemini 2.0 (best approach)"""
    try:
        if not model:
            return {"fields": [{"name": "Error", "value": "Gemini API not configured"}]}
        
        # Read PDF as binary
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        
        # Create the prompt
        prompt = get_analysis_prompt()
        
        # Prepare the PDF for Gemini
        pdf_part = {
            "mime_type": "application/pdf",
            "data": pdf_data
        }
        
        # Generate analysis
        response = model.generate_content([prompt, pdf_part])
        
        logger.info("Direct PDF analysis completed successfully")
        return parse_gemini_response(response.text)
        
    except Exception as e:
        logger.error(f"Direct PDF analysis failed: {e}")
        return {"fields": [{"name": "Error", "value": f"Direct PDF analysis failed: {str(e)}"}]}

def analyze_with_gemini_image(image_path):
    """Analyze image with Gemini Vision"""
    try:
        if not model:
            return {"fields": [{"name": "Error", "value": "Gemini API not configured"}]}
        
        with safe_image_context(image_path) as img:
            prompt = get_analysis_prompt()
            response = model.generate_content([prompt, img])
            
        logger.info("Image analysis completed successfully")
        return parse_gemini_response(response.text)
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return {"fields": [{"name": "Error", "value": f"Image analysis failed: {str(e)}"}]}

def get_analysis_prompt():
    """Comprehensive CAD analysis prompt"""
    return """
    Analyze this CAD/technical drawing and extract ALL visible information systematically.
    
    Look for and extract:
    
    **IDENTIFICATION:**
    - Drawing numbers, part numbers, revision codes
    - Sheet numbers, project codes, file references
    
    **DIMENSIONS & GEOMETRY:**
    - All measurements, dimensions, coordinates
    - Angles, radii, tolerances, scales
    - Geometric features (holes, slots, profiles)
    
    **MATERIALS & SPECIFICATIONS:**
    - Material types, grades, treatments
    - Surface finishes, coatings, hardness
    
    **MANUFACTURING:**
    - Machining operations, processes
    - Assembly instructions, notes
    - Quality requirements, inspection points
    
    **METADATA:**
    - Creation dates, designers, approvers
    - Standards referenced, units used
    - Revision history, change notes
    
    **ANNOTATIONS:**
    - Text callouts, symbols, warnings
    - Special instructions, notes
    - Reference designators, item numbers
    
    Return ONLY a valid JSON object with this structure:
    {
        "fields": [
            {"name": "Drawing Number", "value": "extracted value"},
            {"name": "Material", "value": "extracted value"},
            {"name": "Overall Length", "value": "extracted value"}
        ]
    }
    
    Guidelines:
    - Extract EVERYTHING visible, even small details
    - Use clear, descriptive field names
    - If text is unclear, note as "Partially visible: [best guess]"
    - Group related information logically
    - Be thorough and systematic
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
            # Strategy 1: Try direct PDF analysis (best for Gemini 2.0)
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

    # Build document history (list of uploaded files)
    uploads = []
    try:
        for fname in sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True):
            # skip analysis artifacts if you want in history show originals only
            if fname.endswith('.analysis.json'):
                continue
            uploads.append({
                'name': fname,
                'url': url_for('static', filename=os.path.join('..', app.config['UPLOAD_FOLDER'], fname))
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


@app.route('/chat/<path:filename>', methods=['POST'])
def chat_endpoint(filename):
    """Simple chat endpoint that uses the Gemini model when available or returns a canned reply."""
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Load analysis artifact to give context
    safe = secure_filename(filename)
    analysis_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{safe}.analysis.json")
    context_text = ''
    try:
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r', encoding='utf-8') as f:
                art = json.load(f)
                # include a brief summary of extracted fields
                fields = art.get('fields', [])
                context_text = f"Extracted {len(fields)} fields: " + ", ".join([str(f.get('name')) for f in fields[:10]])
    except Exception:
        context_text = ''

    # If model is configured, try to call it
    if model:
        try:
            prompt = f"You are an assistant for CAD document analysis. Context: {context_text}\nUser: {user_message}\nAnswer concisely."
            response = model.generate_content(prompt)
            text = getattr(response, 'text', None) or str(response)
            return jsonify({'reply': text}), 200
        except Exception as e:
            logger.error(f"Chat model call failed: {e}")

    # Fallback canned response
    reply = f"I've successfully analyzed the document. {context_text}. You asked: '{user_message}'. What would you like to know about this drawing?"
    return jsonify({'reply': reply}), 200

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