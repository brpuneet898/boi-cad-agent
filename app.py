from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask_mail import Mail, Message
import os
import json
import time
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
import google.generativeai as genai
from pdf2image import convert_from_path

# Increase PIL's image size limit for large CAD drawings
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API
api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if not api_key:
    print("Warning: No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file")
    model = None
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('EMAIL_USER')

mail = Mail(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_pdf_to_image(pdf_path):
    """Convert PDF to image for Gemini analysis"""
    try:
        # Use local poppler installation in project directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Check different possible poppler locations
        poppler_paths = [
            os.path.join(project_root, 'poppler-25.07.0', 'Library', 'bin'),  # Current location
            os.path.join(project_root, 'poppler', 'bin'),  # Expected location
            None,  # Try system PATH as fallback
        ]
        
        for poppler_path in poppler_paths:
            try:
                # Convert PDF to images (assuming single page CAD drawing)
                if poppler_path and os.path.exists(poppler_path):
                    print(f"Trying poppler path: {poppler_path}")
                    # Use lower DPI to avoid huge images (200 DPI is still good quality)
                    images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1, poppler_path=poppler_path)
                elif poppler_path is None:
                    print("Trying system PATH")
                    images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
                else:
                    print(f"Poppler path does not exist: {poppler_path}")
                    continue
                
                if images:
                    # Get the first (and only) page
                    img = images[0]
                    
                    # Check image size and resize if too large
                    width, height = img.size
                    max_dimension = max(width, height)
                    
                    # If image is very large, resize it (max 2000px on longest side)
                    if max_dimension > 2000:
                        scale_factor = 2000 / max_dimension
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                    
                    # Save as PNG for better quality
                    image_path = pdf_path.replace('.pdf', '_converted.png')
                    img.save(image_path, 'PNG')
                    
                    print(f"PDF converted successfully to: {image_path}")
                    return image_path
                    
            except Exception as e:
                print(f"Failed with poppler path {poppler_path}: {e}")
                continue
        
        # If all paths failed
        raise Exception("Poppler not found. Please check poppler installation")
        
    except Exception as e:
        print(f"PDF conversion failed: {e}")
        return None

def enhance_image(image_path):
    """Enhance image quality for better Gemini analysis"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Keep original mode for CAD drawings (often grayscale/monochrome is better)
        # Only convert to RGB if it's an unsupported mode like CMYK
        if img.mode in ['CMYK', 'LAB']:
            img = img.convert('RGB')
        elif img.mode == 'P':  # Palette mode
            img = img.convert('L')  # Convert to grayscale instead of RGB
        # Keep L (grayscale), LA, RGB, RGBA as they are
        
        # Enhance contrast (more aggressive for technical drawings)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # Increased for better line definition
        
        # Enhance sharpness (important for text readability)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)  # Increased for sharper text/lines
        
        # Resize if too small (min 800px on longest side)
        width, height = img.size
        max_dimension = max(width, height)
        if max_dimension < 800:
            scale_factor = 800 / max_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save enhanced image
        enhanced_path = image_path.replace('.', '_enhanced.')
        img.save(enhanced_path, 'JPEG', quality=95)
        
        return enhanced_path
    except Exception as e:
        print(f"Image enhancement failed: {e}")
        return image_path  # Return original if enhancement fails

def analyze_cad_with_gemini(image_path):
    """Analyze CAD drawing with Gemini Vision and extract all fields"""
    try:
        # Check if model is initialized
        if model is None:
            return {
                "fields": [
                    {"name": "Error", "value": "Gemini API key not configured. Please set GEMINI_API_KEY in your .env file"}
                ]
            }
        # Enhance image first
        enhanced_path = enhance_image(image_path)
        
        # Prepare the image
        img = Image.open(enhanced_path)
        
        # Create detailed prompt for CAD analysis
        prompt = """
        Analyze this CAD/technical drawing thoroughly and extract ALL visible information. 
        Look for every piece of text, number, dimension, annotation, and technical detail.
        
        Organize the extracted information into logical field-value pairs. Include:
        
        1. Drawing identification (drawing numbers, part numbers, revision codes)
        2. Dimensions and measurements (all sizes, angles, tolerances, coordinates)
        3. Material specifications (material types, grades, finishes, coatings)
        4. Manufacturing information (processes, operations, assembly notes)
        5. Quality requirements (tolerances, surface finishes, inspection notes)
        6. Metadata (scales, units, dates, designers, approvals)
        7. Annotations and notes (callouts, symbols, special instructions)
        8. Any other technical information visible in the drawing
        
        Return the results as a JSON object with this exact structure:
        {
            "fields": [
                {"name": "Field Name", "value": "Field Value"},
                {"name": "Another Field", "value": "Another Value"}
            ]
        }
        
        Make field names descriptive and clear. If you can't read something clearly, indicate it as "Unclear text" or "Partially visible".
        Extract everything you can see, even if it seems minor.
        """
        
        # Generate response
        response = model.generate_content([prompt, img])
        
        # Parse JSON from response
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate structure
                if 'fields' in result and isinstance(result['fields'], list):
                    return result
            
            # If JSON parsing fails, create structured response from text
            return {
                "fields": [
                    {"name": "Analysis Result", "value": response_text[:500] + "..." if len(response_text) > 500 else response_text}
                ]
            }
            
        except json.JSONDecodeError:
            # Fallback: return raw response as a single field
            return {
                "fields": [
                    {"name": "Raw Analysis", "value": response_text}
                ]
            }
            
    except Exception as e:
        print(f"Gemini analysis failed: {e}")
        return {
            "fields": [
                {"name": "Error", "value": f"Analysis failed: {str(e)}"}
            ]
        }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PDF, PNG, or JPG files.'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Handle both images and PDFs
            analysis_result = None
            
            if ext.lower() in ['.png', '.jpg', '.jpeg']:
                # Direct image analysis
                analysis_result = analyze_cad_with_gemini(filepath)
                
            elif ext.lower() == '.pdf':
                # Convert PDF to image first
                converted_image_path = convert_pdf_to_image(filepath)
                
                if converted_image_path:
                    # Analyze the converted image
                    analysis_result = analyze_cad_with_gemini(converted_image_path)
                else:
                    analysis_result = {
                        'fields': [
                            {'name': 'Error', 'value': 'Failed to convert PDF to image'}
                        ]
                    }
            
            if analysis_result:
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and analyzed successfully!',
                    'filename': filename,
                    'filepath': filepath,
                    'analysis': analysis_result
                }), 200
            else:
                return jsonify({
                    'success': True,
                    'message': 'File uploaded but analysis failed',
                    'filename': filename,
                    'filepath': filepath,
                    'analysis': {
                        'fields': [
                            {'name': 'Status', 'value': 'Upload successful, analysis failed'}
                        ]
                    }
                }), 200
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        company = request.form['company']
        subject = request.form['subject']
        message = request.form['message']
        
        # Check if email is configured
        if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
            flash('Email service is not configured. Please contact us directly at brpuneet898@gmail.com', 'error')
            return redirect(url_for('contact'))
        
        # Create email message
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
        
        try:
            mail.send(msg)
            flash('Your message has been sent successfully! We\'ll get back to you soon.', 'success')
        except Exception as e:
            flash('There was an error sending your message. Please email us directly at brpuneet898@gmail.com', 'error')
            print(f"Error sending email: {e}")
            print(f"Mail config - Username: {app.config['MAIL_USERNAME']}, Password set: {bool(app.config['MAIL_PASSWORD'])}")
        
        return redirect(url_for('contact'))
    
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
