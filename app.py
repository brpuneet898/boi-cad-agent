from flask import Flask, render_template, request, flash, redirect, url_for
from flask_mail import Mail, Message
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')

mail = Mail(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        company = request.form['company']
        subject = request.form['subject']
        message = request.form['message']
        
        # Create email message
        msg = Message(
            subject=f'ADPA Contact Form: {subject}',
            sender=app.config['MAIL_USERNAME'],
            recipients=['brpuneet898@gmail.com']
        )
        
        msg.body = f"""
New contact form submission from ADPA website:

Name: {name}
Email: {email}
Company: {company}
Subject: {subject}

Message:
{message}

---
This email was sent from the ADPA contact form.
        """
        
        try:
            mail.send(msg)
            flash('Your message has been sent successfully!', 'success')
        except Exception as e:
            flash('There was an error sending your message. Please try again.', 'error')
            print(f"Error sending email: {e}")
        
        return redirect(url_for('contact'))
    
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
