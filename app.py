from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

# Function to extract text from TXT files
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# General function to extract text from a file
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

# Function to find missing keywords from the job description in the resume
def find_missing_keywords(job_description, resume_text):
    job_keywords = set(job_description.lower().split())  # Split job description into keywords (lowercase)
    resume_keywords = set(resume_text.lower().split())  # Split resume into keywords (lowercase)
    
    missing_keywords = job_keywords - resume_keywords  # Keywords in job description but not in the resume
    return missing_keywords

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/matchresume")
def matchresume():
    return render_template('matchresume.html')

@app.route("/result")
def result():
    return render_template('result.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_files = request.files.getlist('resumes')
        
        if not resume_files or not job_description or resume_files[0].filename == '':
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")
        
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        resumes = []
        filenames = []
        missing_keywords_list = []
        
        for resume_file in resume_files:
            if resume_file.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(file_path)
                extracted_text = extract_text(file_path)
                if extracted_text:  # Only add if we could extract text
                    resumes.append(extracted_text)
                    filenames.append(resume_file.filename)
                    
                    # Find missing keywords in the resume
                    missing_keywords = find_missing_keywords(job_description, extracted_text)
                    missing_keywords_list.append(missing_keywords)
        
        if not resumes:
            return render_template('matchresume.html', message="Could not extract text from any of the uploaded files.")
        
        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([job_description] + resumes)
        
        # Calculate cosine similarities
        job_vector = vectors.toarray()[0]
        resume_vectors = vectors.toarray()[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]
        
        # Get top 5 resumes (or fewer if less than 5 were uploaded)
        max_results = min(5, len(resumes))
        top_indices = similarities.argsort()[-max_results:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(similarities[i] * 100, 2) for i in top_indices]  # Convert to percentage
        
        # Create the zipped list of resumes and their similarity scores
        top_resumes_with_scores = zip(top_resumes, similarity_scores)
        
        # Suggestions for improvement based on missing keywords
        suggestions = []
        for i, missing_keywords in enumerate(missing_keywords_list):
            if missing_keywords:
                suggestions.append(f"Resume '{top_resumes[i]}' is missing the following keywords from the job description: {', '.join(missing_keywords)}.")
            else:
                suggestions.append(f"Resume '{top_resumes[i]}' includes all the important keywords from the job description.")

        return render_template('result.html', 
                               message="Top matching resumes:", 
                               top_resumes_with_scores=top_resumes_with_scores, 
                               suggestions=suggestions)
    
    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
