from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Define common English stopwords without requiring NLTK
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 
    'couldn', 'd', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'down', 'during', 'each', 'few', 
    'for', 'from', 'further', 'had', 'hadn', 'has', 'hasn', 'have', 'haven', 'having', 'he', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'it', 'its', 
    'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 'more', 'most', 'mustn', 'my', 'myself', 'needn', 
    'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 
    'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', 'she', 'should', 'shouldn', 'so', 'some', 
    'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 
    'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 
    'we', 'were', 'weren', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 
    'won', 'wouldn', 'y', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

# Function to extract text from TXT files
def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"Error extracting text from TXT: {e}")
            return ""
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

# General function to extract text from a file
def extract_text(file_path):
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Simple tokenization by splitting on whitespace
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    # Basic stemming - remove common suffixes
    stemmed_words = []
    for word in words:
        # Basic stemming rules
        if word.endswith('ing'):
            word = word[:-3]
        elif word.endswith('ly'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-3] + 'y'
        elif word.endswith('es'):
            word = word[:-2]
        elif word.endswith('s') and not word.endswith('ss'):
            word = word[:-1]
        elif word.endswith('ed') and len(word) > 4:
            word = word[:-2]
        if len(word) > 1:  # Only keep words with more than 1 character
            stemmed_words.append(word)
    # Join words back into text
    return ' '.join(stemmed_words)

# Function to extract important keywords from job description
def extract_keywords(text, top_n=15):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Simple approach: count word frequencies
    words = processed_text.split()
    word_freq = {}
    
    for word in words:
        if len(word) > 2:  # Only consider words with more than 2 characters
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    top_keywords = [word for word, freq in sorted_words[:top_n]]
    
    # If we have TfidfVectorizer available, use it as backup
    if not top_keywords and TfidfVectorizer:
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=top_n)
            # Fit and transform the text
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            # Sort by score
            keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keywords.sort(key=lambda x: x[1], reverse=True)
            # Return top keywords
            top_keywords = [keyword for keyword, score in keywords]
        except:
            pass
    
    return top_keywords

# Function to find missing keywords from the job description in the resume
def find_missing_keywords(job_description, resume_text, top_n=15):
    job_keywords = extract_keywords(job_description, top_n)
    resume_text_processed = preprocess_text(resume_text)
    missing_keywords = []
    
    for keyword in job_keywords:
        if keyword not in resume_text_processed.split():
            missing_keywords.append(keyword)
    
    return missing_keywords

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/single")
def single_resume():
    return render_template('single_resume.html')

@app.route("/multiple")
def multiple_resumes():
    return render_template('multiple_resumes.html')

@app.route("/result")
def result():
    return render_template('result.html')

@app.route('/match_single', methods=['POST'])
def match_single():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_file = request.files.get('resume')
        
        if not resume_file or not job_description or resume_file.filename == '':
            return render_template('single_resume.html', message="Please upload a resume and enter a job description.")
        
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        try:
            # Save and process the resume
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(file_path)
            resume_text = extract_text(file_path)
            
            if not resume_text:
                return render_template('single_resume.html', message="Could not extract text from the uploaded resume.")
            
            # Simple matching approach that doesn't rely on complex NLP
            # First, let's do some basic cleaning
            job_text = job_description.lower()
            resume_text_lower = resume_text.lower()
            
            # Try to use the preprocessing if it works
            try:
                processed_job = preprocess_text(job_description)
                processed_resume = preprocess_text(resume_text)
                
                # Use TF-IDF vectorizer if it works
                try:
                    vectorizer = TfidfVectorizer()
                    vectors = vectorizer.fit_transform([processed_job, processed_resume])
                    similarity = cosine_similarity(vectors)[0, 1]
                except:
                    # Fallback to simple word overlap
                    job_words = set(processed_job.split())
                    resume_words = set(processed_resume.split())
                    common_words = job_words.intersection(resume_words)
                    if len(job_words) > 0:
                        similarity = len(common_words) / len(job_words)
                    else:
                        similarity = 0
            except:
                # Very basic fallback
                # Count how many words from job description appear in resume
                job_words = set(w.strip(string.punctuation) for w in job_text.split() if len(w) > 3)
                resume_words = set(w.strip(string.punctuation) for w in resume_text_lower.split() if len(w) > 3)
                common_words = job_words.intersection(resume_words)
                if len(job_words) > 0:
                    similarity = len(common_words) / len(job_words)
                else:
                    similarity = 0
            
            match_percentage = round(similarity * 100, 2)
            
            # Find missing keywords with error handling
            try:
                missing_keywords = find_missing_keywords(job_description, resume_text)
            except:
                # Fallback for keyword extraction
                job_words = [w.strip(string.punctuation) for w in job_text.split() 
                            if len(w) > 3 and w.lower() not in STOPWORDS]
                word_freq = {}
                for word in job_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get most frequent words from job description
                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                missing_keywords = [word for word, _ in top_keywords if word.lower() not in resume_text_lower]
            
            # Generate suggestions
            suggestions = []
            if missing_keywords:
                suggestions.append(f"Your resume is missing these key skills/keywords: {', '.join(missing_keywords[:5])}")
                suggestions.append("Consider adding these skills to your resume if you have experience with them.")
            else:
                suggestions.append("Your resume contains all the important keywords from the job description.")
            
            return render_template('single_result.html',
                                resume_name=resume_file.filename,
                                match_percentage=match_percentage,
                                suggestions=suggestions)
        except Exception as e:
            # If any error occurs, show a simplified version
            return render_template('single_resume.html', 
                                message=f"An error occurred while processing your resume. Please try again with a different file.")
    
    return render_template('single_resume.html')

@app.route('/match_multiple', methods=['POST'])
def match_multiple():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_files = request.files.getlist('resumes')
        
        if not resume_files or not job_description or resume_files[0].filename == '':
            return render_template('multiple_resumes.html', message="Please upload resumes and enter a job description.")
        
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        try:
            resumes = []
            filenames = []
            original_texts = []
            missing_keywords_list = []
            
            # Process each resume with error handling
            for resume_file in resume_files:
                if resume_file.filename:
                    try:
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                        resume_file.save(file_path)
                        extracted_text = extract_text(file_path)
                        
                        if extracted_text:  # Only add if we could extract text
                            original_texts.append(extracted_text)
                            
                            try:
                                # Try with our preprocessing
                                processed_text = preprocess_text(extracted_text)
                                resumes.append(processed_text)
                            except:
                                # Fallback to simple lowercase if preprocessing fails
                                resumes.append(extracted_text.lower())
                                
                            filenames.append(resume_file.filename)
                    except:
                        # Skip files that can't be processed
                        continue
            
            if not resumes:
                return render_template('multiple_resumes.html', 
                                      message="Could not extract text from any of the uploaded files.")
            
            # Find missing keywords for each resume with error handling
            for i, extracted_text in enumerate(original_texts):
                try:
                    missing_keywords = find_missing_keywords(job_description, extracted_text)
                    missing_keywords_list.append(missing_keywords)
                except:
                    # Fallback for keyword extraction
                    job_text = job_description.lower()
                    resume_text = extracted_text.lower()
                    
                    # Simple keyword extraction
                    job_words = [w.strip(string.punctuation) for w in job_text.split() 
                                if len(w) > 3 and w.lower() not in STOPWORDS]
                    word_freq = {}
                    for word in job_words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                    
                    # Get most frequent words from job description
                    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                    missing_kw = [word for word, _ in top_keywords if word.lower() not in resume_text]
                    missing_keywords_list.append(missing_kw)
            
            # Calculate similarities with error handling
            try:
                # Try with TF-IDF vectorizer
                processed_job = preprocess_text(job_description)
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([processed_job] + resumes)
                job_vector = vectors[0:1]
                resume_vectors = vectors[1:]
                similarities = cosine_similarity(job_vector, resume_vectors)[0]
            except:
                # Fallback to simple word overlap
                similarities = []
                job_text = job_description.lower()
                job_words = set(w.strip(string.punctuation) for w in job_text.split() if len(w) > 3)
                
                for resume_text in resumes:
                    resume_words = set(w.strip(string.punctuation) for w in resume_text.split() if len(w) > 3)
                    common_words = job_words.intersection(resume_words)
                    if len(job_words) > 0:
                        similarity = len(common_words) / len(job_words)
                    else:
                        similarity = 0
                    similarities.append(similarity)
            
            # Create a list of resumes with their scores
            resume_scores = list(zip(filenames, similarities))
            
            # Sort resumes by score (descending)
            resume_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 5 resumes (or fewer if less than 5 were uploaded)
            max_results = min(5, len(resumes))
            top_resumes = resume_scores[:max_results]
            
            # Format the results
            top_resumes_with_scores = [(name, round(score * 100, 2)) for name, score in top_resumes]
            
            # Create suggestions for each top resume
            suggestions = []
            for i, (name, _) in enumerate(top_resumes):
                idx = filenames.index(name)
                missing_kw = missing_keywords_list[idx]
                
                if missing_kw:
                    suggestions.append(f"Resume '{name}' is missing these keywords: {', '.join(missing_kw[:5])}" + 
                                    (f" and {len(missing_kw) - 5} more" if len(missing_kw) > 5 else ""))
                else:
                    suggestions.append(f"Resume '{name}' includes all the important keywords from the job description.")

            return render_template('multiple_result.html', 
                                top_resumes_with_scores=top_resumes_with_scores, 
                                suggestions=suggestions)
        
        except Exception as e:
            # If any error occurs, show a simplified version
            return render_template('multiple_resumes.html', 
                                message=f"An error occurred while processing the resumes. Please try again.")
    
    return render_template('multiple_resumes.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

@app.route('/downloads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Bind to 0.0.0.0 instead of localhost to make it accessible externally
    app.run(host='0.0.0.0', port=port, debug=True)