from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import re

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key
os.environapi_key = os.environ.get('OPENAI_API_KEY')

# LangChain setup
llm = OpenAI(temperature=0.7)
# Assume a similar prompt template for resume analysis
resume_template = PromptTemplate(
    input_variables=['pdf_text'],
    template=''' Analyze the resume, referred to as {pdf_text}, and provide integrated feedback that is deeply relevant to its content. Throughout your analysis, reference the individual, whose name is provided in the resume, to personalize the feedback. Focus on creating a unified and coherent paragraph that interweaves insights from the Professional Summary, Work Experience, Skills, ATS Optimization, and General Feedback. Your response should address the unique strengths and areas for improvement in the resume, making sure the advice is directly tailored to the experiences and qualifications of the individual named in the resume. The goal is to provide clear, actionable advice and specific recommendations. These should be aimed at helping the individual enhance their resume, better highlight their skills and achievements, and address any areas that need improvement, thereby aiding their professional advancement.
    '''
)
resume_chain = LLMChain(llm=llm, prompt=resume_template, verbose=True, output_key='analysis')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file.stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    resume_file = request.files['resume']
    pdf_text = extract_text_from_pdf(resume_file)
    
    # Analyze the resume
    analysis_output = resume_chain.run(pdf_text)

    return jsonify({"analysis": analysis_output})
