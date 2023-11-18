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
    template=''' Analyze the provided resume text closely and generate feedback that is finely tuned to the individual's experiences and qualifications. Tailor your recommendations to directly reflect the content of each resume section:

1. Professional Summary: Examine the professional summary to identify the individual's key strengths and career ambitions. Offer tailored advice on how to make this summary more dynamic and reflective of their unique background and experience. Highlight any specific phrases or accomplishments that should be emphasized.

2. Work Experience: Review the work experience section in detail. Provide targeted suggestions for enhancing the presentation of specific roles, responsibilities, and achievements. Point out any particular experiences that stand out and how they could be more effectively showcased.

4. Skills: Assess the listed skills critically. Identify and suggest essential skills that align closely with the job field and may be missing or underplayed. Offer advice on how to balance technical, soft, and transferable skills to present a well-rounded skill set.

5. ATS Optimization: Evaluate the resume for the strategic use of keywords and phrases that are pertinent to the job field and for ATS optimization. Advise on keyword enhancements or alterations that could improve the resume's visibility in Applicant Tracking Systems, making sure to reference specific terms currently used in the resume.

6. General Feedback: Provide overall feedback that is specific to the resume's content. Where can clarity be improved? Are there areas that are overly detailed or too vague? Offer concise recommendations that align with the individual's evident goals and experiences as presented in the resume.

In your analysis, refer explicitly to the resume text provided as {pdf_text}, ensuring your feedback is as relevant and personalized as possible to the actual content of the resume.
    '''
 # Your resume analysis prompt template
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
    analysis_output = resume_chain.run(pdf_text=pdf_text)

    return jsonify({"analysis": analysis_output})
