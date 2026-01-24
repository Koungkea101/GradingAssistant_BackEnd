from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from groq import Groq

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Lazy initialization of Groq client
def get_groq_client():
    """Get or create Groq client instance"""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return Groq(api_key=api_key)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/grade', methods=['POST'])
def grade_answer():
    """
    Grade a student's answer against a question
    Expected JSON body:
    {
        "question": "What is the capital of France?",
        "student_answer": "Paris",
        "rubric": "Optional grading criteria" (optional)
    }
    """
    try:
        data = request.json
        question = data.get('question', '')
        student_answer = data.get('student_answer', '')
        rubric = data.get('rubric', '')

        if not question or not student_answer:
            return jsonify({"error": "Both question and student_answer are required"}), 400

        # Get Groq client
        client = get_groq_client()

        # Construct the grading prompt
        if rubric:
            prompt = f"""You are an expert grading assistant. Grade the following student answer.

Question: {question}

Grading Rubric: {rubric}

Student's Answer: {student_answer}

Please provide:
1. A score (0-100)
2. Detailed feedback on what's correct
3. Detailed feedback on what's incorrect or missing
4. Suggestions for improvement
5. A corrected/ideal answer

Format your response as JSON with the following structure:
{{
    "score": <number 0-100>,
    "feedback_correct": "<what the student got right>",
    "feedback_incorrect": "<what needs improvement>",
    "suggestions": "<specific suggestions>",
    "corrected_answer": "<ideal answer>"
}}"""
        else:
            prompt = f"""You are an expert grading assistant. Grade the following student answer.

Question: {question}

Student's Answer: {student_answer}

Please provide:
1. A score (0-100)
2. Detailed feedback on what's correct
3. Detailed feedback on what's incorrect or missing
4. Suggestions for improvement
5. A corrected/ideal answer

Format your response as JSON with the following structure:
{{
    "score": <number 0-100>,
    "feedback_correct": "<what the student got right>",
    "feedback_incorrect": "<what needs improvement>",
    "suggestions": "<specific suggestions>",
    "corrected_answer": "<ideal answer>"
}}"""

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert grading assistant. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",  # Free model on Groq
            temperature=0.3,  # Lower temperature for more consistent grading
            max_tokens=1024,
            response_format={"type": "json_object"}  # Ensure JSON response
        )

        # Extract and parse the JSON response
        result_string = chat_completion.choices[0].message.content
        result = json.loads(result_string)

        return jsonify({
            "success": True,
            "score": result.get("score"),
            "feedback_correct": result.get("feedback_correct"),
            "feedback_incorrect": result.get("feedback_incorrect"),
            "suggestions": result.get("suggestions"),
            "corrected_answer": result.get("corrected_answer")
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/correct', methods=['POST'])
def correct_answer():
    """
    Correct a student's answer without detailed grading
    Expected JSON body:
    {
        "question": "What is the capital of France?",
        "student_answer": "Paris is capital"
    }
    """
    try:
        data = request.json
        question = data.get('question', '')
        student_answer = data.get('student_answer', '')

        if not question or not student_answer:
            return jsonify({"error": "Both question and student_answer are required"}), 400

        # Get Groq client
        client = get_groq_client()

        prompt = f"""Given the following question and student answer, provide a corrected version of the answer.

Question: {question}

Student's Answer: {student_answer}

Provide a clear, concise, and grammatically correct version of the answer. Only return the corrected answer text without any additional explanation."""

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=512
        )

        corrected = chat_completion.choices[0].message.content.strip()

        return jsonify({
            "success": True,
            "corrected_answer": corrected
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# help OCR to fix some words that doesnt make sense
@app.route('/adjust_ocr', methods=['POST'])
def adjust_ocr():
    """
    Correct OCR output text by fixing incorrect words and improving readability
    Expected JSON body:
    {
        "ocr_text": "The qick brown fox jmps over the lazy dog",
        "context": "Optional context about the document type" (optional)
    }
    """
    try:
        data = request.json
        ocr_text = data.get('ocr_text', '')
        context = data.get('context', '')

        if not ocr_text:
            return jsonify({"error": "ocr_text is required"}), 400

        # Get Groq client
        client = get_groq_client()

        # Construct the OCR correction prompt
        if context:
            prompt = f"""You are an OCR text correction expert. The following text was extracted using OCR and contains errors. Please correct spelling mistakes, fix garbled words, and improve readability while preserving the original meaning.

Context: {context}

OCR Text to correct:
{ocr_text}

Please provide the corrected text that:
1. Fixes spelling errors
2. Corrects obvious OCR mistakes (like 'rn' misread as 'm', '0' as 'O', etc.)
3. Maintains the original structure and formatting
4. Preserves the original meaning
5. Uses proper grammar and punctuation
6. Without adding new words not present in the original text

Only return the corrected text without any additional explanation or commentary."""
        else:
            prompt = f"""You are an OCR text correction expert. The following text was extracted using OCR and contains errors. Please correct spelling mistakes, fix garbled words, and improve readability while preserving the original meaning.

OCR Text to correct:
{ocr_text}

Please provide the corrected text that:
1. Fixes spelling errors
2. Corrects obvious OCR mistakes (like 'rn' misread as 'm', '0' as 'O', etc.)
3. Maintains the original structure and formatting
4. Preserves the original meaning
5. Uses proper grammar and punctuation
6. Without adding new words not present in the original text

Only return the corrected text without any additional explanation or commentary."""

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCR text correction assistant. Correct OCR errors while preserving the original meaning and structure."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,  # Very low temperature for consistent corrections
            max_tokens=2048
        )

        # Extract the corrected text
        corrected_text = chat_completion.choices[0].message.content.strip()

        return jsonify({
            "success": True,
            "original_text": ocr_text,
            "corrected_text": corrected_text
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
