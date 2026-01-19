from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Lazy initialization of Groq client
# Get your free API key from: https://console.groq.com/keys
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

        # Extract the response
        result = chat_completion.choices[0].message.content

        return jsonify({
            "success": True,
            "grading_result": result
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

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
