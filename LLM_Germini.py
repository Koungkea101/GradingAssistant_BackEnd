from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from google import genai

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

def get_germini_client():
    """Get or create Gemini client instance"""
    api_key = os.environ.get("GERMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GERMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)

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

        # Get Gemini client
        client = get_germini_client()

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
        # Call Gemini API
        chat_completion = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a helpful grading assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )

        #extract result from response
        result_string = chat_completion.choices[0].message.response
        result = json.loads(result_string)

        return jsonify({
            "score": result.get("score"),
            "feedback_correct": result.get("feedback_correct"),
            "feedback_incorrect": result.get("feedback_incorrect"),
            "suggestions": result.get("suggestions"),
            "corrected_answer": result.get("corrected_answer"),
            "full_response": result
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

        # Get Gemini client
        client = get_germini_client()

        # Construct the correction prompt
        prompt = f"""You are an expert grading assistant. Correct the following student answer.

Question: {question}

Student's Answer: {student_answer}

Provide a clear, concise, and grammatically correct version of the answer. Only return the corrected answer text without any additional explanation."""
        # Call Gemini API
        chat_completion = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a helpful grading assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
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
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
