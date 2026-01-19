# Grading Assistant Backend

A simple Flask API using Groq's free LLM API to grade and correct student answers.

## Setup

1. **Get a free Groq API key:**
   - Visit: https://console.groq.com/keys
   - Sign up (it's free!)
   - Create an API key

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   # Copy the example env file
   cp .env.example .env

   # Edit .env and add your Groq API key
   GROQ_API_KEY=your_actual_key_here
   ```

4. **Run the server:**
   ```bash
   python LLM_main.py
   ```

   The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Grade Answer
```bash
POST /grade
Content-Type: application/json

{
  "question": "What is photosynthesis?",
  "student_answer": "Plants make food from sunlight",
  "rubric": "Must mention: chlorophyll, CO2, water, glucose" // optional
}
```

**Response:**
```json
{
  "success": true,
  "grading_result": {
    "score": 75,
    "feedback_correct": "Correctly identified the basic concept",
    "feedback_incorrect": "Missing key components",
    "suggestions": "Include chlorophyll, CO2, water, and glucose",
    "corrected_answer": "Complete answer..."
  }
}
```

### 3. Correct Answer (Simple)
```bash
POST /correct
Content-Type: application/json

{
  "question": "What is the capital of France?",
  "student_answer": "paris is capital"
}
```

**Response:**
```json
{
  "success": true,
  "corrected_answer": "Paris is the capital of France."
}
```

## Testing with cURL

```bash
# Grade an answer
curl -X POST http://localhost:5000/grade \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 2+2?",
    "student_answer": "4"
  }'

# Correct an answer
curl -X POST http://localhost:5000/correct \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "student_answer": "paris"
  }'
```

## Why Groq?

- **Free:** Generous free tier
- **Fast:** Extremely fast inference speeds
- **Quality:** Access to Llama 3.3 70B and other powerful models
- **No Credit Card:** Sign up without payment info

## Alternative Free LLM APIs

If you prefer other options:

1. **Google Gemini** (free tier): https://ai.google.dev/
2. **Together AI** (free credits): https://www.together.ai/
3. **Hugging Face Inference API**: https://huggingface.co/inference-api

Just replace the Groq client code with the respective API client.
