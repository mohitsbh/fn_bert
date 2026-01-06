from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fake_news_model')

print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model loaded successfully on {device}!")


def predict_fake_news(text):
    """
    Predict whether the given text is fake news or real news.
    Returns prediction label and confidence score.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map prediction to label (0 = Real, 1 = Fake typically, but may vary)
    labels = {0: 'Real News', 1: 'Fake News'}
    prediction = labels.get(predicted_class, f'Class {predicted_class}')
    
    return {
        'prediction': prediction,
        'confidence': round(confidence * 100, 2),
        'is_fake': predicted_class == 1
    }


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for fake news prediction."""
    try:
        # Get text from request
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text.strip():
            return jsonify({
                'error': 'Please provide some text to analyze.'
            }), 400
        
        # Get prediction
        result = predict_fake_news(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
