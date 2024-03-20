from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('Parkinsons_Disease\model.pkl', 'rb'))

# Assuming you know the required number of features (22 in this case)
NUM_FEATURES = 22

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form['text']
        input_text_sp = input_text.split(',')
        
        # Check if input has less than required features
        if len(input_text_sp) < NUM_FEATURES:
            # Pad input with zeros to match the required number of features
            input_text_sp += ['0'] * (NUM_FEATURES - len(input_text_sp))
        
        np_data = np.asarray(input_text_sp, dtype=np.float32)
        prediction = model.predict(np_data.reshape(1, -1))

        if prediction == 1:
            output = "This person has Parkinson's Disease."
        else:
            output = "This person does not have Parkinson's Disease."
    except Exception as e:
        output = f"Error: {str(e)}"

    return render_template('index.html', message=output)

if __name__ == "__main__":
    app.run(debug=True)
