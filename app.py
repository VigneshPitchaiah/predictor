from flask import Flask, render_template, request, url_for
import joblib
from l1_reg import L1RegularizedLinearRegression


app = Flask(__name__,template_folder='app')

model = joblib.load('app/l1_regularized_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            years_experience = float(request.form['years_experience'])
            prediction = model.predict(years_experience)
            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return render_template('error.html', error=e)

if __name__ == '__main__':
    app.run(debug=True)
