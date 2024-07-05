from flask import Flask,request,render_template
from source.main_project.pipeline.predict_pipeline import PredicPipeline,UserData
import numpy as np

app = Flask(__name__)
@app.route('/')
def homepage():
    return render_template('salarypred.html')

@app.route('/predict',methods=['POST'])
def do_prediction():
    user_input = UserData(
        designation=request.form.get('des'),
        age=request.form.get('age'),
        unit=request.form.get('unit'),
        ratings=request.form.get('rate'),
        experience=request.form.get('exp')
    )
    user_df = user_input.get_data_as_df()
    
    predictpipe = PredicPipeline()
    results = np.round(predictpipe.predict(user_df),2)
    
    msg = f"Estimated salary is ${results}"
    
    return render_template('salarypred.html',text=msg)

if __name__ == "__main__":
    app.run(debug=True)