from flask import Flask, render_template, request
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import io
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64

import AD_analysis

from lib.utils import compute_morganfps, create_model, morgan_csv, csv_result, format_smiles

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'


#processing for applicability domain
train_data = pd.read_csv('./data/train_smiles.csv')['PUBCHEM_EXT_DATASOURCE_SMILES'].to_list()
test_data = pd.read_csv('./data/test_smiles.csv')['PUBCHEM_EXT_DATASOURCE_SMILES'].to_list()


ad_plot = AD_analysis.AD(train_data)

@app.route("/")
def hello_world():
    return render_template('home.html')


@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/tutorial")
def tutorial():
    return render_template('tutorial.html')

@app.route("/result",  methods = ['Post'])
def result():
    smile = request.form.get('smile')
    model_type = request.form.get('model')

    # getting molecule from smile
    mol = Chem.MolFromSmiles(smile)
    error = mol == None
    if error:
        return render_template("upload.html", error = error)
    
    # get molecule image
    try:
        cpd_image = Draw.MolToImage(mol, returnPNG=True)
        cpd_buffer = BytesIO()
        cpd_image.save(cpd_buffer, format="PNG")
        cpd_image = base64.b64encode(cpd_buffer.getvalue()).decode()
    except:
        return "could not generate image from smile"

    data = compute_morganfps(mol)
    model = create_model(model_type)

    print(model.predict(data))
    print(model.predict_proba(data))
    # get model

    prediction = model.predict(data)

    #generating and displaying applicability domain picture
    fig, ax, withinAD = ad_plot.plot_distance(test_data, threshold=0.04, input_info=smile)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')

    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    if prediction[0] == 1:
        activity = "active"
        confidence = "{:.2%}".format(model.predict_proba(data)[0][1])
    else:
        activity = "inactive"
        confidence = "{:.2%}".format(model.predict_proba(data)[0][0])

    isWithin = True


    return render_template('result.html', smile=smile, activity=activity, confidence=confidence, isWithin=isWithin, cpd_image=cpd_image, encoded_img=encoded_image)


@app.route('/results_csv', methods = ['Post', 'Get'])
def results_csv():
    if request.method == "POST":
        #get csv file and model to use from request object
        file = request.files["smile_csv"]
        model_type = request.form.get('model')

        #select model type to use
        model = create_model(model_type)

        
        
        descriptors, smiles = morgan_csv(file)

        #is Within
        _, _, isWithin = ad_plot.plot_distance(test_data, threshold=0.04, input_info=list(smiles))

        df_table = csv_result(descriptors, smiles, model, isWithin)

        
        styled_df = df_table.style  \
                            .format({'Canonical Smiles' : format_smiles})

        data_table = styled_df.to_html(escape = False)

        return render_template("result_csv.html", df_html = data_table)
    else:
        return "Method not allowed"