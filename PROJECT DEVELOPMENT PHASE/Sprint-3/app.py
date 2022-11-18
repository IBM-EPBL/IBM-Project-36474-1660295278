from cloudant.client import Cloudant
import os 
import tensorflow 
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np 
from keras.models import load_model
from tensorflow.python.ops.gen_array_ops import concat
from keras.applications.inception_v3 import preprocess_input

# Authenticating using an IAM API key
client = Cloudant.iam("3450975b-5d47-4536-99ca-71c6f554b8fa-bluemix", "uUcTSmT_SxhQtV3mp9Fz1UnYTvUYAAV7A65fbJWjSO2K", connect=True)

# Creating a database using an initialized client
my_database = client.create_database('my_database')

# Loading the models
model1 = load_model('D:\\Nalaiya Thiran Project\\Juypter\\app_model\\model\\body.h5')
model2 = load_model('D:\\Nalaiya Thiran Project\\Juypter\\app_model\\model\\level.h5')

from flask import Flask,render_template,request,redirect,url_for

app = Flask(__name__)

# Home page setting 

@app.route('/')
def home():
    return render_template('index.html')

# Process page setting 

@app.route('/process')
def process():
    return render_template('process.html')

# Register page setting 

@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/afterRegister',methods=['POST'])
def afterregister():
    x = [x for x in request.form.values()]
    print(x)
    data = {
        '_id':x[1],
        'name':x[0],
        'psw' : x[2]
    }
    print(data)

    query = {'_id':{'$eq' : data['_id']}}
    docs = my_database.get_query_result(query)

    if(len(docs.all())==0):
        url = my_database.create_document(data)
        return render_template('register.html', message="Registration is Successfully Completed")
    else:
        return render_template("register.html", message="You are already a member!")


# Login page setting 

@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/afterLogin',methods=['POST','GET'])
def afterlogin():
    user = request.form['_id']
    passw = request.form['psw']
    print(user,passw)

    query = {'_id':{'$eq':user}}

    docs = my_database.get_query_result(query)
    print(docs)
    print(len(docs.all()))

    if(len(docs.all())==0):
        return render_template('login.html',message='The username is not found')
    else:
        if((user==docs[0][0]['_id'] and passw==docs[0][0]['psw'])):
            return redirect(url_for('process'))
        else:
            return render_template("login.html",message="Invalid User Details")
    

# Prediction

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Logout page 

@app.route('/logout')
def logout():
    return render_template('logout.html')

# Results 

@app.route('/result', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['_file']
        basepath = os.path.dirname(__name__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img = load_img(filepath,target_size=(224,224))
        x = img_to_array(img)
        x = np.expand_dims(x,axis=0)
        img_data = preprocess_input(x)

        prediction1 = np.argmax(model1.predict(img_data))
        prediction2 = np.argmax(model2.predict(img_data))

        index1 = ['front','rear','side']
        index2 = ['minor','moderate','severe']

        result1 = index1[prediction1]
        result2 = index2[prediction2]

        if(result1=="front" and result2=="minor"):
            value= "3000 - 5000 Inr (Minor Damage in Front)"
        elif(result1=="front" and result2=="moderate"):
            value ="6000 - 8000 Inr (Moderate Damage in Front)"
        elif(result1=="front" and result2=="severe"):
            value="9000 - 11000 Inr (Severe Damage in Front)"
        elif(result1=="rear" and result2=="minor"):
            value="4000 to 6000 Inr (Minor Damage in Rear)"
        elif(result1=="rear" and result2=="moderate"):
            value="7000 - 9000 Inr (Moderate Damage in Rear)"
        elif(result1=="rear" and result2=="severe"):
            value="11000 - 13000 Inr (Severe Damage in Rear)"
        elif(result1=="side" and result2=="minor"):
            value="6000 - 8000 Inr (Minor Damage in Side)"
        elif(result1=="side" and result2=="moderate"):
            value="9000 - 11000Inr (Moderate Damage in Side)"
        elif(result1=="side" and result2=="severe"):
            value="12000 - 15000 Inr (Severe Damage in Side)"
        else:
            value = "16000 - 50000 Inr (Damage beyond critical)"
        
        return render_template("prediction.html",prediction=value)



if (__name__ == '__main__'):
    app.run(debug=True) 