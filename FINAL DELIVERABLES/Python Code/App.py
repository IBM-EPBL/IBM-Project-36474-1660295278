import re
import numpy as np
import os
from flask import Flask, app, request, render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for

# #Loading the model
from cloudant.client import Cloudant


# Authenticate using an IAM API key
client = Cloudant.iam('username','apikey',connect=True)

# Create a database using an initialized client
my_database = client.create_database ('my_database')

model1=load_model('body.h5')
model2=load_model('level.h5')

app=Flask(__name__)

#default home page or route
@app.route('/')
def index():
    return render_template('index.html')

#register page
@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/afterreg',methods=['POST'])
def afterreg():
    x=[x for x in request.form.values()]
    print(x)
    data={
        '_id':x[1],
        'name':x[0],
        'psw':x[2]
    }
    print(data)
    query={'_id':{'$eq':data['_id']}}
    docs=my_database.get_query_result(query)
    print(docs)

    print(len(docs.all()))

    if(len(docs.all())==0):
        url=my_database.create_document(data)
        return render_template('login.html',pred="Registeration successfull, Please login your details")
    else:
        return render_template('register.html',pred="You re already member, Please login using r details")
       

#login page
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/afterlogin',methods=['POST'])
def afterlogin():
    user=request.form['_id']
    password=request.form['psw']
    print(user,password)
    query={'_id':{'$eq':user}}

    docs=my_database.get_query_result(query)    
    print(docs)

    print(len(docs.all()))

    # if(len(docs.all())==0):
    #     return render_template('login.html',pred="The usernmae not found")
    # else:
    if(user==docs[0][0]['_id'] and password==docs[0][0]['psw']):
        return render_template(url_for('prediction'))
    else:
        return render_template('error.html')

#prediction page
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')
#logout page
@app.route('/logout')
def logout():
    return render_template('logout.html')

#prediction
@app.route('/result',methods=["GET","POST"])
def result():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__) #getting the current path Le where app.py #print("current path", basepath)
        filepath=os.path.join(basepath,'uploads',f.filename) #print("upload folder is", filepath)
        f.save(filepath)
        img=image.load_img(filepath, target_size=(224,224))
        x=image.img_to_array(img)#ing to array x-np.expand dims(x,axis-e) used for adding
        x=np.expand_dims(x,axis=0)
        img_data=preprocess_input(x)
        prediction1=np.argmax(model1.predict(img_data))
        prediction2=np.argmax(model2.predict(img_data))

        index1=['front','rear', 'side'] 
        index2=['minor', 'moderate', 'severe']

        result1=index1[prediction1]
        result2=index2[prediction2]

        if(result1=='front' and result2=='minor'):
            value="3000 - 5000 INR"
        elif(result1=='front' and result2=='moderate'):
            value="6000 - 8000 INR"
        elif(result1=='front' and result2=='severe'):
            value="9000 - 11000 INR"
        elif(result1=='rear' and result2=='minor'):
            value="4000 - 6000 INR"
        elif(result1=='rear' and result2=='moderate'):
            value="7000 - 9000 INR"
        elif(result1=='rear' and result2=='severe'):
            value="11000 - 13000 INR"
        elif(result1=='side' and result2=='minor'):
            value="6000 - 8000 INR"
        elif(result1=='side' and result2=='moderate'):
            value="9000 - 11000 INR"
        elif(result1=='side' and result2=='severe'):
            value="12000 - 15000 INR"
        else:
            value="16000 - 50000 INR"
        
        return render_template('prediction.html',value=value)
        

#run app
if __name__ == "__main__":
    app.run(debug=True,port=8080)

