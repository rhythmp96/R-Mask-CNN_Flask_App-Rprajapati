# https://youtu.be/pI0wQbJwIIs
"""

Werkzeug provides a bunch of utilities for developing WSGI-compliant applications. 
These utilities do things like parsing headers, sending and receiving cookies, 
providing access to form data, generating redirects, generating error pages when 
there's an exception, even providing an interactive debugger that runs in the browser. 
Flask then builds upon this foundation to provide a complete web framework.
"""

from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from main import getPrediction
import os
from PIL import Image
import numpy as np


#Save images to the 'static' folder as Flask serves images from this directory
UPLOAD_FOLDER = 'static/images'
MASKED_FOLDER = 'static/masked_images'

#Create an app object using the Flask class. 
app = Flask(__name__)

#Add reference fingerprint. 
#Cookies travel with a signature that they claim to be legit. 
#Legitimacy here means that the signature was issued by the owner of the cookie.
#Others cannot change this cookie as it needs the secret key. 
#It's used as the key to encrypt the session - which can be stored in a cookie.
#Cookies should be encrypted if they contain potentially sensitive information.
app.secret_key = "secret key"

#Define the upload folder to save images uploaded by the user. 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASKED_FOLDER'] = MASKED_FOLDER

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, index function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

def resize(file):
    
    mywidth = 800

    img = Image.open(file)
    wpercent = (mywidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    file_R_I = img.resize((mywidth,hsize), Image.ANTIALIAS)
    
    return file_R_I




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mask', methods = ['GET'])
def mask():
    return render_template('mask.html')


#Add Post method to the decorator to allow for form submission. 
@app.route('/', methods=['POST','GET'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            
           
            #print(file_R_I)
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            print(filename)
            print(type(filename))
            image_U = resize(file)
            image_U.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(image)
            print(type(image))
            #getPrediction(filename)
            getPrediction(filename).savefig(os.path.join(app.config['MASKED_FOLDER'],'M_'+filename), bbox_inches='tight', pad_inches=0, transparent=True)
            masked_image = os.path.join(app.config['MASKED_FOLDER'],'M_'+filename)
            print(masked_image)
            print(type(masked_image))
            return render_template('mask.html', image = image, masked_image = masked_image)



if __name__ == "__main__":
    app.run(debug=True, port = 5000)