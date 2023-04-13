import os
import cv2
import requests
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

IMG_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'gif', 'png', 'jpg', 'jpeg'}
IMG_HASHER = cv2.img_hash.BlockMeanHash_create()

app = Flask(__name__)
app.config['IMG_FOLDER'] = IMG_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000


def allowed_file(fname) -> bool:
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/bad_submission')
def bad_submission():
    return '''
            <!doctype html>
            <style>
                p {text-align: center;} 
            </style>
            <p>Invalid Image File Type.</p>
            <center> <button onclick="history.back()">Go Back</button> <br> <center>
        '''
     
@app.route('/invalid_submission')   
def invalid_submission():
    return '''
            <!doctype html>
            <style>
                p {text-align: center;} 
            </style>
            <p>Invalid Submission: No Image or Public Key was submitted.</p> <br>
            <center> <button onclick="history.back()">Go Back</button> <center>
        '''
        
@app.route('/submitted')
def submitted():
    return '''
            <!doctype html>
            <style>
                p {text-align: center;} 
            </style>
            <p>Report is submitted.</p>
            <center> <button onclick="history.back()">Go Back</button> <br> <center>
        '''

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        FILE = request.files['image']
        PKEY = request.form.get('public_key')
        
        if FILE.filename == '' or PKEY == '':
            return redirect(url_for('invalid_submission'))
        
        if allowed_file(FILE.filename):
            fname = secure_filename(FILE.filename)
            # instead of saving, we want to predict and send here
            

            return redirect(url_for('submitted'))
        else:
            return redirect(url_for('bad_submission'))
    
    return '''

        <head>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size: 18px;
                    background-color: #F4F4F4;
                    margin: 0;
                    padding: 0;
                }
                
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 85vh;
                }
                
                form {
                    width: 50%;
                    margin: 20px auto;
                    padding: 20px;
                    background-color: #FFFFFF;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }
                
                label {
                    display: inline-block;
                    margin-bottom: 10px;
                    font-weight: bold;
                }
                
                input[type="text"], input[type="file"] {
                    padding: 5px;
                    border: none;
                    border-radius: 3px;
                    box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
                    width: 100%;
                    margin-bottom: 20px;
                    font-size: 20px;
                }
                
                input[type="file"] {
                    background-color: #FFFFFF;
                    color: #555555;
                    cursor: pointer;
                    display: inline-block;
                    font-size: 20px;
                    width: auto;
                    padding: 10px 20px;
                }
                
                button[type="submit"] {
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 3px;
                    font-size: 20px;
                    cursor: pointer;
                    transition: background-color 0.3s ease-in-out;
                    margin-left:155px;
                }
                
                button[type="submit"]:hover {
                    background-color: #2E8B57;
                }

                .logo {
                    font-size: 48px;
                    font-weight: bold;
                    color: #006400;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 20px;
                    font-family: 'Lucida Handwriting', cursive;
                    letter-spacing: 5px;
                }

                img {
                    width: 300px;
                    max-height: 300px;
                    margin-top: 10px;
                }
            </style>
        </head>
    
        <body>
        
            <div class="container">
                <div class="logo"><span style="color: #ec6464; font-size:60px">C</span>rime<span style="color: #ec6464; font-size:60px">C</span>anary</div>
                <form>
                    <ul style="list-style-type:none;">
                        <li>
                            <label for="publickey"> Public Key </label>
                            <input type="text" id="publickey" name="public_key" required/>
                        </li>
                        <li>
                            <label for="image"> Image </label>
                            <input type="file" id="image" name="image" accept="image/*" required/>
                        </li>
                        <li class="button">
                            <button type="submit"> Submit </button>
                        </li>
                    </ul>
                </form>
            </div>

            <script>
                const imageInput = document.querySelector('#image');
                const imagePreview = document.createElement('img');
                imagePreview.setAttribute('alt', 'Image preview');
                
                imageInput.addEventListener('change', function() {
                    const file = this.files[0];

                    if (file) {
                        const reader = new FileReader();

                        reader.addEventListener('load', function() {
                            imagePreview.setAttribute('src', this.result);
                        });

                        reader.readAsDataURL(file);

                        document.querySelector('form').appendChild(imagePreview);
                    } else {
                        imagePreview.setAttribute('src', '');
                    }
                });
            </script>

        </body>
 	
    '''
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)