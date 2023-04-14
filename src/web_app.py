import os
import torch
import numpy as np
from PIL import Image
from functools import partial
from string import Template
from model import SiameseNetwork
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for
from model_utils import get_models, predict, resize_image
from oracle_utils import call_app

from algo_utils import (
    verify_transaction, hash_image, get_testnet_indexer, get_testnet_client, get_default_wallet, make_send_transaction
)

INDEXER = get_testnet_indexer()
CLIENT = get_testnet_client()

IMG_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'gif', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['IMG_FOLDER'] = IMG_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

DEFAULT_WALLET_PKEY, DEFAULT_WALLET_SKEY = get_default_wallet()
DEFAULT_ASSET_ID = 176869006

MODELS = get_models(['../saved_models/accident_best_model.pth', '../saved_models/severity_best_model.pth'])
predict_fn = partial(predict, models = MODELS)


def allowed_file(fname: str) -> bool:
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen_response(apred: int, spred: int):
    responseA = responseB = ''
    
    if apred == 1:
        responseA = 'an accident has occurred.'
        
        if spred == 0:
            responseB = 'Severity of the accident is low. You are rewarded with 1 Canary tokens.'
        elif spred == 1:
            responseB = 'Severity of the accident is medium. You are rewarded with 2 Canary tokens.'
        else:
            responseB = 'Severity of the accident is high. You are rewarded with 3 Canary tokens.'
    else:
        responseA = 'an accident has not occurred.'
        
    return responseA, responseB


@app.route('/bad_submission')
def bad_submission():
    return '''
            <!doctype html>
            <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {
                            background-color: #f8f8f8;
                            font-family: Arial, sans-serif;
                        }
                        .container {
                            width: 60%;
                            margin: 0 auto;
                            margin-top: 250px;
                            text-align: center;
                            padding: 20px;
                            background-color: #ffffff;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                            border-radius: 5px;
                            border: 1px solid #e0e0e0;
                        }
                        p {
                            font-size: 1.2em;
                        }
                        .error-icon:before {
                            content: "\26Error !!";
                            color: #FFA500;
                            font-size: 1em;
                            margin-right: 10px;
                        }
                        button {
                            background-color: #4CAF50;
                            border: none;
                            color: #ffffff;
                            padding: 10px 20px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin-top: 20px;
                            border-radius: 5px;
                            cursor: pointer;
                        }
                        button:hover {
                            background-color: #3e8e41;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <p><span class="error-icon"></span></p>
                        <p>Invalid Image File Type</p>
                        <button onclick="history.back()">Go Back</button>
                    </div>
                </body>
            </html>
        '''
        

@app.route('/invalid_transaction')
def invalid_transaction():
        return '''
            <!doctype html>
            <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {
                            background-color: #f8f8f8;
                            font-family: Arial, sans-serif;
                        }
                        .container {
                            width: 60%;
                            margin: 0 auto;
                            margin-top: 250px;
                            text-align: center;
                            padding: 20px;
                            background-color: #ffffff;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                            border-radius: 5px;
                            border: 1px solid #e0e0e0;
                        }
                        p {
                            font-size: 1.2em;
                        }
                        .error-icon:before {
                            content: "\26Error !!";
                            color: #FFA500;
                            font-size: 1em;
                            margin-right: 10px;
                        }
                        button {
                            background-color: #4CAF50;
                            border: none;
                            color: #ffffff;
                            padding: 10px 20px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin-top: 20px;
                            border-radius: 5px;
                            cursor: pointer;
                        }
                        button:hover {
                            background-color: #3e8e41;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <p><span class="error-icon"></span></p>
                        <p>Something went wrong with the transaction. Make sure that you had opted in.</p>
                        <button onclick="history.back()">Go Back</button>
                    </div>
                </body>
            </html>
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
        
        
@app.route('/invalid_publickey')   
def invalid_publickey():
    return '''
            <!doctype html>
            <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {
                            background-color: #f8f8f8;
                            font-family: Arial, sans-serif;
                        }
                        .container {
                            width: 60%;
                            margin: 0 auto;
                            margin-top: 250px;
                            text-align: center;
                            padding: 20px;
                            background-color: #ffffff;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                            border-radius: 5px;
                            border: 1px solid #e0e0e0;
                        }
                        p {
                            font-size: 1.2em;
                        }
                        .error-icon:before {
                            content: "\26Error !!";
                            color: #FFA500;
                            font-size: 1em;
                            margin-right: 10px;
                        }
                        button {
                            background-color: #4CAF50;
                            border: none;
                            color: #ffffff;
                            padding: 10px 20px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin-top: 20px;
                            border-radius: 5px;
                            cursor: pointer;
                        }
                        button:hover {
                            background-color: #3e8e41;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <p><span class="error-icon"></span></p>
                        <p>You either sent an incorrect public key or you did not confirm your report.</p>
                        <button onclick="history.back()">Go Back</button>
                    </div>
                </body>
            </html>
        '''
        
@app.route('/submitted/<apred>-<spred>')
def submitted(apred, spred):
    A, B = gen_response(int(apred), int(spred))
    
    html = '''\
           
            <!doctype html>
            <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {
                            background-color: #f8f8f8;
                            font-family: Arial, sans-serif;
                        }
                        .container {
                            width: 60%;
                            margin: 0 auto;
                            margin-top: 250px;
                            text-align: center;
                            padding: 20px;
                            background-color: #ffffff;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                            border-radius: 5px;
                            border: 1px solid #e0e0e0;
                        }
                        p {
                            font-size: 1.2em;
                        }
                        .success-icon:before {
                            content: "\Sucess !!";
                            color: green;
                            font-size: 1em;
                            margin-right: 10px;
                        }
                        button {
                            background-color: #4CAF50;
                            border: none;
                            color: #ffffff;
                            padding: 10px 20px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin-top: 20px;
                            border-radius: 5px;
                            cursor: pointer;
                        }
                        button:hover {
                            background-color: #3e8e41;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <p><span class="success-icon"></span></p>
                        <p>Report is submitted.</p>
                        <p>Based on the submitted image, $accident $severity</p>
                        <center> <button onclick="history.back()">Go Back</button> <br> <center>
                    </div>
                </body>
            </html>

        '''
    
    return Template(html).safe_substitute(accident = A, severity = B)


@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        FILE = request.files['image']
        PKEY = request.form.get('public_key')
        
        if FILE.filename == '' or PKEY == '':
            return redirect(url_for('invalid_submission'))
        
        if allowed_file(FILE.filename) and allowed_file(secure_filename(FILE.filename)):        
            
            image = Image.open(FILE.stream)
            img_hash = hash_image(np.asarray(image))
            
            if not(verify_transaction(img_hash, PKEY, INDEXER)):
                return redirect(url_for('invalid_publickey'))
            
            img = resize_image(image)
            apred, spred = predict_fn(img)
            
            if apred == 1:
                tokens = spred + 1
                
                flag = make_send_transaction(
                    CLIENT, DEFAULT_WALLET_PKEY, PKEY, DEFAULT_ASSET_ID, DEFAULT_WALLET_SKEY, tokens
                )
                app_id = 190577661
                app_args = [b"notify_classification", intToBytes(apred), intToBytes(spred)]
                call_app(CLIENT, DEFAULT_WALLET_PKEY, DEFAULT_WALLET_SKEY, app_id, app_args, [PKEY])
                
                if not(flag):
                    return redirect(url_for('invalid_transaction'))
    
            return redirect(url_for('submitted', apred=apred, spred=spred))
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
                <form method="post" enctype=multipart/form-data>
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
    port = int(os.environ.get('PORT', 5005))
    app.run(host="0.0.0.0", port=port)