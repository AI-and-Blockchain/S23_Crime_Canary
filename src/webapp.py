import os 
from flask import Flask, flash, request, redirect, url_for, render_template, abort
from werkzeug.utils import secure_filename

IMG_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'gif', 'png', 'jpg', 'jpeg'}

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
        
            print(fname)
            
            # instead of saving, we want to predict and send here

            return redirect(url_for('submitted'))
        else:
            return redirect(url_for('bad_submission'))
    
    return '''
        <form method="post" enctype=multipart/form-data>
        <center>
        <ul style="list-style-type:none;">
            <li>
                <label for="publickey"> Public Key: </label>
                <input type="text" id="publickey" name="public_key" />
            </li> <br>
            <li>
                <label for="image"> Image: </label>
                <input type="file" class="hidden" id=image name="image" />
            </li> <br>
            <li class="button">
                <button type="submit"> Submit </button>
            </li>
        </ul></center>
        </form>
    '''
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)