from flask import Flask, render_template
from flask import send_from_directory
import subprocess

app = Flask(__name__)


@app.route('/login-signup')
def login_signup():
    return send_from_directory('static', 'Sign_in&sign_up.html')
@app.route('/')
def index():
    return render_template('new2.html')

# @app.route('/login-signup')
# def login_signup():
#     return render_template('static/Sign_in&sign_up.html')

@app.route('/run-script')
def run_script():
    try:
        # Execute the command to run your Python script
        command = 'python -u -Xutf8 "c:\\Users\\vishv\\Documents\\New folder\\face expression\\main.py"'
        subprocess.run(command, shell=True)
        return 'Happiness is contagious, keep spreading joy!'
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
