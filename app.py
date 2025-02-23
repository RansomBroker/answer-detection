from flask import Flask
from routes import api  # Import blueprint

app = Flask(__name__)

# Register blueprint without prefix or with prefix if desired
app.register_blueprint(api)

if __name__ == '__main__':
    app.run(debug=True)