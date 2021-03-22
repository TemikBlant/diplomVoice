from flask import Flask
from flask import request
import recognizer

temp_dir = 'temp'

app = Flask(__name__)
r = recognizer.Recognizer(temp_dir=temp_dir)
r.load_models()


@app.route('/upload', methods=['POST'])
def rec():
    if request.method == 'POST':
        data = request.data
        f = open(temp_dir + '/test.wav', 'wb')
        f.write(data)
        f.close()
        response = r.recognize()
        return str(response)
    return 'who'


if __name__ == '__main__':
    app.run()
