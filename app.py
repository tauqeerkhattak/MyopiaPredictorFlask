from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    query = str(request.args['query'])
    return 'Hello Flutter + Python and '+query


if __name__ == '__main__':
    app.run()
