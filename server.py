#!/usr/bin/env python3
import os
from flask import Flask, render_template, jsonify, request
import json
from src.server.utils.predict import predict


app = Flask(__name__, static_folder = 'static', template_folder = 'templates')
app.config["JSON_AS_ASCII"] = False

#loader = DataLoader("./data", "roadlamp")

# ===============================

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get_result",methods=['POST'])
def get_result():
    data = request.get_json()
    input_ = json.loads(data['data'])
#    print(input_)
#    return ""
    out = predict(input_)
#    print(out)
    for key in out:
        out[key] = str(out[key])
    print(out)
    return jsonify(out)

"""
@app.route("/position", methods=['POST'])
def get_position_by_datatype():
    data = loads(request.data.decode('utf-8')
                 if type(request.data) is bytes else request.data)
    print(data)
    datatype = data["type"]
    return jsonify(loader.get_position_by_datatype(datatype))

@app.route("/ndata", methods=['POST'])
def get_n_lateset_data():
    data = loads(request.data.decode('utf-8')
                 if type(request.data) is bytes else request.data)
    print(data)
    device_id = data["id"]
    datatypes = data["type"]
    n = data["n"]
    return jsonify(loader.get_n_lateset_data(device_id, datatypes, n))

@app.route("/download", methods=['POST'])
def get_download_file():
    data = loads(request.data.decode('utf-8')
                 if type(request.data) is bytes else request.data)
    print(data)
    device_id = data["id"]
    datatypes = data["type"]
    n = data["n"]
    fmt = data["fmt"]
    return jsonify(loader.get_download_file(device_id, datatypes, n, fmt))
"""
# To prevent caching
@app.after_request

def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# ================================

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080), debug=True)
