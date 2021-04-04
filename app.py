from os import sched_get_priority_min
from auth.auth import auth_api, authenticate, identity
from flask import Flask, render_template, request
from model import NeuralNet
from utils import bag_of_words, tokenize
from train import train as t
from flask_jwt import JWT, jwt_required, current_identity
import random
import torch
import requests
import json
from flask_cors import CORS
import boto3
import _thread

import pymongo
from bson.objectid import ObjectId


client = pymongo.MongoClient(
    "mongodb+srv://miketsui3a:aA26761683@cluster0.bnkhm.azure.mongodb.net")
db = client['fyp']
userRepo = db['user']


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'super-secret'
app.config['JWT_AUTH_HEADER_PREFIX'] = 'bearer'
jwt = JWT(app, authenticate, identity)
app.static_folder = 'static'
app.register_blueprint(auth_api, url_prefix='/authenticate')


print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# url = 'https://raw.githubusercontent.com/python-engineer/pytorch-chatbot/master/intents.json'
# response = requests.get(url).text
intent = {
    "intent": json.load(open('./intents.json'))
}
# intent = json.loads(response)

FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

models = []
models_dict = {}

# model = NeuralNet(hidden_size=data["hidden_size"],
#                   input_size=data["input_size"], num_classes=data["output_size"])
# model.load_state_dict(data["model_state"])
# model.eval()

# models.append(model)


@app.route('/get', methods=['GET'])
@jwt_required()
def chat():
    text = tokenize(request.args.get('msg'))
    print(text)
    text = bag_of_words(text, models_dict[current_identity['identity']]["all_words"])
    text = text.reshape(1, text.shape[0])
    text = torch.from_numpy(text).to(device)
    out = models_dict[current_identity['identity']]["model"](text)

    _, predicted = torch.max(out, dim=1)
    tag = data['tags'][predicted.item()]

    return {
        "status": 200,
        "message": random.choice(models_dict[current_identity['identity']]["data"]['intents'][predicted.item()]['responses'])
    }


@app.route('/upload-model', methods=['POST'])
@jwt_required()
def upload_model():
    print("ffffff")
    d = torch.load(request.files['file'], map_location=torch.device('cpu'))
    print(d["hidden_layers"])
    model = NeuralNet(
        hidden_size=d["hidden_size"], input_size=d["input_size"], num_classes=d["output_size"], hidden_layers=d["hidden_layers"])
    model.load_state_dict(d["model_state"])
    model.eval()

    print(model)

    models_dict[current_identity['identity']] = {
        "model": model,
        "all_words": d['all_words'],
        "tags": d["tags"],
        "data": d["data"]
    }
    # print(models_dict[current_identity['identity']])
    return {
        "status": 200
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/train', methods=['POST'])
@jwt_required()
def train():
    if request.files['file'].filename != '':
        uploadFile = json.load(request.files['file'])
        intent['intent'] = uploadFile

        hidden_layers = json.loads(request.form["hidden_layers"])

        _thread.start_new_thread(
            t, (8, 8, 0.001, 1000, uploadFile, current_identity['identity'], hidden_layers))

        return{
            "status": 200
        }
    return{
        "status": 500
    }


@app.route('/file')
@jwt_required()
def getFileList():
    print(current_identity['identity'])
    response = userRepo.find_one(
        {'_id': ObjectId(current_identity['identity'])})
    print(response)
    return {
        "data": response['s3']
    }



if __name__ == '__main__':
    app.run()
