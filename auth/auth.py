from flask import Blueprint
from flask.globals import request
from .UserModel import User
import pymongo
import json
from passlib.hash import argon2


client = pymongo.MongoClient(
    "mongodb+srv://miketsui3a:aA26761683@cluster0.bnkhm.azure.mongodb.net")
db = client['fyp']


auth_api = Blueprint('auth_api', __name__)

@auth_api.route("/register", methods=['POST'])
def register():
    data = json.loads(request.data)
    if db['user'].count_documents({'username': data['username']}) != 0:
        return {
            "status": 400,
            "message": "user already registered"
        }
    result = db['user'].insert_one(User(data['username'], argon2.hash(data['password'])).__dict__)
    return {
        "id": str(result.inserted_id)
    }


def authenticate(username, password):
    try:
        user = db['user'].find_one({'username':username})
        print(user)
        if user is None:
            return False
        if argon2.verify(password, user['password']):
            res = type('obj', (object,), {'id' : str(user['_id'])})
            return res
    except:
        return {
            "message": "wrong username or password"
        }


def identity(payload):
    return payload
