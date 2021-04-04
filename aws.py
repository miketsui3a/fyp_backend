import boto3
import pymongo
import datetime 
from bson.objectid import ObjectId

bucketName = "fyp-mike2021"

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucketName)

client = pymongo.MongoClient(
    "mongodb+srv://miketsui3a:aA26761683@cluster0.bnkhm.azure.mongodb.net")
db = client['fyp']
userRepo = db['user']


def upload(filename, user_id, loss):
    r = bucket.upload_file(Filename=filename, Key=filename, ExtraArgs={'ACL': 'public-read'})
    print(r)
    # presign_link = create_presign_link(bucketName, filename)
    presign_link = 'https://fyp-mike2021.s3-ap-southeast-1.amazonaws.com/'+filename

    userRepo.update_one({'_id':ObjectId(user_id)},{"$push":{"s3":{
        "presign_link": presign_link,
        "created_at": datetime.datetime.now(),
        "loss": loss
    }}})

def create_presign_link(bucketName, filename):
    s3Client = boto3.client('s3')
    response = s3Client.generate_presigned_url('get_object', Params={
        "Bucket": bucketName,
        "Key": filename,
    })

    return response


def test():

    print(userRepo.update_one({'username':'ac'},{"$push":{"www":{
        "presign_link": "presign_link",
        "created_at": datetime.datetime.now()
    }}}))
    print(userRepo.find_one({'username':'ac'}))


