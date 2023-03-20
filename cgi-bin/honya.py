#!/usr/local/anaconda3/envs/detectron2/bin/python
# -*- coding: UTF-8 -*-

#import subprocess
import cgi, os, sys
import random
import string
import json
import base64
from dotenv import load_dotenv

load_dotenv()
upload_dir = "/tmp"


def save_upload_file (item):
    filename = ramdom_chr(16)+".jpg"
#    print("filename=", item.filename)
    try:
        path = os.path.join(upload_dir, filename)
        chunk = item.file.read()
        if chunk:
            with open(path, mode="wb") as f:
                f.write(chunk)

    except Exception as ee:
        return "", ee

    return path, "";


def ramdom_chr (num):
    random_list = [random.choice(string.ascii_letters + string.digits) for n in range(num)]
    return "".join(random_list)

####################
#
########

if os.getenv('HTTP_X_API_KEY'):
    key = os.environ['HTTP_X_API_KEY']

if not 'key' in locals():
    print("Status: 403 Forbidden\n\n")
    exit()

if key != os.environ['API_KEY']:
    print("Status: 403 Forbidden\n\n")
    exit()


form = cgi.FieldStorage()
if "zumen" in form:
    fileitem = form["zumen"]
    upload_filename, ee = save_upload_file(fileitem)
    if ee:
        print (ee)
        exit()

else:
    print ("Status: 400 Bad Requesti\n\n")
    exit()


os.environ['MPLCONFIGDIR'] = "/tmp/matplotlib"

sys.path.append('../detectron2')

import predict
json_filename, result_jpg_filename, result_other_filename = predict.main(upload_filename)

with open(json_filename, "rb") as f:
    file_data = base64.b64encode(f.read())

with open(result_jpg_filename, "rb") as f:
    predict_data = base64.b64encode(f.read())

with open(result_other_filename, "rb") as f:
    other_data = base64.b64encode(f.read())

res = {'result': '00000', 'file_data': file_data.decode(), 'predict_data': predict_data.decode(), 'other_data': other_data.decode()}

print("Content-Type: application/json\n\n")
print (json.dumps(res))

os.remove(json_filename)
os.remove(result_jpg_filename)
os.remove(upload_filename)

exit()