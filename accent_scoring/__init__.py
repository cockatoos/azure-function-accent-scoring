import base64
import json
import logging
import sys
import tempfile
from os import listdir, path, remove
import glob

import azure.functions as func

from .predict import classify_accent


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # running on Azure Function
    # model_path = "/model/new_model.pt"
    # running locally
    model_path = "./eval_model.onnx"
    data_path = "./data/"
    file_path = path.join(data_path, "temp.mp3")

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        mp3_blob = req_body.get("blob")

        # clear data path
        files = glob.glob('./data/*')
        for f in files:
            remove(f)
        files = glob.glob('./data/.*')
        for f in files:
            remove(f)

        try:
            decode(file_path, mp3_blob)
        except Exception as e:
            logging.info(f"decode failed with exception: {e}")
            res = {"status":"failure", "reason":e}
        
        try:
            res = classify_accent(data_path, model_path, save_onnx=False)
            logging.info(f"Classification result: {res}")
        except Exception as e:
            logging.info(f"classification failed with exception: {e}")
            res = {"status": "failure", "reason": e}

    headers = {"Content-type": "application/json", "Access-Control-Allow-Origin": "*"}

    return func.HttpResponse(json.dumps(res), headers=headers)
    # return func.HttpResponse("success", headers=headers)


# TODO (jyz16): add an option to store mp3 file to File Share for debug purposes
def decode(file_path, blob_str):
    """Decode blob string into mp3 and store as temp file."""
    # temporary files generated and used by your functions during execution
    # have to be stored in the temp directory.
    # tempFilePath = tempfile.gettempdir()
    # fp = tempfile.NamedTemporaryFile()
    mp3_data = base64.b64decode(blob_str)
    logging.info("MP3 file is decoded, writing into temp file...")
    # TODO (jyz16): send HTTPResponse to notify front-end of progress
    with open(file_path, "wb+") as mp3:
        mp3.write(mp3_data)
    logging.info("File write successful.")