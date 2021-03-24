import base64
import json
import logging
import sys
import tempfile
from os import listdir, path

import azure.functions as func

from .predict import classify_accent


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # running on Azure Function
    # model_path = "/model/new_model.pt"
    # running locally
    model_path = "../new_model.pt"
    data_path = "../data/"
    file_path = path.join(data_path, "temp.mp3")

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        mp3_blob = req_body.get("blob")
        decode(file_path, mp3_blob)
        res = classify_accent(data_path, model_path)
        logging.info(f"Classification result: {res}")

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