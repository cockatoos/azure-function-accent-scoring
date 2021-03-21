import azure.functions as func
import base64
import logging
from os import listdir
import sys
import tempfile

from .predict import classify_accent

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # running on Azure Function
    model_path = '/model/new_model.pt'
    # running locally
    # mode_path = '../new_model.pt'

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        mp3_blob = req_body.get('blob')
        decode(mp3_blob)
        res = classify_accent(tempfile.gettempdir(), model_path)
        logging.info(f'accent classification result: {res}')

    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    return func.HttpResponse(json.dumps(res), headers = headers)

# TODO (jyz16): add an option to store mp3 file to File Share for debug purposes
def decode(blob_str):
    """Decode blob string into mp3 and store as temp file."""
    # temporary files generated and used by your functions during execution
    # have to be stored in the temp directory.
    tempFilePath = tempfile.gettempdir()
    fp = tempfile.NamedTemporaryFile()
    mp3_data = base64.b64decode(blob_str)
    logging.info('MP3 file is decoded, writing into temp file...')
    fp.write(mp3_data)