import base64
import logging
from os import listdir
import sys
import tempfile

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        blob = req_body.get('blob')
        decode(blob)

    if blob:
        return func.HttpResponse(f"This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "Fail?",
             status_code=505
        )

def decode(blob_str):
    # temporary files generated and used by your functions during execution
    # have to be stored in the temp directory.
    tempFilePath = tempfile.gettempdir()
    fp = tempfile.NamedTemporaryFile()
    mp3_data = base64.b64decode(blob_str)
    print('Mp3 file is decoded, writing into temp file...')
    fp.write(mp3_data)