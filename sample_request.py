import requests
import json
import os
import sys


REST = os.getenv("REST") or "localhost:5000"


def mkReq(reqmethod, endpoint, data):
    print(f"Response to http://{REST}/{endpoint} request is")
    jsonData = json.dumps(data)
    response = reqmethod(f"http://{REST}/{endpoint}", data=jsonData,
                         headers={'Content-type': 'application/json'})
    if response.status_code == 200:
        jsonResponse = json.dumps(response.json(), indent=4, sort_keys=True)
        print(jsonResponse)
        return
    else:
        print(
            f"response code is {response.status_code}, raw response is {response.text}")
        return response.text


mkReq(requests.post, "apiv1/preprocess",
      data={
          "model": "sentiment",
          "file": "house_price_1",
          "callback": {
              "url": "http://localhost:5000",
              "data": {"some": "arbitrary", "data": "to be returned"}
          }
      }
      )

# mkReq(requests.post, "apiv1/prediction/house_price_1",
#       data={
#           "model": "sentiment",
#           "file": "house_price_1",
#           "callback": {
#               "url": "http://localhost:5000",
#               "data": {"some": "arbitrary", "data": "to be returned"}
#           }
#       }
#       )
