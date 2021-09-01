from wsgiref import simple_server
from flask import Flask, request
from flask import Response
import os
from flask_cors import CORS,cross_origin
import json
from flask import Flask, render_template, request,jsonify
import pymongo
from scrapper_utils import getdata, getreviews, nextpage

from prediction.predictApp import PredictApi
from training.trainApp import TrainApi
from utils.utils import createDirectoryForUser, extractDataFromTrainingIntoDictionary, get_scrap_train_data

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
# app.config['DEBUG'] = True

trainingDataFolderPath = "trainingData/"


class ClientApi:
    def __init__(self):
        stopWordsFilePath = "data/stopwords.txt"
        self.predictObj = PredictApi(stopWordsFilePath)
        self.trainObj = TrainApi(stopWordsFilePath)

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        searchString = request.form['content'].replace(" ","") # obtaining the search string entered in the form
        try:
            dbConn = pymongo.MongoClient("mongodb://localhost:27017/")  # opening a connection to Mongo
            db = dbConn['crawlerDB'] # connecting to the database called crawlerDB
            reviews = db[searchString].find({}) # searching the collection with the name same as the keyword
            if reviews.count() > 0: # if there is a collection with searched keyword and it has records in it
                return render_template('results.html',reviews=reviews) # show the results to user
            else:
                flipkart_url = "https://www.flipkart.com/search?q=" + searchString # preparing the URL to search the product on flipkart
                flipkart_html = getdata(flipkart_url)
                x = flipkart_html.find_all("a", {"class":"_1fQZEK"})[0]["href"]

                productLink = "https://www.flipkart.com" + x
                product_html = getdata(productLink)
                box = product_html.find_all("div", {"class":"col JOpGWq"})[0].find_all('a')[-1].get('href')

                reviewLink = "https://www.flipkart.com"+box
                review_html = getdata(reviewLink)

                url = reviewLink
                reviews = []
                i = 1
                while i<=10:
                    soup = getdata(url+str("&page=")+str(i))
                    reviews = getreviews(soup, searchString, reviews)
                    url = nextpage(soup)
                    i = i + 1
                else:
                    print("end of line")

            get_scrap_train_data(searchString)
            return render_template('results.html', reviews=reviews)
        except:
            return "something is wrong"
    else:
        return render_template('index.html')



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if request.json['text'] is not None and request.json['userId'] is not None and request.json['projectId'] is not None:
            text = request.json['text']
            userId = str(request.json['userId'])
            projectId = str(request.json['projectId'])
            #csvFilePath = trainingData + userId + "/" + projectId + "/trainingData.csv"
            jsonFilePath = trainingDataFolderPath + userId + "/" + projectId + "/trainingData.json"
            modelPath = trainingDataFolderPath + userId + "/" + projectId + "/modelForPrediction.sav"
            vectorPath = trainingDataFolderPath + userId + "/" + projectId + "/vectorizer.pickle"
            result = clntApp.predictObj.executePreocessing(text, jsonFilePath,modelPath,vectorPath)


    except ValueError:
        return Response("Value not found inside  json trainingData")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        return Response((str(e)))
    return Response(result)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainModel():

    try:
        if request.get_json() is not None:
            data = request.json['data']
        if request.json['userId'] is not None:
            userId = str(request.json['userId'])
            # path = trainingData+userId
        if request.json['projectId'] is not None:
            projectId = str(request.json['projectId'])


            createDirectoryForUser(userId, projectId)

        path = trainingDataFolderPath + userId + "/" + projectId

        trainingDataDict = extractDataFromTrainingIntoDictionary(data)

        with open(path + '/trainingData.json', 'w', encoding='utf-8') as f:
            json.dump(trainingDataDict, f, ensure_ascii=False, indent=4)
        #dataFrame = pd.read_json(path + '/trainingData.json')
        jsonpath = path + '/trainingData.json'
        modelPath = path
        modelscore = clntApp.trainObj.training_model(jsonpath,modelPath)
        #dataFrame.to_csv(path + '/trainingData.csv', index=None, header=True)
    except ValueError as val:
        return Response("Value not found inside  json trainingData", val)
    except KeyError as keyval:
        return Response("Key value error incorrect key passed", keyval)
    except Exception as e:
        return Response((str(e)))

    return Response("Success")



if __name__ == "__main__":
    clntApp = ClientApi()
    app.run(host="192.168.29.188",port=8080, debug=False)

