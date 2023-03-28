from socket import SocketIO
from flask_cors import CORS
from yaml import emit
import models
import tools
import tensorflow as tf
import gc
import numpy
from pluginFactory import PluginFactory
from flask import Flask, jsonify, render_template, Response, request
import cv2
import subprocess

app = Flask(__name__)
# CORS(app)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, async_handlers=True)
# socketio.run(app)

def analyse(sentence):
    subjects, types, stopwords, dictionnary = tools.defaultValues()
    modelSubjects= models.getModelSubjects(dictionnary, subjects)
    modelSubjects.load("data/modelSubjects.tflearn")
    resultS= modelSubjects.predict([tools.bagOfWords(sentence, dictionnary, stopwords)])
    tf.keras.backend.clear_session()
    del modelSubjects
    gc.collect()

    modelTypes= models.getModelTypes(dictionnary, types)
    modelTypes.load("data/modelTypes.tflearn")
    resultT= modelTypes.predict([tools.bagOfWords(sentence, dictionnary, stopwords)])
    tf.keras.backend.clear_session()
    del modelTypes
    gc.collect()

    modelValues= models.getModelValues(dictionnary)
    modelValues.load("data/modelValues.tflearn")
    resultV= modelValues.predict([tools.bagOfWords(sentence, dictionnary, stopwords)])
    tf.keras.backend.clear_session()
    del modelValues
    gc.collect()
    return resultS[0], resultT[0], resultV[0][0]

def searchAnswer(sentence, subject, typeS):
    plugin = PluginFactory.getPlugin(subject, typeS)
    return plugin.response(sentence)

# messages = []

# def searchAnswer(sentence, subject, typeS):
#     global messages
#     plugin = PluginFactory.getPlugin(subject, typeS)
#     response, terminal_messages = plugin.response(sentence)
#     messages += terminal_messages
#     return response, messages

# @app.route('/')
# def index():
#     return render_template('index.html')

# @socketio.on('message')
# def handle_message(message):
#     print(message)
#     response = 'Bonjour, vous avez envoyé le message suivant: ' + message['data']
#     emit('response', response)

# @app.route('/', methods=['POST', 'GET'])
# def index():
#     global messages
#     subjects, types, stopwords, dictionnary = tools.defaultValues()
#     if request.method == 'POST':
#         sentence = request.form['message']
#         rSubject, rType, rValue = analyse(sentence)
#         result, messages = searchAnswer(sentence, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
#         return render_template('index.html', message=sentence, response=result, messages=messages)
#     else:
#         # On la première requête GET, messages est vide donc on l'initialise avec un message d'accueil
#         # messages = ["Bienvenue sur notre chatbot ! Posez-nous une question et nous vous aiderons à trouver la réponse."]
#         return render_template('index.html', messages=messages)

@app.route('/', methods=['POST', 'GET'])
def index():
    subjects, types, stopwords, dictionnary = tools.defaultValues()
    if request.method == 'POST':
        sentence = request.form['message']
        rSubject, rType, rValue = analyse(sentence)
        result = searchAnswer(sentence, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
        return render_template('index.html', message=sentence, response=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0', port=5050)
    app.run(debug=True, host="0.0.0.0", port=5050)
    # while True:
    #     print("Tape your sentence:")
    #     test= input()
    #     rSubject, rType, rValue= analyse(test)
    #     result = searchAnswer(test, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
    #     print(result)