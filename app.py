import models
import tools
import tensorflow as tf
import gc
import numpy
from pluginFactory import PluginFactory
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

def searchAnswer(sentence, subject, typeS):
    plugin = PluginFactory.getPlugin(subject, typeS)
    return plugin.response(sentence)
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)
    # subjects, types, stopwords, dictionnary = tools.defaultValues()
    # while True:
    #     print("Tape your sentence:")
    #     test= input()
    #     rSubject, rType, rValue= analyse(test)
    #     result = searchAnswer(test, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
    #     print(result)
    
