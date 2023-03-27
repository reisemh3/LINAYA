import models
import tools
import tensorflow as tf
import gc
import numpy
from pluginFactory import PluginFactory
from flask import Flask, jsonify, render_template, Response, request
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


def searchAnswer(sentence, subject, typeS):
    plugin = PluginFactory.getPlugin(subject, typeS)
    return plugin.response(sentence)


@app.route('/')
def index():
    subjects, types, stopwords, dictionnary = tools.defaultValues()
    return render_template('index.html')

# @app.route('/terminal', methods=['POST'])
# def terminal():
#     subjects, types, stopwords, dictionnary = tools.defaultValues()
#     input = request.form['input']
#     rSubject, rType, rValue = analyse(input)
#     result = searchAnswer(input, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
#     return jsonify(result)

# @app.route('/')
# def index():
#     subjects, types, stopwords, dictionnary = tools.defaultValues()
#     input = request.form['input']
#     rSubject, rType, rValue = analyse(input)
#     result = searchAnswer(input, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
#     # return jsonify(result)
#     return render_template('index.html')

# @app.route('/terminal', methods=['POST'])
# def terminal():
# 	input = request.form['input']
# 	rSubject, rType, rValue = analyse(input)
# 	result = searchAnswer(input, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
# 	return jsonify(result)

if __name__ == '__main__':
    subjects, types, stopwords, dictionnary = tools.defaultValues()
    app.run(debug=True, host="0.0.0.0", port=5050)
    while True:
        print("Tape your sentence:")
        test= input()
        rSubject, rType, rValue= analyse(test)
        result = searchAnswer(test, subjects[numpy.argmax(rSubject)], types[numpy.argmax(rType)])
        print(result)
    
        

    
