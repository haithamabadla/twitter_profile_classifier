from flask import Flask, jsonify, request
import spacy
import joblib
import re

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')
#nlp = joblib.load('nlp_pipeline.sav')
ot_classifier = joblib.load('ot_classifier.sav')  # Classifier model
transformer = joblib.load('tfidf_transformer.sav')  # TF-IDF model


def predict_tweet(tweet):
    x = re.sub(r'http\S+', '', tweet)  # remove URLs
    x = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ",x).split())  # remove special characters and extra spaces

    tweet = nlp(x)  # add the text to spacy pipeline
    # clean text by removing stopwords, punctuation, digits, lemmatize the tokens and turn them into lowercase.
    tweet = ' '.join([token.lemma_.lower() for token in tweet if not token.is_stop and not token.is_punct and not token.text.isdigit() and len(token.text) > 2])

    # Predictions
    # pass the clean text to the TF-IDF to transform the text and then use the classifier to predict
    result = ot_classifier.predict(transformer.transform([tweet]))
    # covert results into readable classes
    result = 'Donald Trump' if result == 0 else 'Barak Obama'

    # return result
    return result


@app.route('/parameters', methods=['GET'])
def parameters():
    tweet = request.args.get('tweet')
    #result = predict_tweet('FBI Director Christopher Wray just admitted that the FISA Warrants and Survailence of my campaign were illegal. So was the Fake Dossier. THEREFORE, THE WHOLE SCAM INVESTIGATION, THE MUELLER REPORT AND EVERYTHING ELSE FOR THREE YEARS, WAS A FIXED HOAX. WHO PAYS THE PRICE?....')
    result = predict_tweet(tweet)
    return jsonify(prediction=result)


if __name__ == '__name__':
    app.run(threaded=True)
