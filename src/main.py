from flask import *
from naive_bayes_model import NaiveBayes
from svm_model import SupportVectorMachine
from flask import render_template
from dt_model import DecisionTree


app = Flask(__name__)
app.static_folder='templates/static'



@app.route('/')

def home():
    return  render_template('index.html')


@app.route('/results',methods=['POST','GET'])

def results():
    if request.method == 'POST':
        result = request.form['video_id']
        NB = NaiveBayes( str(result)[-11:] )
        NB.collectData()
        NB.processData()
        NB.createModel()
        NB.visualRepresentation()
        print(NB.sentiment_counts)

        SVM = SupportVectorMachine( str(result)[-11:] )
        SVM.collectData()
        SVM.processData()
        SVM.createModel()
        SVM.visualRepresentation()
        
        DT = DecisionTree(str(result)[-11:])
        DT.collectData()
        DT.processData()
        DT.createModel()
        DT.visualRepresentation()
        

        return render_template('results.html',
            video_id = NB.video_name ,
            thumbnail = NB.thumbnail , 
            channel_id = NB.channel_name , 
            NBscore = NB.score,
            SVMscore = SVM.score,
            DTscore = DT.score,
            rating = ratingCalculator(NB.sentiment_counts["positive"],NB.sentiment_counts["negative"],NB.sentiment_counts["neutral"])
        ) 



def ratingCalculator(pos , neg , neu):
    agg = pos + neg*(-1)
    total = pos + neg + neu
    normalizedValue = (agg + total) / (2*total)
    rating = (normalizedValue * 4) + 1
    return round(rating,1)

if __name__ == '__main__':
    app.run(debug=True,port=1234)