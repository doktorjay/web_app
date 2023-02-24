from flask import Flask
from recommender import recommend_random, recommend_with_NMF, recommend_neighborhood
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Hello, World!')
    
@app.route('/recommend')
def recommender():
    # html_form_data = dict(request.args)
    # # a python dictionary consisting of
    # # "name"-value pairs from the HTML form!

    # recs = recommend_random()
    # # at this point, we would then pass this
    # #information as an argument into our recommender function.

    # print(html_form_data)

    # return render_template('recommendations.html',
    #                         movies = recs)

    if request.args['algo']=='Random':
        recs = recommend_random()
        print(request.args)
        
        titles = request.args.getlist('title')
        ratings = request.args.getlist('Ratings')
        user_input = dict(zip(titles,ratings))

        print(user_input)
        
        for keys in user_input:
            user_input[keys] = int(user_input[keys])
        return render_template('recommendations.html', recs = recs)
    else:
        return f"Function not defined"

    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
