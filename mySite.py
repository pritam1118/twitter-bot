# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request,session,Response
from werkzeug import secure_filename
import os
from supportFile import *
import pickle
from sms import sendSMS
import sqlite3
from datetime import datetime

app = Flask(__name__)

app.secret_key = '1234'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET', 'POST'])
def landing():
	return render_template('home.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
	return render_template('home.html')


@app.route('/input', methods=['GET', 'POST'])
def input():
	if request.method == 'POST':
		if request.form['sub']=='Upload':
			savepath = r'upload/'
			dataset = request.files['dataset']
			dataset.save(os.path.join(savepath,(secure_filename('dataset.csv'))))
			return render_template('input.html',mgs="Dataset Uploaded..!!!")

	return render_template('input.html')		

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
	df = pd.read_csv('upload/dataset.csv')
	return render_template('dataset.html', tables=[df.to_html(classes='w3-table-all w3-hoverable')], titles=df.columns.values)

@app.route('/clean', methods=['GET', 'POST'])
def clean():
	df = pd.read_csv('upload/dataset.csv')
	df.drop(['id_str', 'screen_name', 
                    'location', 'description', 
                    'url', 'created_at', 
                    'lang', 'status',
                    'default_profile',
                    'default_profile_image',
                    'has_extended_profile','name'],axis=1,inplace=True)

	return render_template('clean.html', tables=[df.to_html(classes='w3-table-all w3-hoverable')], titles=df.columns.values)

@app.route('/best', methods=['GET', 'POST'])
def best():
	return render_template('best.html',algo=best_algo)

@app.route('/result', methods=['GET', 'POST'])
def result():
	df = pd.read_csv('Result.csv')
	df = df.replace(0,'Non Bot')
	df = df.replace(1,'Bot')
	return render_template('result.html', tables=[df.to_html(classes='w3-table-all w3-hoverable')], titles=df.columns.values)
@app.route('/user', methods=['GET', 'POST'])
def user():
	if request.method == 'POST':
		if request.form['sub']=='Predict':
			name = request.form['name']
			followers_count = int(request.form['followers_count'])
			friends_count = int(request.form['friends_count'])
			listedcount = int(request.form['listedcount'])
			favourites_count = int(request.form['favourites_count'])
			verified = int(request.form['verified'])
			statuses_count = int(request.form['statuses_count'])
			test_sample = [[followers_count,friends_count,listedcount,favourites_count,verified,statuses_count]]
			clf = pickle.load(open('dtc_model.sav', 'rb'))
			result = clf.predict(test_sample)
			result = 'Bot' if result==1 else 'Non Bot'
			#sendSMS(sender, recipient, name + " is "+result+" Account")
			now = datetime.now()
			dt_string = now.strftime("%d/%m/%Y %H:%M:%S")			
			con = sqlite3.connect('mydatabase.db')
			cursorObj = con.cursor()
			cursorObj.execute("CREATE TABLE IF NOT EXISTS Users (Date text,Name text,Result text)")
			cursorObj.execute("INSERT INTO Users VALUES(?,?,?)",(dt_string,name,result))
			con.commit()
			return render_template('user.html',result=result,name=name,followers_count=followers_count,friends_count=friends_count,listedcount=listedcount,favourites_count=favourites_count,verified=verified,statuses_count=statuses_count)
	return render_template('user.html')

@app.route('/history', methods=['GET', 'POST'])
def history():
	global name
	conn = sqlite3.connect('mydatabase.db', isolation_level=None,
						detect_types=sqlite3.PARSE_COLNAMES)
	db_df = pd.read_sql_query(f"SELECT * from Users;", conn)
	
	return render_template('history.html',tables=[db_df.to_html(classes='w3-table-all w3-hoverable w3-padding')], titles=db_df.columns.values)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
	# response.cache_control.no_store = True
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)
