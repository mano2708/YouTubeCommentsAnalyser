from googleapiclient.discovery import build
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

API_KEY = 'AIzaSyDuz1Y0z7Z6FC8G2wQ_au3oZLpoDnT95aU' 

class SupportVectorMachine():
	def __init__(self,video_id):
		self.video_id = video_id
		self.video_name = ""
		self.channel_name = ""
		self.thumbnail = ""
		self.comments = []
		self.relevant_comments = []
		self.labels = []
		self.data = pd.DataFrame()
		self.model = None
		self.score = 0
		self.classification_report = []

	def collectData(self):
		# Initialize YouTube API
		youtube = build('youtube', 'v3', developerKey=API_KEY)

		# Get channel ID of the video uploader
		video_response = youtube.videos().list(part='snippet', id = self.video_id).execute()
		
		# Splitting the response for ChannelID
		video_snippet = video_response['items'][0]['snippet']
		uploader_channel_id = video_snippet['channelId']

		# Fetch comments
		print("Fetching Comments...")
		nextPageToken = None

		while len(self.comments) < 600:
			request = youtube.commentThreads().list(
			part='snippet',
			videoId=self.video_id,
			maxResults=100,  # Fetch up to 100 comments per request
			pageToken=nextPageToken
			)
			response = request.execute()
			for item in response['items']:
				comment = item['snippet']['topLevelComment']['snippet']
				# Check if the comment is not from the video uploader
				if comment['authorChannelId']['value'] != uploader_channel_id:
					self.comments.append(comment['textDisplay'])
			
			nextPageToken = response.get('nextPageToken')
			if not nextPageToken:
				break

	def sentimentLabel(self,comment):
		sentiment_analyzer = SentimentIntensityAnalyzer()
		sentiment_score = sentiment_analyzer.polarity_scores(comment)['compound']
		if sentiment_score > 0.05:
			return 'positive'
		elif sentiment_score < -0.05:
			return 'negative'
		else:
			return 'neutral'

	def processData(self):
		# Filter comments based on content
		print("Filtering comments...")
		hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

		threshold_ratio = 0.65

		relevant_comments = []

		# Inside your loop that processes comments
		for comment_text in self.comments:
			comment_text = comment_text.lower().strip()
			emojis = emoji.emoji_count(comment_text)
			text_characters = len(re.sub(r'\s', '', comment_text))
    		
			if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
				if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
					self.relevant_comments.append(comment_text)


		self.labels = [self.sentimentLabel(comment) for comment in self.relevant_comments]

	def createModel(self):
		# Create DataFrame for modeling
		self.data = pd.DataFrame({'comment': self.relevant_comments, 'label': self.labels})

		# Split data into training and test sets
		X_train, X_test, y_train, y_test = train_test_split(self.data['comment'], self.data['label'], test_size=0.2, random_state=42)

		# Create and train the SVM model
		self.model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
		self.model.fit(X_train, y_train)
		self.modelEvaluation(X_test,y_test)

	def modelEvaluation(self,X_test,y_test):

		# Predict and evaluate the model
		y_pred = self.model.predict(X_test)
		print("Classification Report:\n", classification_report(y_test, y_pred))
		print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
		self.score = round(self.model.score(X_test, y_pred) ,4)* 100

		#Example prediction
		predictions = self.model.predict(self.relevant_comments[9:13])
		for comment, prediction in zip(self.relevant_comments[9:13], predictions):
			print(f"Comment: {comment}\nPredicted Sentiment: {prediction}\n")

	def visualRepresentation(self):
		# Visualization (Optional)
		sentiment_counts = self.data['label'].value_counts()
		plt.figure(figsize=(8, 6))# Plot data
		ax = plt.axes()
		print(sentiment_counts.info())
		ax.plot(sentiment_counts,sentiment_counts.keys(), linewidth=2, linestyle='solid')
		ax.fill(sentiment_counts,sentiment_counts.keys(), 'b', alpha=0.4)
		plt.title('Sentiment Distribution of Comments')
		plt.savefig('/home/mano2708u/Intern/CommentsAnalyzer/src/templates/static/images/2.png')