from flask import Flask, render_template, redirect, url_for, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SelectField
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect

from pre_process import TweetTextCleaner
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import json
import pandas as pd
import numpy as np
import plotly
import plotly.express as px

app = Flask(__name__)
app.config['SECRET_KEY'] = 'MAGE]5ZA9hZC_:w~olqC.4aOCt/c,0@.e3B&OlPy2Z]"z"4IBmnh]0a/EdFiYe|'
app.config['UPLOAD_FOLDER'] = 'datasets'

csrf = CSRFProtect()
csrf.init_app(app)

tweet_text_cleaner = None

# Form
class SentimentForm(FlaskForm):
    dataset_file_field = FileField('Dataset File', validators=[
        FileRequired(message='There is no file uploaded'),
        FileAllowed(['csv'], message='File has to be in CSV format')
    ])

    model_select_field = SelectField('Select sentiment analysis model', choices=[
        (0, 'Hybrid CNN-LSTM'),
        (1, 'Single CNN'),
        (2, 'LSTM')
    ], default=0)

# Utility f
def pre_process(df, file_name):
    tweet_text_cleaner = TweetTextCleaner()

    df['cleaned'] = df['text'].apply(lambda sentiment: tweet_text_cleaner.remove_html_tags(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_retweets(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_urls(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_mentions(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_hashtags(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_non_ascii(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.change_word_to_number(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_numbers(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.case_folding(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.expand_contractions(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.replace_negation(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_punctuations(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.remove_stopwords(sentiment))
    df['cleaned'] = df['cleaned'].apply(lambda sentiment: tweet_text_cleaner.lemmatize(sentiment))

    df.to_csv(os.path.join(app.root_path, 'datasets', file_name), index=False)

def set_sentiment_analysis_model(selected_model, df):
    if selected_model == '0':
        df['model'] = 'cnn-lstm'
    elif selected_model == '1':
        df['model'] = 'cnn'
    else:
        df['model'] = 'lstm'

    return df

def set_topic(selected_topic, df):
    if selected_topic == 'sinovac':
        df['topic'] = 'sinovac'
    elif selected_topic == 'astrazeneca':
        df['topic'] = 'astrazeneca'
    elif selected_topic == 'pfizer':
        df['topic'] = 'pfizer'
    else:
        df['topic'] = 'moderna'
    
    return df

def select_topic(df):
    if df['topic'].iloc[0] == 'sinovac':
        df['topic'] = 'sinovac'
    elif df['topic'].iloc[0] == 'astrazeneca':
        df['topic'] = 'astrazeneca'
    elif df['topic'].iloc[0] == 'pfizer':
        df['topic'] = 'pfizer'
    else:
        df['topic'] = 'moderna'
    
    return df

def dencode_label(sentiment):
    if sentiment == 0:
        return 'Negative'
    else:
        return 'Positive'

def transform_data(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['cleaned'].values)

    X_tokenized = tokenizer.texts_to_sequences(df['cleaned'].values)
    X_padded = pad_sequences(X_tokenized, maxlen=100)

    return X_padded

def select_model_predict(df):
    selected_model_text = ''
    model = None

    if df['model'].iloc[0] == 'cnn-lstm':
        selected_model_text = 'Hybrid CNN-LSTM'
        file_name = df['topic'].iloc[0] + '_cnn_lstm_model.h5'
        model = load_model(os.path.join(app.root_path, 'models', file_name))
        X_test = transform_data(df)
        y_pred_raw = model.predict(X_test)
        y_pred = np.argmax(y_pred_raw, axis=1)
        df['sentiment'] = y_pred
        df['sentiment'] = df['sentiment'].apply(lambda x: dencode_label(x))
        df = df[['text', 'sentiment']]
    elif df['model'].iloc[0] == 'cnn':
        selected_model_text = 'Single CNN Model'
        file_name = df['topic'].iloc[0] + '_cnn_model.h5'
        model = load_model(os.path.join(app.root_path, 'models', file_name))
        X_test = transform_data(df)
        y_pred_raw = model.predict(X_test)
        y_pred = np.argmax(y_pred_raw, axis=1)
        df['sentiment'] = y_pred
        df['sentiment'] = df['sentiment'].apply(lambda x: dencode_label(x))
        df = df[['text', 'sentiment']]
    else:
        selected_model_text = 'Single LSTM Model'
        file_name = df['topic'].iloc[0] + '_lstm_model.h5'
        model = load_model(os.path.join(app.root_path, 'models', file_name))
        X_test = transform_data(df)
        y_pred_raw = model.predict(X_test)
        y_pred = np.argmax(y_pred_raw, axis=1)
        df['sentiment'] = y_pred
        df['sentiment'] = df['sentiment'].apply(lambda x: dencode_label(x))
        df = df[['text', 'sentiment']]

    return df, selected_model_text

def plot_bar(df):
    fig = px.bar(df, x="sentiment", y="count", color="sentiment", text='count')
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def plot_pie(df):
    fig = px.pie(df, values='count', names='sentiment')
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

# Error views
@app.errorhandler(404)
def page_not_found(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_server(error):
    return render_template('errors/500.html'), 500

# Home view
@app.route('/', methods=['GET'])
def render_index_view():
    title = 'SentiVacc19 | Home'
    return render_template('index.html', title=title)

# About view
@app.route('/about', methods=['GET'])
def render_about_view():
    title = 'SentiVacc19 | About'
    return render_template('about.html', title=title)

# Sentiment analysis views
@app.route('/sentiment-analysis', methods=['GET'])
def render_sentiment_analysis_view():
    title = 'SentiVacc19 | Sentiment Analysis'
    return render_template('sentiment-analysis.html', title=title)

# Sinovac views
@app.route('/sentiment-analysis/sinovac', methods=['GET', 'POST'])
def render_sinovac_view():
    if 'sinovac_session' in session:
        return redirect(url_for('render_sinovac_result_view'))

    title = 'SentiVacc19 | Sentiment Analysis: Sinovac'
    form = SentimentForm()

    if form.validate_on_submit():
        f = form.dataset_file_field.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.root_path, 'datasets', filename))

        selected_model = form.model_select_field.data        
        df_sinovac = pd.read_csv(os.path.join(app.root_path, 'datasets', filename))
        df_sinovac = set_sentiment_analysis_model(selected_model, df_sinovac)
        df_sinovac = set_topic('sinovac', df_sinovac)
        pre_process(df_sinovac, 'sinovac_cleaned.csv')
        session['sinovac_session'] = 1

        return redirect(url_for('render_sinovac_result_view'))
            
    return render_template('pages/sinovac.html', title=title, form=form)

# AstraZeneca views
@app.route('/sentiment-analysis/astrazeneca', methods=['GET', 'POST'])
def render_astrazeneca_view():
    if 'astrazeneca_session' in session:
        return redirect(url_for('render_astrazeneca_result_view'))

    title = 'SentiVacc19 | Sentiment Analysis: AstraZeneca'
    form = SentimentForm()

    if form.validate_on_submit():
        f = form.dataset_file_field.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.root_path, 'datasets', filename))

        selected_model = form.model_select_field.data        
        df_astrazeneca = pd.read_csv(os.path.join(app.root_path, 'datasets', filename))
        df_astrazeneca = set_sentiment_analysis_model(selected_model, df_astrazeneca)
        df_astrazeneca = set_topic('astrazeneca', df_astrazeneca)
        pre_process(df_astrazeneca, 'astrazeneca_cleaned.csv')
        session['astrazeneca_session'] = 1

        return redirect(url_for('render_astrazeneca_result_view'))

    return render_template('pages/astrazeneca.html', title=title, form=form)

# Pfizer views
@app.route('/sentiment-analysis/pfizer', methods=['GET', 'POST'])
def render_pfizer_view():
    if 'pfizer_session' in session:
        return redirect(url_for('render_pfizer_result_view'))

    title = 'SentiVacc19 | Sentiment Analysis: Pfizer'
    form = SentimentForm()

    if form.validate_on_submit():
        f = form.dataset_file_field.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.root_path, 'datasets', filename))
    
        selected_model = form.model_select_field.data        
        df_pfizer = pd.read_csv(os.path.join(app.root_path, 'datasets', filename))
        df_pfizer = set_sentiment_analysis_model(selected_model, df_pfizer)
        df_pfizer = set_topic('pfizer', df_pfizer)
        pre_process(df_pfizer, 'pfizer_cleaned.csv')
        session['pfizer_session'] = 1

        return redirect(url_for('render_pfizer_result_view'))

    return render_template('pages/pfizer.html', title=title, form=form)

# Moderna views
@app.route('/sentiment-analysis/moderna', methods=['GET', 'POST'])
def render_moderna_view():
    if 'moderna_session' in session:
        return redirect(url_for('render_moderna_result_view'))

    title = 'SentiVacc19 | Sentiment Analysis: Moderna'
    form = SentimentForm()

    if form.validate_on_submit():
        f = form.dataset_file_field.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.root_path, 'datasets', filename))

        selected_model = form.model_select_field.data        
        df_moderna = pd.read_csv(os.path.join(app.root_path, 'datasets', filename))
        df_moderna = set_sentiment_analysis_model(selected_model, df_moderna)
        df_moderna = set_topic('moderna', df_moderna)
        pre_process(df_moderna, 'moderna_cleaned.csv')
        session['moderna_session'] = 1

        return redirect(url_for('render_moderna_result_view'))

    return render_template('pages/moderna.html', title=title, form=form)

# Result
@app.route('/sentiment-analysis/sinovac/result', methods=['GET'])
def render_sinovac_result_view():
    if 'sinovac_session' not in session:
        return redirect(url_for('render_sinovac_view'))

    title = 'SentiVacc19 | Sentiment Analysis: Sinovac Result'
    df_sinovac = pd.read_csv(os.path.join(app.root_path, 'datasets', 'sinovac_cleaned.csv'))

    df_sinovac, selected_model_text = select_model_predict(df_sinovac)
    sinovac_data = df_sinovac.copy()
    sinovac_count = sinovac_data['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
    sinovac_tweets = sinovac_data.shape[0]
    sinovac_positive = sinovac_count[sinovac_count['sentiment'] == 'Positive']['count'].values[0]
    sinovac_negative = sinovac_count[sinovac_count['sentiment'] == 'Negative']['count'].values[0]
    sinovac_data = sinovac_data.head(10)
    sinovac_bar_plot = plot_bar(sinovac_count)
    sinovac_pie_plot = plot_pie(sinovac_count)

    return render_template(
        'pages/sinovac-result.html',
        title=title,
        selected_model_text=selected_model_text,
        sinovac_tweets=sinovac_tweets,
        sinovac_positive=sinovac_positive,
        sinovac_negative=sinovac_negative,
        sinovac_data=sinovac_data,
        sinovac_bar_plot=sinovac_bar_plot,
        sinovac_pie_plot=sinovac_pie_plot
    )

@app.route('/sentiment-analysis/astrazeneca/result', methods=['GET'])
def render_astrazeneca_result_view():
    if 'astrazeneca_session' not in session:
        return redirect(url_for('render_astrazeneca_view'))

    title = 'SentiVacc19 | Sentiment Analysis: AstraZeneca Result'
    df_astrazeneca = pd.read_csv(os.path.join(app.root_path, 'datasets', 'astrazeneca_cleaned.csv'))

    df_astrazeneca, selected_model_text = select_model_predict(df_astrazeneca)
    astrazeneca_data = df_astrazeneca.copy()
    astrazeneca_count = astrazeneca_data['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
    astrazeneca_tweets = astrazeneca_data.shape[0]
    astrazeneca_positive = astrazeneca_count[astrazeneca_count['sentiment'] == 'Positive']['count'].values[0]
    astrazeneca_negative = astrazeneca_count[astrazeneca_count['sentiment'] == 'Negative']['count'].values[0]
    astrazeneca_data = astrazeneca_data.head(10)
    astrazeneca_bar_plot = plot_bar(astrazeneca_count)
    astrazeneca_pie_plot = plot_pie(astrazeneca_count)

    return render_template(
        'pages/astrazeneca-result.html',
        title=title,
        selected_model_text=selected_model_text,
        astrazeneca_tweets=astrazeneca_tweets,
        astrazeneca_positive=astrazeneca_positive,
        astrazeneca_negative=astrazeneca_negative,
        astrazeneca_data=astrazeneca_data,
        astrazeneca_bar_plot=astrazeneca_bar_plot,
        astrazeneca_pie_plot=astrazeneca_pie_plot
    )

@app.route('/sentiment-analysis/pfizer/result', methods=['GET'])
def render_pfizer_result_view():
    if 'pfizer_session' not in session:
        return redirect(url_for('render_pfizer_view'))

    title = 'SentiVacc19 | Sentiment Analysis: Pfizer Result'
    df_pfizer = pd.read_csv(os.path.join(app.root_path, 'datasets', 'pfizer_cleaned.csv'))

    df_pfizer, selected_model_text = select_model_predict(df_pfizer)
    pfizer_data = df_pfizer.copy()
    pfizer_count = pfizer_data['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
    pfizer_tweets = pfizer_data.shape[0]
    pfizer_positive = pfizer_count[pfizer_count['sentiment'] == 'Positive']['count'].values[0]
    pfizer_negative = pfizer_count[pfizer_count['sentiment'] == 'Negative']['count'].values[0]
    pfizer_data = pfizer_data.head(10)
    pfizer_bar_plot = plot_bar(pfizer_count)
    pfizer_pie_plot = plot_pie(pfizer_count)

    return render_template(
        'pages/pfizer-result.html',
        title=title,
        selected_model_text=selected_model_text,
        pfizer_tweets=pfizer_tweets,
        pfizer_positive=pfizer_positive,
        pfizer_negative=pfizer_negative,
        pfizer_data=pfizer_data,
        pfizer_bar_plot=pfizer_bar_plot,
        pfizer_pie_plot=pfizer_pie_plot
    )

@app.route('/sentiment-analysis/moderna/result/', methods=['GET'])
def render_moderna_result_view():
    if 'moderna_session' not in session:
        return redirect(url_for('render_moderna_view'))

    title = 'SentiVacc19 | Sentiment Analysis: Moderna Result'
    df_moderna = pd.read_csv(os.path.join(app.root_path, 'datasets', 'moderna_cleaned.csv'))

    df_moderna, selected_model_text = select_model_predict(df_moderna)
    moderna_data = df_moderna.copy()
    moderna_count = moderna_data['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
    moderna_tweets = moderna_data.shape[0]
    moderna_positive = moderna_count[moderna_count['sentiment'] == 'Positive']['count'].values[0]
    moderna_negative = moderna_count[moderna_count['sentiment'] == 'Negative']['count'].values[0]
    moderna_data = moderna_data.head(10)
    moderna_bar_plot = plot_bar(moderna_count)
    moderna_pie_plot = plot_pie(moderna_count)

    return render_template(
        'pages/moderna-result.html',
        title=title,
        selected_model_text=selected_model_text,
        moderna_tweets=moderna_tweets,
        moderna_positive=moderna_positive,
        moderna_negative=moderna_negative,
        moderna_data=moderna_data,
        moderna_bar_plot=moderna_bar_plot,
        moderna_pie_plot=moderna_pie_plot
    )

# Back to home and reset
@app.route('/sentiment-analysis/reset', methods=['GET'])
def remove_all_files():
    datasets_path = os.path.join(app.root_path, 'datasets')
    
    for file in os.scandir(datasets_path):
        os.remove(file.path)

    session.pop('sinovac_session', None)
    session.pop('astrazeneca_session', None)
    session.pop('pfizer_session', None)
    session.pop('moderna_session', None)

    return redirect(url_for('render_index_view'))

if __name__ == '__main__':
    app.run()
