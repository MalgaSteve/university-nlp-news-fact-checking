# News Fact Checking

Deadline 17/12

## How to run

clone repo

'''
git clone https://github.com/MalgaSteve/university-nlp-news-fact-checking.git
'''

### API Keys

You need an API from NewsAPI and Google Fact Check API.

Add .env file in the working directory with both API Keys.

'''
touch .env
echo "NEWS_API_KEY=Your Key" >> .env
echo "FACT_API_KEY=Your Key" >> .env
'''

### Run main.py

'''
python main.py
'''
