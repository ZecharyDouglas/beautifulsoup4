import requests
from bs4 import BeautifulSoup
from transformers import pipeline
#import torch #pytorch

url = "https://m.facebook.com/legal/terms"
page = requests.get(url).text

soup = BeautifulSoup(page, 'html.parser')
article = soup.find_all('div') #Contains the terms of service

#print(soup.title.get_text())
fixed_text=""
#Removes all the tags
for x in article:
    fixed_text+=x.get_text()

#    print(x.get_text())

print(article)    


#Summarization nlp model that can handle a large quantity of text
hf_name = 'pszemraj/led-large-book-summary'

AnotherSummarizer = pipeline(
    "summarization",
    hf_name,
   # device=0 if torch.cuda.is_available() else -1,
)


result = AnotherSummarizer(
   fixed_text[:16384],
    min_length=16,
    max_length=256,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    early_stopping=True,
)

print(result)

#Token indices sequence length is longer than the specified maximum sequence length for this model (49267 > 
#16384). Running this sequence through the model will result in indexing errors
#Will need to make sure that the text being inputted fits the models restraints
