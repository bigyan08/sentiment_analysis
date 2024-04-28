from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tweet ="@Bigyan I am happy today https://bigyanaryal08.com.np"

tweet_words=[]

for word in tweet.split(" "):
    if word.startswith("@") and len(word) >1:
        word="@user"
    elif word.startswith("https"):
        word="https"
    tweet_words.append(word)

tweet_proc= " ".join(tweet_words)

#loading the model
roberta= "cardiffnlp/twitter-roberta-base-sentiment"

model= AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer= AutoTokenizer.from_pretrained(roberta)

labels=['Negative','Neutral','Positive']

encoded_tweet=tokenizer(tweet_proc,return_tensors="pt")
#output=model(encoded_tweet['input_ids'],encoded_tweet['attention_mask']) or,
output=model(**encoded_tweet)

scores=output[0][0].detach().numpy()
scores=softmax(scores)

for i in range(len(scores)):
    l=labels[i]
    s=scores[i]
    print(l,s)