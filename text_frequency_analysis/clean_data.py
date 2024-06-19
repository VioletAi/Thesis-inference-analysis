import json
from collections import Counter
import re

# stopwords from mltk but removed ones that could potentially denote relative possition
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'into',
    'through', 'during', 'before', 'after', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "to"]

with open('preds_epoch-1_step0_scanrefer.json', 'r') as file:
    data = json.load(file)

# filter samples where pred does not match ref_captions
mismatch_samples = [sample for sample in data if sample['pred']!= sample['ref_captions'][0]]

# extract and process the content within the first set of quotation marks from prompts
def extract_quoted_text(prompt):
    # find text within the first set of quotation marks
    match = re.search(r'"(.*?)"', prompt)
    return match.group(1) if match else ''

prompts = [extract_quoted_text(sample['prompt']) for sample in mismatch_samples]

# function to clean and split text into words
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return [word for word in words if word not in stop_words]

# count common words in the extracted parts of prompts
word_count = Counter(word for prompt in prompts for word in clean_text(prompt))

# display the most common words
common_words = word_count.most_common(20)  # adjust number as needed

print("Most common words in the extracted parts of prompts where 'pred' does not match 'ref_captions':")
for word, freq in common_words:
    print(f"{word}: {freq}")
