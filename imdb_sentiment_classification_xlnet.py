import time

import matplotlib.pyplot as plt
import pandas as pd
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from util import denoise_text, remove_stopwords, summarize

start_time = time.perf_counter()

# Read the news dataset from Kaggle
news_df = pd.read_csv('data/IMDB Dataset Demo.csv')

# Step 2: Define classes
classes = ['positive', 'negative']

# Step 3: Generate random sentences
positive_prompts = [
    "This movie is well written",
    "The movie is funny and great",
    "The film is visually pleasing",
    "The story in the movie is wonderfully polarizing with masterful production and technique",
    "The actors in the movie played their part well"
]

negative_prompts = [
    "The film was horrible. Did not finish it",
    "The worst film I have ever watched, it was a mistake",
    "Do not watch this movie. Very poor editing, terrible performance",
    "This film is boring. I wouldn't recommend this movie at all",
    "Sick of this actor. The actor brought nothing to the movie"
]

prompts = positive_prompts + negative_prompts

# Step 4: Get vector representation of random sentences
tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
model = AutoModel.from_pretrained('xlnet-base-cased')

prompt_vectors = []
for prompt in prompts:
    tmp = remove_stopwords(prompt)
    inputs = tokenizer(tmp, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    prompt_vectors.append(vector)


# Step 5: Get vector representation of news corpus
news_vectors = []
for i, row in news_df.iterrows():
    tmp = denoise_text(row['review'])
    tmp = summarize(tmp)
    inputs = tokenizer(tmp, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    news_vectors.append(vector)

# Step 6: Calculate cosine similarity
similarity_scores = cosine_similarity(news_vectors, prompt_vectors)

# Step 7: Assign classes
predictions = []
isCorrect = []
promptIds = []
prompt_count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
for i, row in news_df.iterrows():
    max_index = similarity_scores[i].argmax()
    promptIds.append(max_index)
    reduced_index = max_index // 5
    predicted_class = classes[reduced_index]
    predictions.append(predicted_class)

    if row['sentiment'] == predicted_class:
        isCorrect.append(1)
        prompt_count_dict[max_index] = prompt_count_dict[max_index] + 1
    else:
        isCorrect.append(0)

# Step 8: Evaluate
print("Correctly labeled prompt counts:")
for key, value in prompt_count_dict.items():
    print(prompts[key] + ": " + str(value))

# Calculate overall accuracy
accuracy = accuracy_score(news_df['sentiment'], predictions)
print("Accuracy:", accuracy)

# Calculate weighted recall, precision, and F-score
weighted_recall, weighted_precision, weighted_fscore, _ = precision_recall_fscore_support(news_df['sentiment'], predictions, average='weighted')
print("Weighted recall:", weighted_recall)
print("Weighted precision:", weighted_precision)
print("Weighted F-score:", weighted_fscore)

plot_confusion_matrix(news_df['sentiment'], predictions, normalize=True, title='Confusion Matrix IMDB (xlnet)')
plt.show()

# Write the predictions
news_df['predicted_class'] = predictions
news_df['Correct'] = isCorrect
news_df['prompt_id'] = promptIds
news_df.to_csv('output.csv', index=False)


end_time = time.perf_counter()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
