# Step 1: Get news corpus
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

# Read the news dataset from Kaggle
news_df = pd.read_csv('data/AG News Demo.csv')

# Step 2: Define classes
classes = ['world', 'sports', 'business', 'Science']

# Step 3: Generate random sentences
world_prompts = [
    "This news is about political and social events around the world",
    "Different people outside the US",
    "About leaders of countries",
    "Geo-politics among countries and continents",
    "Relationship between nations"
]

sport_prompts = [
    "Talks about sport competition, winners and contenders ",
    "Game play, victory and records of contenders in sport team competition",
    "About sport teams and managers",
    "Talks about different sport competitions",
    "Athletes transfer from one team to another"
]

business_prompts = [
    "This news is covers finance",
    "This is about the market",
    "Covers manufacturing and products",
    "Customer and business relationship",
    "Different stories of businesses buying and selling"
]

science_prompts = [
    "This news is about researches",
    "Various stories about tech",
    "This covers science, technology and innovation",
    "Talks about animals, plants as well as technology and innovation in science",
    "This is regarding new findings"
]



prompts = world_prompts + sport_prompts + business_prompts + science_prompts

# Step 4: Get vector representation of random sentences
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

prompt_vectors = []
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    prompt_vectors.append(vector)


# Step 5: Get vector representation of news corpus
news_vectors = []
for i, row in news_df.iterrows():
    inputs = tokenizer(row['Description'], return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    news_vectors.append(vector)

# Step 6: Calculate cosine similarity
similarity_scores = cosine_similarity(news_vectors, prompt_vectors)

# Step 7: Assign classes
predictions = []
isCorrect = []
promptIds = []
prompt_count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0}
for i, row in news_df.iterrows():
    max_index = similarity_scores[i].argmax()
    promptIds.append(max_index)
    reduced_index = max_index // 5
    predicted_class = classes[reduced_index]
    predictions.append(reduced_index + 1)

    if row['Class Index'] - 1 == reduced_index:
        isCorrect.append(1)
        prompt_count_dict[max_index] = prompt_count_dict[max_index] + 1
    else:
        isCorrect.append(0)

# Step 8: Evaluate
print("Correctly labeled prompt counts:")
for key, value in prompt_count_dict.items():
    print(prompts[key] + ": " + str(value))

# Calculate overall accuracy
accuracy = accuracy_score(news_df['Class Index'], predictions)
print("Accuracy:", accuracy)

# Calculate weighted recall, precision, and F-score
weighted_recall, weighted_precision, weighted_fscore, _ = precision_recall_fscore_support(news_df['Class Index'], predictions, average='weighted')
print("Weighted recall:", weighted_recall)
print("Weighted precision:", weighted_precision)
print("Weighted F-score:", weighted_fscore)

plot_confusion_matrix(news_df['Class Index'], predictions, normalize=True, title='Confusion Matrix new classification (bert)')
plt.show()

# Write the predictions
news_df['predicted_class'] = predictions
news_df['Correct'] = isCorrect
news_df['prompt_id'] = promptIds
news_df.to_csv('output.csv', index=False)


end_time = time.perf_counter()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
