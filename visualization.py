import matplotlib.pyplot as plt
from collections import Counter

def visualize_dataset(dataset, classes, file_name):
    # Count the occurrences of each label
    label_counts = Counter(dataset['labels'])

    # Prepare data for plotting
    labels = list(classes[label] for label in label_counts.keys())
    counts = list(label_counts.values())

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Distribution of Labels in the Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./outputs/{file_name}')
    plt.close()