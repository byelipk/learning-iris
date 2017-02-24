def check_distribution(data, category):
    return data[category].value_counts() / len(data)
