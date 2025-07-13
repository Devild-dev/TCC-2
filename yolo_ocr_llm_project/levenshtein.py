from Levenshtein import distance as levenshtein_distance

def calculate_levenshtein_metrics(yolo_class_name, ocr_results):
    yolo_words = yolo_class_name.lower().split()
    similarities, distances = [], []

    for ref_word in yolo_words:
        best_similarity = 0
        lowest_distance = float("inf")

        for ocr_list in ocr_results:
            for ocr_word in ocr_list:
                dist = levenshtein_distance(ref_word, ocr_word)
                max_len = max(len(ref_word), len(ocr_word))
                similarity = 1 - (dist / max_len)
                best_similarity = max(best_similarity, similarity)
                lowest_distance = min(lowest_distance, dist)

        similarities.append(best_similarity)
        distances.append(lowest_distance)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    avg_distance = sum(distances) / len(distances) if distances else 0.0

    if avg_similarity >= 0.9:
        adjustment = 0.45
    elif avg_similarity >= 0.8:
        adjustment = 0.225
    elif avg_similarity >= 0.7:
        adjustment = 0.1125
    else:
        adjustment = 0.0

    return avg_similarity, avg_distance, adjustment
