import random
import nltk
from nltk.corpus import wordnet
import pandas as pd

nltk.download('punkt_tab')

# 동의어 치환 함수
def synonym_replacement(words, n=1):
    new_words = words.copy()
    word_count = 0
    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words[i] = synonym
            word_count += 1
        if word_count >= n:
            break
    return new_words

# 랜덤 삽입 함수
def random_insertion(words, n=1):
    for _ in range(n):
        new_word = random.choice(words)
        synonyms = wordnet.synsets(new_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name().replace('_', ' ')
            random_index = random.randint(0, len(words)-1)
            words.insert(random_index, synonym)
    return words

# 문장 내 단어 위치 바꾸기 함수
def random_swap(words, n=1):
    length = len(words)
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return words

# 문장 내 단어 삭제 함수
def random_deletion(words, p=0.5):
    if len(words) == 0:
        return words
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return new_words if len(new_words) > 0 else [random.choice(words)]

# EDA 증강 기법을 위한 함수 (각 문장에 대해 증강 처리)
def eda_single_row(row):
    context_words = nltk.word_tokenize(row['context'])
    question_words = nltk.word_tokenize(row['question'])

    context_aug = random.choice([synonym_replacement(context_words),
                                 random_insertion(context_words),
                                 random_swap(context_words),
                                 random_deletion(context_words)])

    question_aug = random.choice([synonym_replacement(question_words),
                                  random_insertion(question_words),
                                  random_swap(question_words),
                                  random_deletion(question_words)])

    return {
        'title': row['title'],
        'context': ' '.join(context_aug),
        'question': ' '.join(question_aug),
        'id': row['id'],
        'answers': row['answers'],
        'document_id': row['document_id'],
        '__index_level_0__': row['__index_level_0__']
    }

# 여러 행에 대해 증강 처리
def eda(df, n_augmented=1):
    augmented_rows = []
    for _, row in df.iterrows():
        augmented_rows.append(eda_single_row(row))

    augmented_df = pd.DataFrame(augmented_rows)
    concat_df = pd.concat([df, augmented_df], ignore_index=True)
    return concat_df