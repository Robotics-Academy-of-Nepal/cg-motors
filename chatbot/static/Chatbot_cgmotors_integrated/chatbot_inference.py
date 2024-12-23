import random
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import spacy
import re
from pathlib import Path
from metaphone import doublemetaphone
from rapidfuzz import fuzz, process  
from chatbot.static.Chatbot_cgmotors_integrated.speech_synthesis import synthesize_speech

# Base configuration 
BASE_DIR = Path(__file__).resolve().parent

# Load English and Nepali spaCy models
nlp_en = spacy.load('en_core_web_sm')
nlp_nep = spacy.load('xx_ent_wiki_sm')

# Load intents and models
with open(BASE_DIR / 'cg_intents.json', 'r') as file:
    intents_en = json.load(file)
with open(BASE_DIR / 'cg_nep_intents.json', 'r', encoding='utf-8') as file:
    intents_nep = json.load(file)

# Load pre-trained models and word/class lists
words_en = pickle.load(open(BASE_DIR / 'words.pkl', 'rb'))
classes_en = pickle.load(open(BASE_DIR / 'classes.pkl', 'rb'))
words_nep = pickle.load(open(BASE_DIR / 'words_nep.pkl', 'rb'))
classes_nep = pickle.load(open(BASE_DIR / 'classes_nep.pkl', 'rb'))

# Chatbot Model definition
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Load pre-trained models
model_en = ChatbotModel(input_size=len(words_en), hidden_size=512, output_size=len(classes_en))
model_en.load_state_dict(torch.load(BASE_DIR / 'best_chatbot.pth', map_location=torch.device('cpu')))
model_en.eval()

model_nep = ChatbotModel(input_size=len(words_nep), hidden_size=512, output_size=len(classes_nep))
model_nep.load_state_dict(torch.load(BASE_DIR / 'best_chatbot_nep.pth', map_location=torch.device('cpu')))
model_nep.eval()

# Keyword lists
cg_keywords_en = ["Hi", "Hello", "Hey", "Greetings", "bye", "goodbye", "Farewell", "Thanks", "CG", "Motors", "Neta", "Netav", "V", "V50", "50", "NetaX", "X", "NetaS", "S", "GAC", "Aion", "AionY", "Y","KYC", "Changan", "Kuayue", "Chana", "V5D", "V5", "D", "King", "Long", "Kinglong", "Kingwin", "Win", "EV", "contact"]
cg_keywords_nep = ["नमस्ते", "नमस्कार", "विदा", "धन्यवाद", "सिजी", "मोटर्स", "नेटा", "वी५०", "वी", '५०', "नेटाएक्स", "एक्स", "नेटाएस", "एस","यस","जीएसी", "एसी", "ग्याक", "एयोन", "एयोनवाई","वाई", "केवाईसी", "चाङ्गन", "कुआयुए","कुवायु", "चाना", "भी५डी","भी", "५", "डी", "किङ", "लङ", "किङलङ", "किङवीन", "वीन", "इभी", "सम्पर्क"]



class PhoneticMatcher:
    def calculate_similarity(self, word1, word2):
        """
        Calculate similarity using multiple metrics with caching.
        """
        
        # Normalize inputs
        word1 = word1.lower().strip()
        word2 = word2.lower().strip()
        
        # Generate phonetic keys
        meta1 = doublemetaphone(word1)
        meta2 = doublemetaphone(word2)
        
        # Calculate multiple similarity metrics
        # Phonetic similarity
        phonetic_sim = (
            1.0 if meta1[0] == meta2[0] else
            0.5 if meta1[1] and meta2[1] and meta1[1] == meta2[1] else
            0.0
        )
        
        # Fuzzy similarity
        ratio_sim = fuzz.ratio(word1, word2) / 100
        # partial_sim = fuzz.partial_ratio(word1, word2) / 100
        token_sort_sim = fuzz.token_sort_ratio(word1, word2) / 100

        # Apply partial similarity only if both words have more than 2 letters
        if len(word1) > 2 and len(word2) > 2:
            partial_sim = fuzz.partial_ratio(word1, word2) / 100
        else:
            partial_sim = 0.0  # No partial similarity if one or both words are less than or equal to 2 letters
    
        
        # Weighted combination
        similarity = (
            0.2 * phonetic_sim +
            0.4 * ratio_sim +
            0.3 * partial_sim +
            0.1 * token_sort_sim
        )
        
        return similarity
    
    def find_best_match(self, input_word, keyword_list, threshold=0.72):
        """
        Find best matching keyword with enhanced matching logic.
        """
        best_match = None
        best_score = 0
        
        for keyword in keyword_list:
            similarity = self.calculate_similarity(input_word, keyword)
            if similarity > best_score:
                best_score = similarity
                best_match = keyword
        
        #debug print statement 
        print(f"[DEBUG] Input Word: '{input_word}', Best Match: '{best_match}', Similarity: {best_score}")
    
        
        return best_match if best_score >= threshold else None


def phonetic_keyword_replacement(user_input, lang='en'):
    keywords = cg_keywords_en if lang == 'en' else cg_keywords_nep
    nlp = nlp_en if lang == 'en' else nlp_nep
    doc = nlp(user_input)
    modified_tokens = []
    
    # Create an instance of PhoneticMatcher
    phonetic_matcher = PhoneticMatcher()

    for token in doc:
        best_match = phonetic_matcher.find_best_match(token.text, keywords)  # Call on the instance
        modified_tokens.append(best_match if best_match else token.text)

    return ' '.join(modified_tokens)



# Intent prediction and response generation
def clean_up_sentence(sentence, lang='en'):
    nlp = nlp_en if lang == 'en' else nlp_nep
    doc = nlp(sentence)
    return [token.lemma_.lower() if lang == 'en' else token.text for token in doc if not token.is_punct and not token.is_space]

def bag_of_words(sentence, words, lang='en'):
    sentence_words = clean_up_sentence(sentence, lang)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)


def predict_class(sentence, model, words, classes, lang='en', threshold=0.00):
    """
    Predict the intent of a given input sentence using the trained model.

    Args:
        sentence (str): Input sentence.
        model (torch.nn.Module): Trained model.
        words (list): Vocabulary list.
        classes (list): List of intent classes.
        lang (str): Language of the input (default is 'en').
        threshold (float): Confidence threshold for intent prediction.

    Returns:
        str: Predicted intent or "unknown" if confidence is below the threshold.
    """
    bow = bag_of_words(sentence, words, lang)
    bow_tensor = torch.from_numpy(bow).float().unsqueeze(0)
    outputs = model(bow_tensor)

    # Compute probabilities using softmax
    probabilities = torch.softmax(outputs, dim=1).detach().numpy()[0]
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]

    print(f"[DEBUG] Predicted Index: {predicted_index}, Confidence: {confidence}")

    # Return "unknown" if confidence is below the threshold
    if confidence < threshold:
        print("[DEBUG] Prediction below confidence threshold, returning 'unknown'.")
        return "unknown"

    # Otherwise, return the predicted class
    predicted_class = classes[predicted_index]
    print(f"[DEBUG] Predicted Class: {predicted_class}")
    return predicted_class


def get_response(predicted_intent, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == predicted_intent:
            return random.choice(intent['responses'])




def detect_language(input_text):
    nepali_regex = re.compile(r'[\u0900-\u097F]')
    english_regex = re.compile(r'[a-zA-Z]')
    if nepali_regex.search(input_text):
        return 'nep'
    elif english_regex.search(input_text):
        return 'en'
    return 'unknown'



def clean_sentence(sentence, lang='en'):
    """
    Cleans a sentence by:
    - Removing punctuation, spaces, and stopwords
    - Lemmatizing for English
    - Removing suffixes for Nepali words
    """

    # Custom Nepali stopwords and suffixes
    nepali_stopwords = {"मा", "ले", "गरेको", "के", "यो", "छ", "छन्", "दिनुहोस्", "कति", "गर्न", "कस्तो",
                        'को', 'का', 'की', 'कै', 'र', 'हो', 'छ', 'छन्', 'थियो', 'थिए', 
                        'गर्', 'गर्न', 'हुन', 'हुने', 'भएको', 'भएका', 'भएकी', 
                        'लाई', 'सँग', 'बाट', "बारे"}
    nepali_suffixes = ["मा", "को", "ले", "गर्न", "का", "हरु", "लागि", "दिनुहोस्", "छ", 'हरू', 'लाई','बाट','सँग','कति']

    nlp = nlp_en if lang == 'en' else nlp_nep
    doc = nlp(sentence)
    
    def remove_suffixes(word, suffixes):
        """Remove suffixes from a word."""
        for suffix in suffixes:
            if word.endswith(suffix):
                return word[: -len(suffix)]  # Remove the suffix
        return word

    cleaned_tokens = []
    for token in doc:
        # Remove punctuation and spaces
        if token.is_punct or token.is_space:
            continue

        # Stopword removal
        if lang == 'en' and token.is_stop:
            continue
        if lang == 'nep' and token.text in nepali_stopwords:
            continue

        # Process the token
        if lang == 'en':
            # Lemmatize for English
            cleaned_tokens.append(token.lemma_.lower())
        else:
            # Remove suffixes for Nepali
            cleaned_word = remove_suffixes(token.text, nepali_suffixes)
            cleaned_tokens.append(cleaned_word)

    # Join tokens back into a sentence
    return " ".join(cleaned_tokens)




def is_query_relevant(user_input, lang):
    """
    Check if the user input contains any relevant keywords for CG Motors.
    """
    cg_keywords = cg_keywords_en if lang == 'en' else cg_keywords_nep
    words = user_input.split()  # Split the input into words for comparison

    for word in words:
        # Check if any word directly matches a keyword
        if lang == 'en' and word.lower() in map(str.lower, cg_keywords):
            print(f"[DEBUG] Relevant Keyword Found: '{word}'")  # Debugging
            return True
        elif lang == 'nep' and word in cg_keywords:
            print(f"[DEBUG] Relevant Keyword Found: '{word}'")  # Debugging
            return True

    print("[DEBUG] No Relevant Keywords Found")  # Debugging
    return False






def get_chatbot_response(user_input, user_id="default_user"):
    language = detect_language(user_input)
    cleaned_input = clean_sentence(user_input, lang=language)
    modified_input = phonetic_keyword_replacement(cleaned_input, language)
    print(f"[DEBUG] Modified Input: {modified_input}")

    if not is_query_relevant(modified_input, language):
        if language == 'en':
            response = "Sorry, the question you asked is not related to CG Motors. Please ask about CG Motors, its vehicles, or services."
        elif language == 'nep':
            response = "माफ गर्नुहोस्, तपाईंले सोध्नु भएको प्रश्न सिजी मोटर्ससँग सम्बन्धित छैन। कृपया सिजी मोटर्स, यसको सवारी साधन, वा सेवाहरूको बारेमा सोध्नुहोस्।"
        else:
            response = "Sorry, I couldn't identify the language of your input."

        # Generate speech for the fallback response
        audio_path = synthesize_speech(response)
        return {"response": response, "intent": "unknown", "audio_path": audio_path}

    if language == 'en':
        intent = predict_class(modified_input, model_en, words_en, classes_en, lang='en')
        response = get_response(intent, intents_en)
    elif language == 'nep':
        intent = predict_class(modified_input, model_nep, words_nep, classes_nep, lang='nep')
        response = get_response(intent, intents_nep)
    else:
        response = "Sorry, I couldn't identify the language of your input."
        intent = "unknown"

    # Generate speech for the response
    audio_path = synthesize_speech(response)
    return {"response": response, "intent": intent, "audio_path": audio_path}




