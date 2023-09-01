# UNL Project Submission 
# By Prajna R (PES1UG21CS417), Shravani Hiremath (PES1UG21CS563), Pratyusha Satish Rao (PES2UG21CS394), Dhatri (PES2UG21CS910)

# Importing all the necessary libraries
from deep_translator import GoogleTranslator
from googletrans import Translator
from spacy.pipeline import EntityRuler
from collections import defaultdict

import re
import os
import pydot
import spacy
import difflib
import string

# Load the English model from spaCy
nlp = spacy.load("en_core_web_sm")

# Identifying all the different types of Karakas, Kriyas and Visheshana-Visheshya pairs
def identify_karakas(sentence):
    doc = nlp(sentence)
    kartru = []
    karma = []
    kriya = []
    sampradana = []
    apadana = []
    karana = []
    adhikarana = []
    visheshana = []  # List to store Visheshana (adjectives)

    for token in doc:
        # Identify Kartru (Subject)
        if token.dep_ in ["nsubj", "nsubjpass"]:
            kartru.append(token.text)
        # Identify Karma (Object)
        elif token.dep_ == "dobj":
            karma.append(token.text)
        # Identify Kriya (Verb)
        elif token.pos_ == "VERB":
            kriya.append(token.text)
        # Identify Apadana (Source)
        elif token.text.lower() == "from" and token.i < len(doc) - 1:
            next_token = doc[token.i + 1]
            if next_token.pos_ == "NOUN" or next_token.pos_ == "PROPN":
                apadana.append(next_token.text)
        # Identify Karana (Instrument)
        elif token.dep_ == "pobj" and token.head.dep_ == "prep":
            if token.head.text.lower() in ["with", "using"]:
                karana.append(token.text)
        # Identify Sampradana (Goal)
        elif token.pos_ == "ADP" and token.text.lower() in ["with", "to"]:
            # Look for the object governed by the preposition
            obj = token.head
            if obj is not None and obj.dep_ in ["dobj", "xcomp"]:
                sampradana.append(obj.text)
        # Identify Adhikarana (Locus)
        elif token.pos_ == "ADP":
            # Look for noun phrases connected to the adposition
            noun_phrase = " ".join([child.text for child in token.subtree if child.pos_ == "NOUN"])
            if noun_phrase:
                adhikarana.append(noun_phrase)
        # Identify Visheshana (Adjectives)
        elif token.pos_ == "ADJ":
            if token.head.pos_ == "NOUN":
                visheshana.append((token.text, token.head.text))  # Store the adjective and the noun it modifies

    return kartru, karma, kriya, sampradana, apadana, karana, adhikarana, visheshana

# Function to find Karma (object) and Sampradana for the verb
def find_karma_sampradana_for_verb(verb_token):
    karma = set()  # Use a set to store Karma entities to prevent duplicates
    sampradana = set()  # Use a set to store Sampradana entities to prevent duplicates
    for child in verb_token.children:
        if child.dep_ in ["dobj", "attr", "acomp"]:
            # Check if the token is an adjective and its head is a noun
            if child.pos_ == "ADJ" and child.head.pos_ == "NOUN":
                karma_text = child.head.text + " " + child.text
                karma.add(karma_text)
            else:
                karma_text = child.text
                karma.add(karma_text)
        elif child.dep_ in ["prep"]:
            # Check for Sampradana (object preceded by "to")
            if child.text.lower() == "to":
                for grandchild in child.children:
                    sampradana_text = grandchild.text
                    sampradana.add(sampradana_text)
    return list(karma), list(sampradana)  # Convert sets back to lists for consistent processing

# Function to find Apadana for the verb
def find_apadana_for_verb(verb_token):
    apadana = []
    for child in verb_token.children:
        if child.dep_ in ["prep"] and child.text.lower() == "from":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    apadana_text = grandchild.text
                    apadana.append(apadana_text)
    return apadana

# Function to find Adhikarana for the verb
def find_adhikarana_for_verb(verb_token):
    adhikarana = []
    for child in verb_token.children:
        if child.dep_ in ["prep"] and child.head.pos_ == "VERB" and child.text.lower() != "with":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    adhikarana_text = grandchild.text
                    adhikarana.append(adhikarana_text)
    return adhikarana

# Function to find Karana for the verb
def find_karana_for_verb(verb_token):
    karana = []
    for child in verb_token.children:
        if child.dep_ in ["prep"] and child.head.pos_ == "VERB" and child.text.lower() == "with":
            for grandchild in child.children:
                if grandchild.pos_ in ["NOUN", "PROPN"]:
                    karana_text = grandchild.text
                    karana.append(karana_text)
    return karana

# Function to create the mind map based on the Karaka framework for a given sentence
def create_mind_map_for_sentence(sentence, graph, verb_karma_dict):
    doc = nlp(sentence)
    kartru, karma, kriya, sampradana, apadana, karana, adhikarana, visheshana = identify_karakas(sentence)

    for token in doc:
        if token.pos_ in ["VERB", "AUX"]:
            verb = token.text

            # Find Kartru (subject)
            kartru = [child.text for child in token.children if child.dep_ in ["nsubj", "csubj"]]
            compound_kartru = [child.text for child in token.children if child.dep_ in ["compound"] and child.head.dep_ in ["nsubj", "csubj"]]
            kartru.extend(compound_kartru)

            # Find Karma (object) & Sampradana (goal)
            karma, sampradana = find_karma_sampradana_for_verb(token)

            # Find Adhikarana (location and time)
            adhikarana = find_adhikarana_for_verb(token)
            
             # Find Apadana (source)
            apadana = find_apadana_for_verb(token)
            
            # Find Karana
            karana = find_karana_for_verb(token)

            # Add nodes and edges for the verb and its associated entities
            verb_node = pydot.Node(name=verb, shape="ellipse", style="filled", fillcolor="#a3d9ff")
            graph.add_node(verb_node)

            for entity in [kartru, karma, adhikarana, karana]:
                for item in entity:
                    entity_node = pydot.Node(name=item, shape="box", style="filled", fillcolor="#ffff99")
                    graph.add_node(entity_node)

                    if entity == kartru:
                        edge = pydot.Edge(entity_node, verb_node, label="Kartru", color="#009688")
                    elif entity == karma:
                        edge = pydot.Edge(verb_node, entity_node, label="Karma", color="#2196F3")
                    elif entity == sampradana:
                        edge = pydot.Edge(verb_node, entity_node, label="Sampradana", color="#FFC107")
                    elif entity == apadana:
                        edge = pydot.Edge(verb_node, entity_node, label="Apadana", color="#FF0000")    
                    elif entity == adhikarana:
                        edge = pydot.Edge(entity_node, verb_node, label="Adhikarana", color="#9C27B0")
                    elif entity == karana:
                        edge = pydot.Edge(entity_node, verb_node, label="Karana", color="#FF5722")
                    graph.add_edge(edge)

    # Find Visheshana (Adjectives) and connect them to the nouns they modify
    for adj, noun in visheshana:
        # Connect the adjective node to the noun node
        adj_node = pydot.Node(name=adj, shape="box", style="filled", fillcolor="#ffcccb")
        noun_node = pydot.Node(name=noun, shape="box", style="filled", fillcolor="#ffff99")
        graph.add_node(adj_node)
        graph.add_node(noun_node)
        edge = pydot.Edge(adj_node, noun_node, label="Visheshana", color="#FF1493")
        graph.add_edge(edge)
           
# Function to create the mind map for a paragraph
def create_mind_map_for_paragraph(paragraph, output_file):
    # Create the graph for the entire paragraph
    graph = pydot.Dot(graph_type="graph", rankdir="LR")  # Horizontal layout for better readability

    # Split the paragraph into individual sentences
    sentences = [sent.text.strip() for sent in nlp(paragraph).sents]

    for sentence in sentences:
        create_mind_map_for_sentence(sentence, graph, {})

    # Save the mind map for the entire paragraph to a file
    graph.write_png(output_file)

english_to_sanskrit = {
    "Once": "एकदा",
    "there":"तत्र",
    "f":"|",
    "was": "आसीत्",
    "a": "एका",
    "young": "किशोरी",
    "woman": "स्त्री",
    "named": "नाम",
    "Vidya": "विद्या",
    "who": "या",
    "lived": "आसीत्",
    "in": "",
    "village": "ग्रामे",
    "She": "सा",
    "wanted": "इच्छती",
  
    "to travel": "प्रयाणम्",
    "and": "च",
    "explore": "अन्वेषितुम्",
    "the": "तत्",
    "world": "जगत्",
    "but": "तथापि",
    "she": "सा",
    "couldn": "न शक्नोति स्म",
    "t": "",
    "couldn't": "नाशक्नोति",
    "speak": "वक्तुम्",
    "English": "अङ्ग्लभाषा",
    "So": "ततः",
    "joined": "सम्प्रविश्य",
    "an": "एकं",
    "course": "अभ्यासक्रम:",
    "her": "तस्याः",
    "Her": "तस्याः",
    "Vidya": "विद्या",
    "learned": "अधीतवती",
    "grammar": "व्याकरणम्",
    "vocabulary": "शब्दकोश",
    "and": "च",
    "how": "कथम्",
    "to": "इत्यस्मै",
    "pronounce": "उच्चारणम्",
    "words": "शब्द:",
    "She": "सा",
    "practised": "अभ्यासत",
    "talking": "वक्तुम्",
    "with": "सह",
    "She": "सा",
    "classmates": "सहपाठिनः",
    "and": "च",
    "understood": "अवगच्छत",
    "simple": "सरला",
    "conversations": "संभाषणानि",
    "Vidya": "विद्या",
    "felt": "भवति",
    "more": "अधिकम्",
    "confident": "आत्मविश्वाससम्पन्ना",
    "and": "च",
    "decided": "निर्णीत",
    "travel": "यात्राम्",
    "alone": "एका",
    "an": "एकं",
    "English speaking": "अङ्ग्लभाषाभाषिणीभूतां",
    "speaking" : "भाषिणीभूतां",
    "country": "देश:",
    "She": "सा",
    "visited": "आगच्छत",
    "big": "महत्तरां",
    "cities": "नगराणि",
    "talked": "वक्तुम्",
    "to": "सह",
    "local": "स्थानिक",
    "people": "जनाः",
    "and": "च",
    "learned": "अधीतवती",
    "about": "विचार्य",
    "different": "भिन्नः",
    "cultures": "संस्कृतिः",
    "Vidya": "विद्या",
    "realized": "अवगच्छत",
    "that": "या",
    "learning": "अध्ययनानि",
    "English": "अङ्ग्लभाषा",
    "opened": "उद्घाटितानि",
    "doors": "द्वाराणि",
    "new": "नवानि",
    "experiences": "अनुभवानि",
    "and": "च",
    "helped": "सहायितवती",
    "she": "सा",
    "connect": "सम्बद्धवती",
    "with": "सह",
    "people": "जनाः",
    "all": "सर्वत्र",
    "over": "अधिकम्",
    "the": "तत्",
    "world": "जगत्",
    "she": "सा",
    "journey": "प्रयाणः",
    "showed": "दर्शितवती",
    "that": "या",
    "with": "सह",
    "determination": "निश्चयात्मिका",
    "and": "च",
    "learning": "अध्ययनात्",
    "anything": "किमपि",
    "is": "भवति",
    "possible": "सम्भवम्",
    "She": "सा",
    "came": "आगत्य",
    "from": "ततः",
    "village.": "ग्रामे",
    "but": "तथापि",
    "conquered": "विजयीभूता",
    "world": "जगत्",
    "with": "सह",
    "She": "सा",
    "grit": "साहसेन",
    ".": "|"
}

english_to_sanskrit_with_vibhakti = {
    "Once": "एकदा",
    "there": "तत्र",
    "was": "आसीत्",
    "a": "एका",
    "young": "किशोरी",
    "woman": "युवती",
    "named": "नामिका",
    "Vidya": "विद्या",
    "who": "या",
    "lived": "निवसति स्म",
    "in": "",
    "in village": "ग्रामे",
    "village": "ग्रामे",
    "She": "सा",
    "wanted": "इच्छति स्म",
    "to": "तुम्",
    "travel": "प्रयाणं",
    "and": "च",
    "explore": "अन्वेषितुम्",
    "the": "तत्",
    "world": "जगत्",
    "but": "परन्तु",
    "she": "सा",
    "couldn": "न शक्नोति स्म",
    "t": "",
    "couldn't": "न शक्नोति स्म",
    "speak": "वक्तुम्",
    "English": "आङ्ग्लभाषां",
    "So": "अतः",
    "joined": "सम्प्रविश्य",
    "an": "एकं",
    "course": "अभ्यासक्रमम्",
    "Her": "तस्याः",
    "learned": "अधीतवती",
    "grammar": "व्याकरणम्",
    "vocabulary": "शब्दावली",
    "how": "कथम्",
    "pronounce": "उच्चारणं",
    "words": "शब्दानां",
    "practised": "शिक्षितवती",
    "talking": "वक्तुम्",
    "with": "सह",
    "her": "तस्याः",
    "classmates": "सहपाठिभिः",
    "and": "च",
    "understood": "अवगच्छत",
    "simple": "सरलं",
    "conversations": "संभाषणानि",
    "felt": "अनुभवति स्म",
    "more": "अधिकम्",
    "confident": "आत्मविश्वासं",
    "decided": "निर्णीता",
    "alone": "एकाम्",
    "English speaking": "आङ्ग्लभाषाभाषिणः",
    "speaking" : "भाषिणीभूतां",
    "country": "देशे",
    "visited": "आगच्छत्",
    "big": "महत्तरां",
    "cities": "नगराणि",
    "talked": "सम्भाषितवती",
    "to": "सह",
    "local": "स्थानिकैः",
    "people": "जनैः",
    "learned": "अधीतवती",
    "about": "विषये",
    "different": "भिन्नाः",
    "cultures": "संस्कृतीनां",
    "realized": "अवगच्छत",
    "that": "यत्",
    "learning": "आङ्ग्लभाषाशिक्षणेन",
    "opened": "उद्घाटितानि",
    "doors": "द्वाराणि",
    "to": "तुम्",
    "new": "नवानि",
    "experiences": "अनुभवानि",
    "her":"तस्याः",
    "helped": "सहाययति",
    "connect": "सम्पर्कं",
    "all": "सर्वत्र",
    "over": "अधिकं",
    "journey": "यात्रायां",
    "showed": "दर्शयति",
    "with": "सह",
    "determination": "निश्चयेन",
    "anything": "किमपि",
    "is": "भवति",
    "possible": "सम्भवम्",
    "came": "आगत्य",
    "from": "ततः",
    "but": "तथापि",
    "conquered": "जितवती",
    "grit": "साहसेन",
    ".": "|"
}

# Karakas lists
kartru_karakas = ['who', 'She', 'she', 'Vidya', 'She', 'Vidya', 'She', 'Vidya', 'English', 'her', 'Her', 'journey', 'anything', 'She']
karma_karakas = ['world', 'English', 'course', 'grammar', 'words', 'conversations', 'cities', 'doors', 'world']
kriya_karakas = ['was', 'named', 'lived', 'wanted', 'travel', 'explore', 'speak', 'joined', 'learned', 'pronounce', 'practised', 'talking', 'understood', 'felt', 'decided', 'travel', 'visited', 'talked', 'learned', 'realized', 'learning', 'opened', 'helped', 'connect', 'showed', 'came', 'conquered']
sampradana_karakas = ['talking','travel']
apadana_karakas = ['village', 'country',  'cultures', 'experiences', 'people', 'world', 'village']
karana_karakas=['classmates', 'people','determination','grit']
adhikarana_karaka=['village', 'village','world']

# Adding space to punctuation so that it is considered as a separate token
def replace_punctuation_with_space(paragraph):
    # Replace each period with " . " (space before and after the period)
    paragraph = paragraph.replace(".", " . ")
    # Replace each comma with " , " (space before and after the comma)
    return paragraph.replace(",", " , ") 

# Translating individual words to Sanskrit
def translate_word_to_sanskrit(word):
    # Use Google Translator for individual word translation
    translation = GoogleTranslator(source='en', target='sa').translate(word)
    return translation

# Identifying Sanskrit numbers
def is_sanskrit_number(word):
    # Function to check if the word is a Sanskrit number
    sanskrit_numbers = set(["१", "२", "३", "४", "५", "६", "७", "८", "९", "०"])
    return all(char in sanskrit_numbers for char in word)

# Function for appending the appropriate Pratyayas based on Vibhakti   
def get_sanskrit_karakas(word, karma, gender, case):
    # Dictionary to map English karakas to Sanskrit karakas with their cases
    karaka_mapping = {
        'subject': 'प्रथमा',
        'object': 'द्वितीया',
        'indirect object': 'तृतीया',
        'source': 'चतुर्थी',
        'destination': 'पञ्चमी',
        'instrument': 'षष्ठी',
        'possessive': 'सप्तमी',
        'relationship': 'सम्बन्ध',
        'agent': 'कर्ता',
    }

    # Word forms mapping based on the vibhakti (case), gender, and number
    word_forms_mapping = {
        'प्रथमा': {
            'singular_masculine': 'ः', 'dual_masculine': 'औ', 'plural_masculine': 'आः',
            'singular_feminine': 'आ', 'dual_feminine': 'ए', 'plural_feminine': 'आः',
            'singular_neuter': 'म्', 'dual_neuter': 'ए', 'plural_neuter': 'आनि',
        },
        'द्वितीया': {
            'singular_masculine': 'अम्', 'dual_masculine': 'औ', 'plural_masculine': 'आन्',
            'singular_feminine': 'आम्', 'dual_feminine': 'ए', 'plural_feminine': 'आः',
            'singular_neuter': 'म्', 'dual_neuter': 'ए', 'plural_neuter': 'आनि',
        },
        'तृतीया': {
            'singular_masculine': 'एन', 'dual_masculine': 'आभ्याम्', 'plural_masculine': 'ऐः',
            'singular_feminine': 'आयाम्', 'dual_feminine': 'आभ्याम्', 'plural_feminine': 'आभिः',
            'singular_neuter': 'एन', 'dual_neuter': 'आभ्याम्', 'plural_neuter': 'ऐः',
        },
        'चतुर्थी': {
            'singular_masculine': 'आय', 'dual_masculine': 'आभ्याम्', 'plural_masculine': 'एभ्यः',
            'singular_feminine': 'आयाम्', 'dual_feminine': 'आभ्याम्', 'plural_feminine': 'आभ्यः',
            'singular_neuter': 'आय', 'dual_neuter': 'आभ्याम्', 'plural_neuter': 'एभ्यः',
        },
        'पञ्चमी': {
            'singular_masculine': 'आत्', 'dual_masculine': 'आभ्याम्', 'plural_masculine': 'एभ्यः',
            'singular_feminine': 'आयाम्', 'dual_feminine': 'आभ्याम्', 'plural_feminine': 'आभ्यः',
            'singular_neuter': 'आत्', 'dual_neuter': 'आभ्याम्', 'plural_neuter': 'एभ्यः',
        },
        'षष्ठी': {
            'singular_masculine': 'अस्य', 'dual_masculine': 'अयोः', 'plural_masculine': 'आनाम्',
            'singular_feminine': 'आयाम्', 'dual_feminine': 'अयोः', 'plural_feminine': 'आनाम्',
            'singular_neuter': 'अस्य', 'dual_neuter': 'अयोः', 'plural_neuter': 'आनाम्',
        },
        'सप्तमी': {
            'singular_masculine': 'ए', 'dual_masculine': 'अयोः', 'plural_masculine': 'एषु',
            'singular_feminine': 'आयाम्', 'dual_feminine': 'अयोः', 'plural_feminine': 'आसु',
            'singular_neuter': 'ए', 'dual_neuter': 'अयोः', 'plural_neuter': 'एषु',
        }
    }

    if word.lower() in karaka_mapping:
        karaka = karaka_mapping[word.lower()]
        word_form = f'singular_{gender}'
        if gender == 'Dual':
            word_form = f'dual_{gender}'
        elif gender == 'Plural':
            word_form = f'plural_{gender}'

        if karaka in word_forms_mapping:
            # Generate the correct word form based on the vibhakti (case), gender, and number
            pratyaya = word_forms_mapping[karaka][word_form]
            return(karma[:-1] + pratyaya)

    return karma
   
def direct_sanskrit_translation(english_paragraph, verb_karma_dict):
    # Directly translate the original English paragraph to Sanskrit using Google Translate API
    translator = GoogleTranslator(source='en', target='sa')
    direct_translation = translator.translate(english_paragraph)

    return direct_translation

def refine_sanskrit_translation(direct_translation, sanskrit_translation, verb_karma_dict):
    # Split the translations into words
    direct_words = direct_translation.split()
    sanskrit_words = sanskrit_translation.split()

    # Initialize an empty list to store the refined translation
    refined_translation = []

    for i, (direct_word, sanskrit_word) in enumerate(zip(direct_words, sanskrit_words)):
        # Find the differences between the direct word and the Sanskrit word
        diff = difflib.ndiff(direct_word, sanskrit_word)

        # Initialize an empty string to store the refined word
        refined_word = ''

        for change in diff:
            # If the character is a space or equal, append it to the refined word
            if change[0] in (' ', '+'):
                refined_word += change[-1]

            # If the character is '-', it means the Sanskrit word has an extra character,
            # and we can skip it as we want to align the words
            elif change[0] == '-':
                continue

        # Append the refined word to the list
        refined_translation.append(refined_word)

    # Join the refined words to form the final refined translation
    final_refined_translation = ' '.join(refined_translation)

    return final_refined_translation

# Function that translates the English paragraph to a Basic Sanskrit Translation
def translate_to_sanskrit(english_sentence, verb_karma_dict):
    sanskrit_sentence = []
    words_to_translate = english_sentence.split()

    for word in words_to_translate:
        # Remove occurrences of the word "the" and Sanskrit numbers from the translation process
        if word.lower() == "the" or is_sanskrit_number(word):
            continue

        if word in verb_karma_dict:
            sanskrit_karma = verb_karma_dict.get(word, {}).get("Karma")
            if sanskrit_karma:
                # Use the first Sanskrit karma in the list for demonstration purposes
                karma = sanskrit_karma[0]
                # You can add more logic here to determine gender and case if available
                gender = "masculine"
                case = "nominative"

                # Get the Sanskrit word with the appropriate form using the get_sanskrit_karakas function
                sanskrit_word = get_sanskrit_karakas(word, karma, gender, case)
                sanskrit_sentence.append(sanskrit_word)
            else:
                # Use Google Translate as fallback for words not found in the dictionary
                google_translation = GoogleTranslator(source='en', target='sa').translate(word)
                sanskrit_sentence.append(google_translation)
        else:
            # Use Google Translate for words not found in the dictionary
            google_translation = GoogleTranslator(source='en', target='sa').translate(word)
            sanskrit_sentence.append(google_translation)

    # Remove any number symbols and periods from the Sanskrit translation
    sanskrit_sentence = [re.sub(r'[१२३४५६७८९०.]', '', word) for word in sanskrit_sentence]

    return ' '.join(sanskrit_sentence)

# Changing the Vibhakti of the words
def change_vibhakti(word, karakas_list):
    return f'{english_to_sanskrit_with_vibhakti.get(word, word)} ({karakas_list})' if word in karakas_list else english_to_sanskrit_with_vibhakti.get(word, word)

def main():

    english_paragraph = "Once there was a young woman named Vidya who lived in a village. She wanted to travel and explore the world, but she couldn't speak English. So, she joined an English course in her village. Vidya learned grammar, vocabulary and how to pronounce words. She practised talking with her classmates and understood simple conversations. Vidya felt more confident and decided to travel alone to an English speaking country. She visited big cities, talked to local people and learned about different cultures. Vidya realized that learning English opened doors to new experiences and helped her connect with people all over the world. Her journey showed that with determination and learning, anything is possible."
    kartru, karma, kriya, sampradana, apadana, karana, adhikarana, visheshana = identify_karakas(english_paragraph)
    print("Kartru (Subject):", kartru)
    print("Karma (Object):", karma)
    print("Kriya (Verb):", kriya)
    print("Sampradana (Goal):", sampradana)
    print("Apadana (Source):", apadana)
    print("Karana (Instrument):", karana)
    print("Adhikarana (Locus):", adhikarana)
    print("Visheshana-Visheshya (Adjective-Noun):", visheshana)
    
    output_file = "karaka_mind_map.png"
    create_mind_map_for_paragraph(english_paragraph, output_file)
    print("\nThe Karaka Mind Map has been created for the given paragraph and the image has been downloaded.\n")
    
    verb_karma_dict = {'lived': {'Kartru': ['विद्या'], 'Karma': [], 'Sampradana': [], 'Apadana': [], 'Karana': [], 'Adhikarana': ['ग्राम']}, 'speak': {'Kartru': ['सा'], 'Karma': ['आंग्ल'], 'Sampradana': [], 'Apadana': [], 'Karana': [], 'Adhikarana': []}, 'joined': {'Kartru': ['सा'], 'Karma': ['वर्गः'], 'Sampradana': [], 'Apadana': [], 'Karana': [], 'Adhikarana': []}, 'pronounce': {'Kartru': [], 'Karma': ['शब्दाः'], 'Sampradana': [], 'Apadana': [], 'Karana': [], 'Adhikarana': []}, 'understood': {'Kartru': [], 'Karma': ['संभाषणानि'], 'Sampradana': [], 'Apadana': [], 'Karana': [], 'Adhikarana': []}, 'travel': {'Kartru': [], 'Karma': [], 'Sampradana': ['देशः'], 'Apadana': [], 'Karana': [], 'Adhikarana': ['देशः']}, 'learned': {'Kartru': [], 'Karma': [], 'Sampradana': [], 'Apadana': [], 'Karana': [], 'Adhikarana': ['संस्कृतिः']}, 'connect': {'Kartru': ['तस्याः'], 'Karma': [], 'Sampradana': [], 'Apadana': [], 'Karana': ['जनाः'], 'Adhikarana': []}, 'is': {'Kartru': ['किमपि'], 'Karma': ['सम्भव'], 'Sampradana': [], 'Apadana': [], 'Karana': [], 'Adhikarana': []}, 'conquered': {'Kartru': [], 'Karma': ['विश्वम्'], 'Sampradana': [], 'Apadana': [], 'Karana': ['ग्रिट्'], 'Adhikarana': []}}
    
    print("\n",verb_karma_dict,"\n")
    
    sanskrit_translation = translate_to_sanskrit(english_paragraph, verb_karma_dict)
    
    print("Basic Sanskrit Translation:")
    print(sanskrit_translation)
    
    formatted_paragraph = replace_punctuation_with_space(english_paragraph)
    
    english_words = formatted_paragraph.split()

    # Translate each word using the dictionary
    translated_words = [english_to_sanskrit.get(word, word) for word in english_words]

    # Join the translated words back to form the translated paragraph
    translated_paragraph = " ".join(translated_words)

    # Add a full stop '|' after each sentence
    translated_paragraph = translated_paragraph.replace('.', ' | ')

    # Split the paragraph into sentences
    sentences = translated_paragraph.split('|')

    # Translate each sentence with Karakas using the dictionary and change vibhakti if applicable
    translated_sentences = []
    for sentence in sentences:
        sanskrit_words = sentence.split()
        translated_words = []
        for word in sanskrit_words:
            if word in kartru_karakas:
                translated_words.append(change_vibhakti(word, 'kartru'))
            elif word in karma_karakas:
                translated_words.append(change_vibhakti(word, 'karma'))
            elif word in kriya_karakas:
                translated_words.append(change_vibhakti(word, 'kriya'))
            elif word in sampradana_karakas:
                translated_words.append(change_vibhakti(word, 'sampradana'))
            elif word in apadana_karakas:
                translated_words.append(change_vibhakti(word, 'apadana'))
            else:
                translated_words.append(change_vibhakti(word, 'karana'))  # Assuming all other words are 'karana'

        # Join the translated words back to form the translated sentence
        translated_sentence = " ".join(translated_words)
        translated_sentences.append(translated_sentence)

    # Join the translated sentences back to form the translated paragraph with Karakas
    translated_paragraph_with_karakas = " | ".join(translated_sentences)

    # Print the translated paragraph with Karakas
    print("\nFinal Sanskrit Translation:")
    print(translated_paragraph_with_karakas)

if __name__ == "__main__":
    main()
