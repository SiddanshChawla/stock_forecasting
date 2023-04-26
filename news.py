from rake_nltk import Rake

def extract_phrases(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    phrases = rake.get_ranked_phrases_with_scores()

    # Extract only phrases without scores
    important_phrases = [phrase for _, phrase in phrases]
    
    return important_phrases
