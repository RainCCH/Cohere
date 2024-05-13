import cohere
from tqdm import tqdm
import time

def test():
    co = cohere.Client("oPnVCi96WtuBRjnRKKugBG3q7H3shQnUElfXdtce")

    chat_history = []
    max_turns = 10

    for _ in range(max_turns):
        # get user input
        message = input("Send the model a message: ")
        
        # simulate loading bar while generating a response
        with tqdm(total=100, desc="Generating response", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for i in range(100):
                time.sleep(0.02)  # simulate progress
                pbar.update(1)
        
        # generate a response with the current chat history
        response = co.chat(
            message=message,
            temperature=0.3,
            chat_history=chat_history
        )
        answer = response.text
            
        print(answer)

        # add message and answer to the chat history
        user_message = {"role": "USER", "text": message}
        bot_message = {"role": "CHATBOT", "text": answer}
        
        chat_history.append(user_message)
        chat_history.append(bot_message)

def classify():
    co = cohere.Client("oPnVCi96WtuBRjnRKKugBG3q7H3shQnUElfXdtce")
    sentences = [
        "I absolutely loved the movie, it was fantastic!",
        "The food at the restaurant was disappointing.",
        "What a beautiful day! I'm feeling so happy and grateful.",
        "This product is not worth the money. Poor quality.",
        # Add more sentences here
    ]

    model_id = "classify"
    use_case = "sentiment_analysis"

    classified_sentences = co.classify(model=model_id, inputs=sentences)

    for sentence, classification in zip(sentences, classified_sentences):
        sentiment = classification.classification
        probability = classification.confidence_score
        print(f"Sentence: {sentence}")
        print(f"Sentiment: {sentiment} (Confidence: {probability:.2%})")
        print()

# test()
classify()
