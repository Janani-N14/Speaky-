import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from spell import analyze

class PronunciationBot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bot_name = "Speaky"
        
        # Load intents
        with open('intents.json', 'r') as json_data:
            self.intents = json.load(json_data)

        # Load trained model
        FILE = "data.pth"
        data = torch.load(FILE)
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        self.model_state = data["model_state"]

        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def check_pronunciation(self):
        """Handle pronunciation check request"""
        word = input(f"{self.bot_name}: Enter the word you want to check: ")
        audio_path = input(f"{self.bot_name}: Enter the path to your recorded audio file (MP3/WAV): ")
        
        # Call the pronunciation analyzer
        result = analyze(word, audio_path)
        
        if 'error' in result:
            print(f"{self.bot_name}: {result['error']}")
        else:
            print(f"{self.bot_name}: Analysis Results:")
            print(f"Similarity Score: {result['similarity_score']:.2f}%")
            print(f"Your pronunciation: {result['transcription']}")
            if result['mismatched_words']:
                print(f"Words to work on: {', '.join(result['mismatched_words'])}")
            print(f"\nTips: {result['tips']}")

    def chat(self):
        print(f"{self.bot_name}: Let's chat! (type 'quit' to exit)")

        while True:
            user_input = input("You: ")##api process
            if user_input.lower() == "quit":
                break

            sentence = tokenize(user_input)
            X = bag_of_words(sentence, self.all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(self.device)

            output = self.model(X)
            _, predicted = torch.max(output, dim=1)
            tag = self.tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            if prob.item() > 0.75:
                for intent in self.intents['intents']:
                    if tag == intent["tag"]:
                        if tag in ["pronunciation", "check_similarity"]:
                            response = random.choice(intent['responses'])
                            print(f"{self.bot_name}: {response}")
                            self.check_pronunciation()
                        else:
                            print(f"{self.bot_name}: {random.choice(intent['responses'])}")
            else:
                print(f"{self.bot_name}: I don't understand. Could you rephrase that?")

if __name__ == "__main__":
    bot = PronunciationBot()
    bot.chat()