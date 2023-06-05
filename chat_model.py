from transformers import AutoTokenizer
from transformers import TFAutoModelForQuestionAnswering
import tensorflow as tf


class ChatModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
        self.model = TFAutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

        with open('context.txt', 'r') as file:
            self.context = file.read()

    def generate_response(self, question):
        inputs = self.tokenizer(question, self.context, add_special_tokens=True, return_tensors='tf')
        input_ids = inputs['input_ids'].numpy()[0]

        outputs = self.model(inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
        answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0]

        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return answer


chat_model = ChatModel()
