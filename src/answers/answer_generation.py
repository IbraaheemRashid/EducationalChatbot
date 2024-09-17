from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class AnswerGenerator:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate_answer(self, query, context):
        input_text = f"Question: {query}\n\nContext: {context}"
        inputs = self.tokenizer([input_text], max_length=1024, return_tensors='pt', truncation=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        
        answer = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return answer
