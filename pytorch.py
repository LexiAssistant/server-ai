import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# 샘플 데이터
texts = [
    "자연어 처리는 컴퓨터 과학, 인공지능, 언어학 분야의 연구 주제입니다.",
    "자연어 처리는 텍스트 데이터에서 유의미한 정보를 추출하는 데 사용됩니다.",
    "자연어 처리 기술은 검색 엔진, 번역기, 챗봇 등에 사용됩니다."
]
keywords = [
    ["자연어 처리", "컴퓨터 과학", "인공지능", "언어학"],
    ["자연어 처리", "텍스트 데이터", "유의미한 정보"],
    ["자연어 처리", "검색 엔진", "번역기", "챗봇"]
]

class TextDataset(Dataset):
    def __init__(self, texts, keywords, vocab=None):
        self.texts = texts
        self.keywords = keywords
        self.vocab = vocab or self.build_vocab(texts)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([kw for kws in keywords for kw in kws])
        
        self.text_sequences = [self.text_to_sequence(text) for text in texts]
        self.keyword_sequences = [self.labels_to_sequence(kws) for kws in keywords]

    def build_vocab(self, texts):
        words = [word for text in texts for word in text.split()]
        vocab = {word: idx+1 for idx, (word, _) in enumerate(Counter(words).items())}
        vocab['<PAD>'] = 0
        return vocab

    def text_to_sequence(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]

    def labels_to_sequence(self, labels):
        return self.label_encoder.transform(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.text_sequences[idx]), torch.tensor(self.keyword_sequences[idx])

class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(KeywordRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return out

vocab_size = len(TextDataset(texts, keywords).vocab)
embed_size = 128
hidden_size = 128
output_size = len(LabelEncoder().fit([kw for kws in keywords for kw in kws]).classes_)

dataset = TextDataset(texts, keywords)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = KeywordRNN(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for text, keyword in train_loader:
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, keyword)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict(model, text, vocab, label_encoder):
    model.eval()
    with torch.no_grad():
        text_sequence = torch.tensor([vocab.get(word, 0) for word in text.split()]).unsqueeze(0)
        output = model(text_sequence)
        _, predicted = torch.max(output, 1)
        return label_encoder.inverse_transform(predicted.numpy())[0]

test_text = "이 문서는 예로부터 전해져 온 문서로 정확한 문서 작성법과 이해를 돕기위해 존재합니다. 어렵지만 이해를 한다면 핵심적인 역할을 수행할 수 있으므로 필수적으로 이해 및 활용법에 대해 익히는 것이 올바른 방법입니다."
predicted_keyword = predict(model, test_text, dataset.vocab, dataset.label_encoder)

print(f"텍스트: {test_text}")
print(f"예측된 키워드: {predicted_keyword}")
