import re
import torch
from torch import nn

WINDOW = 10
SIZE = 32

def words(text):
  return re.findall(r"[а-яёa-z0-9]+", text.lower())

class WordLSTM(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, SIZE)
    self.lstm = nn.LSTM(SIZE, SIZE, batch_first=True)
    self.fc = nn.Linear(SIZE, vocab_size)

  def encode(self, x):
    _, (hidden, _) = self.lstm(self.embedding(x))
    return hidden[-1]

  def forward(self, x):
    return self.fc(self.encode(x))

def load_model(path='word_lstm.pth'):
  checkpoint = torch.load(path, map_location='cpu', weights_only=True)
  vocab = checkpoint['vocab']
  model = WordLSTM(len(vocab))
  model.load_state_dict(checkpoint['model'])
  model.eval()
  return model, vocab

def get_vector(text, model, vocab):
  tokens = words(text)
  if not tokens:
    raise ValueError('Пустой текст')
  ids = [vocab.get(w, 0) for w in tokens[-WINDOW:]]
  ids = [0] * (WINDOW - len(ids)) + ids
  with torch.inference_mode():
    return model.encode(torch.tensor([ids], dtype=torch.long))[0]

def model_id(path):
  import hashlib
  return hashlib.sha256(open(path, 'rb').read()).hexdigest()

def pg_connect():
  import os
  import psycopg
  return psycopg.connect(os.getenv('PG_DSN', 'postgresql://postgres@localhost:5432/postgres'))

def pg_prepare(conn):
  from pgvector.psycopg import register_vector
  with conn.cursor() as cur:
    cur.execute('create extension if not exists vector')
  conn.commit()
  register_vector(conn)
