import argparse
import torch
from torch import nn
from model import WordLSTM, words, WINDOW

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--corpus', default='corpus.txt')
  parser.add_argument('--checkpoint', default='word_lstm.pth')
  parser.add_argument('--epochs', type=int, default=30)
  args = parser.parse_args()

  torch.manual_seed(42)
  tokens = words(open(args.corpus, encoding='utf-8').read())
  vocab = {'<unk>': 0}
  for word in tokens:
    if word not in vocab:
      vocab[word] = len(vocab)
  if len(tokens) <= WINDOW:
    print('В корпусе должно быть больше 10 слов')
    return

  ids = torch.tensor([vocab[w] for w in tokens], dtype=torch.long)
  x = torch.stack([ids[i:i + WINDOW] for i in range(len(ids) - WINDOW)])
  y = ids[WINDOW:]
  
  model = WordLSTM(len(vocab))
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
  criterion = nn.CrossEntropyLoss()
  
  for epoch in range(args.epochs):
    model.train()
    order = torch.randperm(len(x))
    total = 0.0
    for batch in order.split(128):
      optimizer.zero_grad()
      loss = criterion(model(x[batch]), y[batch])
      loss.backward()
      optimizer.step()
      total += loss.item() * len(batch)
    print(f'epoch {epoch + 1:3d}: loss={total / len(x):.4f}')
    
  torch.save({'model': model.state_dict(), 'vocab': vocab}, args.checkpoint)
  print(f'Слов: {len(tokens)}, словарь: {len(vocab)}, окон: {len(x)}')

if __name__ == '__main__':
  main()
