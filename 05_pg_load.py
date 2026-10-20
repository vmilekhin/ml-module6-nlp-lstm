import argparse
import random
import torch
from model import load_model, model_id, pg_connect, pg_prepare, words, WINDOW

p = argparse.ArgumentParser()
p.add_argument('--corpus', default='corpus.txt')
p.add_argument('--checkpoint', default='word_lstm.pth')
p.add_argument('--count', type=int, default=100)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()

if a.count < 1:
  p.error('count должен быть положительным')
model, vocab = load_model(a.checkpoint)
tokens = words(open(a.corpus, encoding='utf-8').read())
n = len(tokens)-WINDOW+1
if n < 1:
  p.error('Недостаточно слов')
count = min(a.count, n)
indices = random.Random(a.seed).sample(range(n), count)
ids = torch.tensor([vocab.get(w, 0) for w in tokens], dtype=torch.long)
version = model_id(a.checkpoint)
with pg_connect() as conn:
  pg_prepare(conn)
  with conn.cursor() as cur:
    cur.execute('''
      create table if not exists lstm_fragments (
        id         bigint primary key,
        model_id   text not null,
        content    text not null,
        embedding  vector(32) not null
      )
    ''')
    cur.execute('delete from lstm_fragments')
    with torch.inference_mode():
      for start in range(0, count, 128):
        selected = indices[start:start+128]
        batch = torch.stack([ids[i:i+WINDOW] for i in selected])
        vectors = model.encode(batch).numpy()
        rows = [(i+1, version, ' '.join(tokens[i:i+WINDOW]), vectors[j])
                for j, i in enumerate(selected)]
        cur.executemany('''
          insert into lstm_fragments (id, model_id, content, embedding)
          values (%s, %s, %s, %s)
        ''', rows)
        print(f'Сохранено {start+len(selected)}/{count}')
  conn.commit()
print(f'Готово: {count} случайных окон, seed={a.seed}')
