import argparse
from model import load_model, get_vector, model_id, pg_connect, pg_prepare

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default='word_lstm.pth')
p.add_argument('--text', required=True)
p.add_argument('--top', type=int, default=5)
a = p.parse_args()

if a.top < 1:
  p.error('top должен быть положительным')
model, vocab = load_model(a.checkpoint)
v = get_vector(a.text, model, vocab).numpy()
with pg_connect() as conn:
  pg_prepare(conn)
  with conn.cursor() as cur:
    cur.execute('''
      select  id,
              content,
              1 - (embedding <=> %s::vector) as similarity
      from    lstm_fragments
      where   model_id = %s
      order by embedding <=> %s::vector, id
      limit   %s
    ''', (v, model_id(a.checkpoint), v, a.top))
    rows = cur.fetchall()
for id, content, score in rows:
  print(f'{score:.5f} [{id-1:6d}] {content}')
if not rows:
  print('Нет векторов. Запустите 05_pg_load.py.')
