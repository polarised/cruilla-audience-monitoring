import pandas as pd
from datetime import datetime, timedelta
import random
import psycopg2

df = pd.read_csv(r'C:\Users\migue\OneDrive\Escritorio\UAB_INTELIGENCIA_ARTIFICIAL\Tercer_Any\3B\Synthesis Project II\results_annotated.csv')
# Conectar a PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="sentiment_analysis",
    user="postgres",
    password="Pitufo09"
)
cur = conn.cursor()

for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO sentiment (tweet, clause, topic, sentiment, sentiment_score)
        VALUES (%s, %s, %s, %s, %s)
    """, (row['tweet'], row['clause'], 
          row['topic'], row['sentiment'], 
          row['sentiment_score']))

conn.commit()
cur.close()
conn.close()
