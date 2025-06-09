import psycopg2

conn = psycopg2.connect(
    database="igdrasil",
    user="clouduser",
    password="Mnb@sdpoi87",
    host="172.24.0.22",  # Ejemplo: '200.100.50.25'
    port="5432"
)

cur = conn.cursor()
cur.execute("SELECT * FROM users;")
print("Total rows are: ", cur.rowcount)
rows = cur.fetchall()
for row in rows:
    print(row)

cur.close()
conn.close()
