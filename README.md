# Tenant Search Benchmark

Compare lexical ranking with hashed feature-vector cosine similarity and reproduce exact ranking in PostgreSQL with pgvector.

[Source](project.py) · [Demo evidence](evidence/pgvector.json) · [Verification](evidence/checks.json)

## Run locally

The baseline and Python vector comparisons need Python 3.11+. For the SQL check, install the requirements and start the dedicated pgvector container.

```sh
python project.py
python -m unittest discover -v
python -m pip install -r requirements.txt
docker compose up --detach --wait
PORT=$(docker compose port db 5432 | cut -d: -f2)
python project.py --pg-dsn "postgresql://search:local-demo@127.0.0.1:$PORT/search"
docker compose down --volumes
```

On PowerShell, use `$vectorPort = ((docker compose port db 5432) -split ':')[-1]` and use `$vectorPort` in the connection string. The SQL check confirms that the exact ranking matches the offline scorer on the fixtures.

## Engineering decision

Select the customer boundary before scoring. Compare original query terms against predeclared synonym expansion using the same documents, top-k and labels.

Eight authored queries and deterministic features. The synonym fixture and smoke-size corpus do not represent a learned production embeddings system.



[Technology or project reference](https://github.com/pgvector/pgvector)
