# Task 3 Report (Dashboard) — How to Run & Deploy

This repo includes an interactive dashboard for **Task 3 (Report stage)** implemented in **Streamlit + Altair**.

## Run locally

1. Activate your environment.
2. From the project root:

```bash
streamlit run streamlit_app.py
```

The dashboard loads `data/cleaned_dataset.csv`, rebuilds the PCA + KMeans model on 2019 indicators, and links the views with true brushing & linking (Altair selections).

## Deploy (Railway)

Railway can run this as a web service.

1. Create a new Railway project (from GitHub or by uploading the repo).
2. Ensure the service uses `requirements.txt`.
3. Set the start command to the `Procfile` command (or let Railway detect it):

```bash
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

4. Deploy and copy the public URL for submission.

## Deploy (Streamlit Community Cloud)

If you are happy to host from a GitHub repo:
1. Push this repo to GitHub.
2. Create a new Streamlit app pointing to `streamlit_app.py`.
3. Use the generated public URL for submission.

