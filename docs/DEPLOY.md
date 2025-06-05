# Streamlit Cloud – Redeploy Steps

1. **Set project secrets**
   • `OPENAI_API_KEY`
   • `PERPLEXITY_API_KEY`
   • `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET`
   • *(optional)* `CRITIC_MODEL`, `ENABLE_VISION`

2. In **Advanced settings → Environment variables** copy-paste every key from `.env.template`.

3. Bump the app memory to **Medium** – recommended when processing vision images.

4. Click **"Deploy from main"** to trigger a fresh build.

5. Watch the logs until you see `App running on port 8501` then open the public URL.