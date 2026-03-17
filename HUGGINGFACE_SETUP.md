# 🤗 HuggingFace AI Setup Guide

This application supports optional AI-powered contextual analysis of sentiment data using the HuggingFace Inference API (Llama-3.2-3B-Instruct).

---

## 📋 Prerequisites

- A free HuggingFace account: https://huggingface.co/join
- Access to the model `meta-llama/Llama-3.2-3B-Instruct` (requires accepting the model license)

---

## 🔑 Getting Your API Token

1. Log in to [huggingface.co](https://huggingface.co).
2. Go to **Settings → Access Tokens**: https://huggingface.co/settings/tokens
3. Click **"New token"**, choose **"Read"** role and give it a name.
4. Copy the token (starts with `hf_`).

---

## ⚙️ Configuring the Token

### Option A — Streamlit Secrets (recommended for Streamlit Cloud)

Create or update `.streamlit/secrets.toml`:

```toml
HF_API_TOKEN = "hf_your_token_here"

# Optional: also configure Groq for an alternative AI provider
GROQ_API_KEY = "gsk_your_groq_key_here"
```

> ⚠️ **Never commit `secrets.toml` to version control.** It is listed in `.gitignore` by default.

### Option B — Environment Variable (local / Docker)

```bash
export HF_API_TOKEN="hf_your_token_here"
streamlit run app.py
```

Or in a `.env` file (if you use `python-dotenv`):

```env
HF_API_TOKEN=hf_your_token_here
```

---

## 🚀 Usage in the App

Once configured, open the **🤖 Insights con IA** tab:

1. Select a **Línea de Negocio** (business line) or choose "Todas las líneas".
2. In the **Proveedor de IA** dropdown you will see **🤗 HuggingFace (Llama 3.2)** as an option.
3. Click **🔮 Generar Insights**.

The app calls the HuggingFace Inference API and returns contextual insights tailored to the Colombian insurance sector.

---

## 🔄 Fallback Behaviour

| Condition | Behaviour |
|---|---|
| `HF_API_TOKEN` configured | Uses HuggingFace Llama-3.2-3B-Instruct |
| `GROQ_API_KEY` configured | Uses Groq Llama-3.1-70B |
| Neither configured | Uses statistical fallback analysis |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `401 Unauthorized` | Token is invalid or expired — regenerate it |
| `403 Forbidden` | Accept the Llama 3.2 model license on HuggingFace |
| `503 Service Unavailable` | Model is loading — wait ~30 seconds and retry |
| Timeout | Normal on first call — model cold-starts; try again |

---

## 🔒 Security Notes

- **Never hardcode** your token in `app.py` or any source file.
- Use Streamlit Secrets or environment variables only.
- Rotate your token if it is accidentally exposed.
