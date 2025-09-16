# Project Setup

## 1. Set Python Version

If your project includes a `.python-version` file, `pyenv` will automatically use the correct version. Verify it by running:

```bash
python --version  # should show 3.12.x
```

If you need to set it manually:

```bash
pyenv local 3.12.6
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

# WatsonX Configuration

This project uses **WatsonX Guardrails** as outlined in the [IBM tutorial](https://www.ibm.com/think/tutorials/llm-guardrails).

1. Ensure you have a WatsonX project created in a supported region (e.g. `us-south`).
2. Configure your `.env` file with the following values:

```env
WATSONX_API_KEY=your-api-key-here
WATSONX_API_URL=https://us-south.ml.cloud.ibm.com
```

3. Verify your setup by running a small test request:

```bash
python src/main.py
```

If successful, you should see a guardrails response (e.g. `unsafe S13`).

---

# Running the App

Run the application from the project root:

```bash
python src/main.py
```
