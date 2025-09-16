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

3. Verify your setup by running a test to WatsonX:

```bash
python src/run-example.py -f test_image_guardrails
```

If successful, you should see a guardrails response (e.g. `unsafe S13`).

---

# Azure Cognitive Search Setup

1. **Create a Resource Group**

In the Azure Portal, create a resource group (e.g. `test_resource_group`).

2. **Create Azure AI Search Resource**

Provision a **Free (F1)** tier Cognitive Search resource (e.g. `test-bank-chat-search`).

3. **Configure Environment Variables**

Add the following to your `.env` file:

```env
AZURE_SEARCH_API_URL=https://<your-search-service>.search.windows.net
AZURE_SEARCH_API_PRIMARY_ADMIN_KEY=<your-primary-admin-key>
AZURE_SEARCH_API_INDEX=bank-faq
```

4. **Create an Index**

Run the provided script to create your search index:

```bash
python src/run-script.py -f create-ai-search-index
```

5. **Upload Sample Documents**

Use the upload script to add sample FAQ data:

```bash
python src/run-script.py -f upload-documents
```

6. **Verify in Portal**

Use the Azure Portal **Search Explorer** to run a test query:

```json
{ "search": "deposit" }
```

You should see your document returned.

---

# Retrieval + Guardrails Test

You can run a deterministic retrieval + guardrails test:

```bash
python src/run-example.py -f test_retrieval_with_guardrails
```

This will:

* Run WatsonX guardrails against known safe queries.
* Query Azure Cognitive Search for sample data.
* Assert that expected content is returned.
* Print PASS/FAIL results for each test case.

---

# To Do / Next Steps

* ⏳ Introduce a minimal LLM to synthesize answers.
* ⏳ Build a full RAG pipeline by combining query + context into a single prompt.
* ⏳ Add post-generation guardrails to validate LLM output before returning to the user.
