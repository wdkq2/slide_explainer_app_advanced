# Advanced Lecture Slide Summarisation Application

This folder contains an enhanced version of the lecture slide summarisation tool. The original project grouped slides by simple differences in text length; this advanced version uses a hybrid approach that combines semantic, visual, and structural cues to identify topic boundaries and detect near‑duplicate slides. It then summarises the unique pages and produces a downloadable document.

## Key Improvements

* **Hybrid Segmentation:** Pages are divided into groups by analysing semantic change (via OpenAI embeddings), visual change (via perceptual hash) and title changes. This reduces misclassifications for image‑heavy slides or slides with similar text lengths but different topics.
* **Duplicate Detection:** Pages that are nearly identical in content (either by cosine similarity of embeddings or low perceptual hash distance) are marked as duplicates. Only the first occurrence is summarised; subsequent duplicates are noted as such.
* **Configurable Weights:** The relative importance of semantic vs. visual vs. title changes can be adjusted via command‑line options.
* **Target Ratio Splitting:** When splitting into two or more groups, a target ratio can be specified to bias where the cut should occur (e.g. 0.55 for a 20/18 split in a 38‑page document).

## Quickstart

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   ```
3. **Run the CLI tool**
   ```bash
   python -m slide_explainer_app_advanced.main \
     --pdf /path/to/lecture.pdf \
     --title "Lecture Summary" \
     --openai-key $OPENAI_API_KEY \
     --output-dir ./output
   ```
   The generated `<title>.txt` file will be saved in the specified directory.

## Streamlit App

To use a web interface, run:

```bash
streamlit run streamlit_app.py
```

Upload a PDF and provide your OpenAI API key in the sidebar. Once processing completes, a download button for the result will appear.

### Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) and click **New app**.
3. Select the repository and set the file path to `streamlit_app.py`.
4. In the app settings, add your OpenAI API key under **Secrets**.
5. Deploy the app and share the URL with users.

## Usage

Optional CLI flags such as `--mode`, `--model`, and `--temperature` can be supplied as needed. Some OpenAI models (for example, `gpt-5-mini`) only accept the default temperature of `1`. If you request an unsupported value, the application now falls back to the model's default to avoid API errors.

### Important Notes

* **Poppler requirement:** The advanced algorithm tries to convert pages to images using ``pdf2image``. This requires the Poppler library to be installed on your system. If Poppler is missing, visual features are skipped and segmentation falls back to semantic and title cues only.
* **Embedding API costs:** Computing embeddings for every page consumes tokens. For long documents consider batching and using more economical models like ``text-embedding-3-small``.
* **Weights and thresholds:** You can adjust the weights for semantic/visual/title features and thresholds for duplicate detection by editing the ``advanced_split_pdf`` function in ``pdf_processor.py``. This may be necessary to fine‑tune the behaviour for different types of slides.
* **Duplicate slides:** Pages flagged as duplicates will have the summary of their canonical page followed by ``(중복 슬라이드)`` in the final document.

For additional details about configuring OpenAI authentication, refer to the README in the parent project folder.
