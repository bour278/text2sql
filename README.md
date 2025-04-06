# Text2SQL

## Description
📔 Text ➡️ SQL 🧑‍💻

## Tools
- [LangChain SQL Q&A Tutorial](https://python.langchain.com/docs/tutorials/sql_qa/) 🦜🔗
- [Vanna.AI](https://vanna.ai/) 🔮
- [LlamaIndex](https://www.llamaindex.ai/) 🦙

## Tutorials
- [How to Use LangChain to Build a Text-to-SQL Solution](https://medium.com/@marvin_thompson/how-to-use-langchain-to-build-a-text-to-sql-solution-54a173f312a5)
- [Text2SQL GitHub Repository](https://github.com/WeitaoLu/Text2SQL)
- [Text2SQL Workshop GitHub Repository](https://github.com/weet-ai/text2sql-workshop)

## Papers
- [A Survey on Employing Large Language Models for Text-to-SQL Tasks](https://arxiv.org/html/2407.15186v2)
- [PET-SQL: A Prompt-enhanced Two-stage Text-to-SQL Framework with Cross-consistency](https://arxiv.org/html/2403.09732v1)
- [SeaD: End-to-end Text-to-SQL Generation with Schema-aware Denoising](https://arxiv.org/pdf/2105.07911)
- [Next-Generation Database Interfaces:A Survey of LLM-based Text-to-SQL](https://arxiv.org/pdf/2406.08426)

## Running

1 - install requirements
```bash
pip install requirements.txt
```

2- navigate to `/data` and run `sqlite-synthetic.py` to create a toy dataset
```bash
cd data
python sqlite-synthetic.py
```
_after this step you should see a `synthetic_data.db` in `/src`_

3- navigate to `src` and run either `main.py` or the UI:

For command line interface:
```bash
cd ../src
# Use OpenAI (default)
python main.py

# Or use Google's Gemini model
python main.py -gemini
```

For the web interface:
```bash
cd ../src
# Run the Streamlit UI
python run_ui.py
```

_Optional_ run `visualize_workflows.py` to show workflow graphs

## API Keys

Create a `keys.env` file in the `src/agents` directory with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_gemini_key_here # Optional, only if using Gemini
```


## Development

See [TODO.md](TODO.md) for planned features and improvements.

## Implementaions

Langgraph Workflow 🦜

- Master Workflow
<p align="center">
  <img src="/src/workflow_visualizations/master_workflow.png" alt="Master Workflow"/>
</p>

- Python Workflow
<p align="center">
  <img src="/src/workflow_visualizations/python_workflow.png" alt="Python Workflow"/>
</p>

- SQL Workflow
<p align="center">
  <img src="/src/workflow_visualizations/sql_workflow.png" alt="SQL Workflow"/>
</p>