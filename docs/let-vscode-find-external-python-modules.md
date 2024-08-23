
To let vscode IDE find definition of function defined in huggingface models, you set `PYTHONPATH` in your `${workspaceFolder}/.env` file.
Suppose that we have download a model to local directory `/data/hg_models`. Your `.env` should look like this:

```bash
PYTHONPATH="/data/hg_models"
```