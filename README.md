# clinical-ai-data

Dear clinical AI data specialist candidate, this is the take home project for the technical portion of this interview. Using AI to help you with this project is not only allowed, it is encouraged. [Claude](https://claude.ai/), [Perplexity](https://www.perplexity.ai/) and [ChatGPT](https://chatgpt.com/) are all great options. Even though AI use is encouraged, understanding and good design are still important, so as much as you can, try to understand what you do, well enough to explain why you did it, AI can help with that as well. 

open your Terminal and in the folder of your choosing, clone the repository to your local computer, then go into the repo.

```
git clone git@github.com:clam-sdx/clinical-ai-data.git
cd clinical-ai-data
```

### install `uv` with one of

```python
curl -LsSf https://astral.sh/uv/install.sh | sh
wget -qO- https://astral.sh/uv/install.sh | sh
```

If `which -a uv` returns nothing, then the "shadowed commands" warning can be ignored.
If `echo $SHELL` returns `/bin/bash` or `/bin/zsh`, use: `source $HOME/.local/bin/env`

### use uv to setup a python virtual environment

Install a version of python and use that python version in a virtual environment

```bash
uv python install 3.12
uv venv .venv --python 3.12
source .venv/bin/activate
```

At this point you should see a change to your terminal such that `(.venv)` appears at the beginning

```
josh@Joshs-MacBook-Pro clinical-ai-data % source .venv/bin/activate
(.venv) josh@Joshs-MacBook-Pro clinical-ai-data % 
```

```
(.venv) josh@Joshs-MacBook-Pro clinical-ai-data % uv pip install -e .
```

You should see several packages downloaded, ie:

```
....
 + websocket-client==1.8.0
 + widgetsnbextension==4.0.14
 + xxhash==3.6.0
 + yarl==1.20.1
 ...
~ clinical-ai-data==0.1.0 (from file:///Users/carson/Desktop/Projects/clinical-ai-data)
```

At this point you should be able to do this:

```
(.venv) josh@Joshs-MacBook-Pro clinical-ai-data % python
Python 3.12.9 (main, Feb 12 2025, 15:09:19) [Clang 19.1.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from clinical_ai.data_utils import hello_world
>>> hello_world()
'Hello, World!'
>>> exit()
(.venv) josh@Joshs-MacBook-Pro clinical-ai-data %
```

Make the environment explicitly available for selection as an ipython kernel in your jupyter notebook

```
uv run python -m ipykernel install --user \
--name smarterdx_env \
--display-name "Python (smarterdx_env)"
```

now open up a jupyter notebook

```
jupyter notebook
```

open the jupyter notebook notebook.ipynb using the Python (smarterdx_env) kernel. 
there is already example code inside to give you the context behind the tasks for you to complete. 
learn from the descriptions of the code in each cell of the notebook. use AI to help you complete the 3 tasks at the end.

send just the notebook.ipynb to your technical interviewer by email and when you meet be ready to explain how you approached the 3 tasks.