# a) Install dependencies
pip install --user -r requirements.txt

# b) Run all necessary parts of the codebase
# Since the README.md does not specify running parts in parallel, we'll run them sequentially
python src/train.py
python src/evaluate.py
python src/visualize.py
