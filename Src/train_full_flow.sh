set -e
python src/data_prep.py
python src/train_abuse.py
python src/train_crisis.py
python src/evaluate.py
echo "Training complete. Models saved under models/"