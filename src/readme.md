cara jalanin sanitary_check.py
python src/sanitary_check.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data" --out results/sanitary

cara jalanin dataset.py
python src/dataset.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data" --variants original --exclude-file results/sanitary/excluded_files.txt --out splits/splits.csv