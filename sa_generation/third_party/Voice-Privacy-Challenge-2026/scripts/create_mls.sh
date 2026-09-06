#apython3 scripts/build_mls_german_data.py --language-dir corpora/MLS/mls_dutch --data-dir data/mls_dutch

cd ../ && for lang in de fr es en; do echo "=== $lang ==="; python3  scripts/build_mls_data.py --language-dir corpora/$lang --data-dir data/$lang; done