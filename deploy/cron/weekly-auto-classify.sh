#!/usr/bin/env bash
# Weekly: regenerate labels/auto_labels.csv against the latest weak labels +
# masks. Run AFTER weekly-metar, weekly-goes, weekly-local-sensors so it sees
# fresh data.

source "$(dirname "$0")/_common.sh"

log "Auto-classify batch"
"${VENV}/bin/python" auto_classify_batch.py

# Summary line for the log
.venv/bin/python -c "
import csv
from collections import Counter
rows = list(csv.DictReader(open('labels/auto_labels.csv')))
hand = sum(1 for _ in open('labels/hand_labeled.csv')) - 1 if __import__('pathlib').Path('labels/hand_labeled.csv').exists() else 0
dist = Counter(r['auto_class'] for r in rows)
conf = Counter(r['auto_confidence'] for r in rows)
print(f'    classes={dict(dist)}  conf={dict(conf)}  hand_labels={hand}')
"
log "Auto-classify done"
