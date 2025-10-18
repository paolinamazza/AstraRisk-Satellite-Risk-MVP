import pandas as pd

leo = pd.read_csv('leo_satellites_enriched.csv')
print(f'Total rows: {len(leo)}')
print(f'Unique IDs: {leo["NORAD_CAT_ID"].nunique()}')

dups = leo[leo.duplicated('NORAD_CAT_ID', keep=False)]
print(f'Duplicates: {len(dups)}')

if len(dups) > 0:
    print("\nDuplicate IDs:")
    print(dups[['NORAD_CAT_ID', 'OBJECT_NAME']].head(20))
