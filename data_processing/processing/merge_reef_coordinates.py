# Matches reef names between the survey coordinate dataset and the vessel/COTS dataset
# using fuzzy string normalisation, then merges geographic coordinates into the vessel data.

import pandas as pd
import numpy as np
import re


def normalize_reef_name(name):
    if pd.isna(name):
        return ''

    name = str(name)

    if not name.strip():
        return ''

    name = name.lower()

    reef_number = re.search(r'(\d+-\d+[a-z]?)', name)
    reef_number = reef_number.group(1) if reef_number else ''

    name = re.sub(r'\(\d+-\d+[a-z]?\)', '', name)

    no_match = re.search(r'no\s+(\d+)', name)
    no_number = no_match.group(1) if no_match else ''
    name = re.sub(r'no\s+\d+', '', name)

    name = re.sub(r'\breef\b', '', name)
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    result = name

    if reef_number and reef_number not in result:
        result = f"{result} {reef_number}".strip()

    if no_number and no_number not in result:
        result = f"{result} {no_number}".strip()

    return result


def get_best_match(name, lookup_dict):
    if name in lookup_dict:
        return name

    base_name = re.sub(r'\d+', '', name).strip()
    base_name = re.sub(r'\s+', ' ', base_name)

    potential_matches = [
        k for k in lookup_dict.keys()
        if base_name and (base_name in k or k in base_name)
    ]

    if potential_matches:
        return sorted(potential_matches, key=lambda x: abs(len(x) - len(name)))[0]

    for key in lookup_dict.keys():
        if name in key or key in name:
            return key

    name_part = re.sub(r'\d+-\d+[a-z]?', '', name).strip()
    if name_part != name:
        for key in lookup_dict.keys():
            if name_part in key:
                return key

    return None


def merge_reef_datasets(locations_file, vessel_file, output_file=None):
    locations_df = pd.read_csv(locations_file)
    vessel_df = pd.read_excel(vessel_file)

    print(f"Loaded locations dataset with {len(locations_df)} rows")
    print(f"Loaded vessel dataset with {len(vessel_df)} rows")

    nan_count = locations_df['reefName'].isna().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in reefName column")

    locations_df['normalized_name'] = locations_df['reefName'].apply(normalize_reef_name)
    vessel_df['normalized_name'] = vessel_df['Reef'].apply(normalize_reef_name)

    print("\nNormalization examples from location dataset:")
    for i in range(min(5, len(locations_df))):
        print(f"Original: {locations_df.iloc[i]['reefName']} -> Normalized: {locations_df.iloc[i]['normalized_name']}")

    print("\nNormalization examples from vessel dataset:")
    for i in range(min(5, len(vessel_df))):
        print(f"Original: {vessel_df.iloc[i]['Reef']} -> Normalized: {vessel_df.iloc[i]['normalized_name']}")

    valid_locations = locations_df[locations_df['normalized_name'] != '']

    coord_lookup = valid_locations.groupby('normalized_name').agg({
        'x': 'mean',
        'y': 'mean'
    }).to_dict('index')

    print(f"\nCreated coordinate lookup for {len(coord_lookup)} unique normalized reef names")

    vessel_df['matched_with'] = ''

    def get_coordinates(row):
        normalized_name = row['normalized_name']
        if not normalized_name:
            return pd.Series([np.nan, np.nan])

        if normalized_name in coord_lookup:
            return pd.Series([coord_lookup[normalized_name]['x'], coord_lookup[normalized_name]['y']])

        best_match = get_best_match(normalized_name, coord_lookup)
        if best_match:
            vessel_df.loc[row.name, 'matched_with'] = best_match
            return pd.Series([coord_lookup[best_match]['x'], coord_lookup[best_match]['y']])

        return pd.Series([np.nan, np.nan])

    vessel_df[['x', 'y']] = vessel_df.apply(get_coordinates, axis=1)

    matched_count = vessel_df['x'].notna().sum()
    print(f"\nSuccessfully matched coordinates for {matched_count} out of {len(vessel_df)} entries "
          f"({matched_count / len(vessel_df) * 100:.1f}%)")

    print("\nSample of successful matches:")
    successful_matches = vessel_df[vessel_df['x'].notna()].head(10)
    for _, row in successful_matches.iterrows():
        print(f"'{row['Reef']}' matched with '{row['matched_with']}'")

    if matched_count < len(vessel_df):
        unmatched = vessel_df[vessel_df['x'].isna()]
        print(f"\nSample of unmatched reefs (up to 10):")
        for _, row in unmatched.head(10).iterrows():
            print(f"'{row['Reef']}' (normalized: '{row['normalized_name']}') - No match found")

            if row['normalized_name']:
                similar_names = [
                    k for k in coord_lookup.keys()
                    if any(word in k for word in row['normalized_name'].split())
                ][:3]
                if similar_names:
                    print(f"  Similar entries in location dataset: {', '.join(similar_names)}")

    vessel_df = vessel_df.drop(columns=['normalized_name', 'matched_with'])

    if output_file:
        vessel_df.to_excel(output_file, index=False)
        print(f"\nMerged dataset saved to {output_file}")
        return None
    else:
        return vessel_df


merged_df = merge_reef_datasets(
    'Data/surveyData[63].csv',
    'COTS INLOC Weather impacts.xlsx',
    'COTS INLOC Weather impacts-WithCoor.xlsx'
)
