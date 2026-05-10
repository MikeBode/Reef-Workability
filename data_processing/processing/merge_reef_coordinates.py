# Matches reef names between the survey coordinate dataset and the vessel/COTS dataset
# using fuzzy string normalisation, then merges geographic coordinates into the vessel data.

import pandas as pd
import numpy as np
import re


def normalize_reef_name(name):
    name = str(name).lower()

    if name == "Batt 16-029".lower():
        pass

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

# Where there are multiple potential matches, we manually choose.
# TODO: Review these, check resonably central.
# TODO 2: Consider replacing with some sort of xy dataset.
MULTI_MATCH_DICT = {
    "fitzroy island": "fitzroy island 16-054a",
    "kelso": "kelso 18-030",
    "little lizard": "lizard island 14-116a",
    "lizard island": "lizard island 14-116a",
    "mackay": "mackay 16-015",

    # TODO: Really check these.
    "big broadhurst no1": "broadhurst 18-100a",
    "big broadhurst no2": "broadhurst 18-100a",
    "big broadhurst no3": "broadhurst 18-100a",
    "big broadhurst no4": "broadhurst 18-100a",

    "michaelmas 16-062": "michaelmas 16-060",

    "ribbon no3": "ribbon 14-146 10",
    "ribbon no4": "ribbon 14-146 10",
    "round russel": "round-russell 17-013",
    "startle 15-028": "startle reefs 15-028",
    "flynn 16-071": "flynn 16-065",

    "u/n 15-071c": "u/n 10-299",

    "round russell": "round-russell 17-013",
}

def manual_multiple_match_reconcilliation(name):
    if name in MULTI_MATCH_DICT:
        return MULTI_MATCH_DICT[name]
    return None

def get_best_match(name, lookup_dict):
    manual_match = manual_multiple_match_reconcilliation(name)
    if manual_match:
        return manual_match

    if name in lookup_dict:
        return name

    base_name = re.sub(r'\d+', '', name).strip()
    base_name = re.sub(r'\s+', ' ', base_name)

    potential_matches = [
        k for k in lookup_dict.keys()
        if base_name and (base_name in k or k in base_name)
    ]

    if len(potential_matches) > 1:
        raise Exception(f"Multiple potential matches found for '{name}': {potential_matches}")

    if len(potential_matches) == 1:
        if potential_matches[0] == '':
            raise Exception(f"Potential match for '{name}' is an empty string.")

        if potential_matches:
            return potential_matches[0]

    return None

def merge_reef_datasets(locations_file, vessel_file, output_file=None):
    locations_df = pd.read_csv(locations_file)
    vessel_df = pd.read_excel(vessel_file)
    vessel_df = vessel_df[vessel_df["Reason"] == "Strong wind and swell conditions"]

    print(f"Loaded locations dataset with {len(locations_df)} rows")
    print(f"Loaded vessel dataset with {len(vessel_df)} rows")
    
    locations_df.dropna(subset=['reefName'], inplace=True)

    locations_df['normalized_name'] = locations_df['reefName'].apply(normalize_reef_name)
    vessel_df['normalized_name'] = vessel_df['Reef'].apply(normalize_reef_name)

    valid_locations = locations_df[locations_df['normalized_name'] != '']

    coord_lookup = valid_locations.groupby('normalized_name').agg({
        'x': 'mean',
        'y': 'mean'
    }).to_dict('index')

    print(f"\nCreated coordinate lookup for {len(coord_lookup)} unique normalized reef names")

    vessel_df['matched_with'] = pd.NA

    def get_coordinates(row):
        normalized_name = row['normalized_name']
        if not normalized_name:
            return pd.Series([np.nan, np.nan])

        if normalized_name in coord_lookup:
            vessel_df.loc[row.name, 'matched_with'] = normalized_name
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
    
    if matched_count < len(vessel_df):
        unmatched = vessel_df[vessel_df['x'].isna()]
        print(f"Unmatched reefs:")
        for _, row in unmatched.iterrows():
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
        return vessel_df
    else:
        return vessel_df