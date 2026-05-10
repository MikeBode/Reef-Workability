import pandas as pd

def merge_visit_dfs(visit_dfs: list[pd.DataFrame], names: list[str], was_successful: list[bool]) -> pd.DataFrame:
    if len(visit_dfs) != len(names):
        raise ValueError("Length of visit_dfs and names must be the same.")
    
    if len(visit_dfs) != len(was_successful):
        raise ValueError("Length of visit_dfs and was_successful must be the same.")
    
    for df, name, success in zip(visit_dfs, names, was_successful):
        df['source_name'] = name
        df['was_successful'] = success
    
    merged_df = pd.concat(visit_dfs, ignore_index=True)
    return merged_df