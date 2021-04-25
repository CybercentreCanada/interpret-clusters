import pandas as pd
import numpy as np

def get_sample_data_row(team_list, size=5):
    selected = np.random.choice(team_list, size=size, replace=False)

    row_dict = {}
    for team in team_list:
        if team in selected:
            row_dict[team] = 1
        else:
            row_dict[team] = 0
    
    return row_dict

def create_sample_data(team_list, num_to_select, cluster_size):
    
    data = []
    for x in range(cluster_size):
        data.append(get_sample_data_row(team_list, size=num_to_select))
    
    df = pd.DataFrame(data)
    return df

def main():

    with open('data/nba_teams.txt', 'r') as f:
        nba_teams = [line.strip() for line in f.readlines()]
    with open('data/mlb_teams.txt', 'r') as f:
        mlb_teams = [line.strip() for line in f.readlines()]
    with open('data/nhl_teams.txt', 'r') as f:
        nhl_teams = [line.strip() for line in f.readlines()]
    with open('data/mls_teams.txt', 'r') as f:
        mls_teams = [line.strip() for line in f.readlines()]

    cluster_size = 100

    nhl_df = create_sample_data(nhl_teams, num_to_select=10, cluster_size=cluster_size)
    mlb_df = create_sample_data(mlb_teams, num_to_select=10, cluster_size=cluster_size)
    mls_df = create_sample_data(mls_teams, num_to_select=8, cluster_size=cluster_size)

    data_no_noise = pd.concat([nhl_df, mlb_df, mls_df]).fillna(0).reset_index(drop=True)

    data_no_noise.to_csv('data/synthetic_data.csv', index=False)

if __name__ == '__main__':
    main()