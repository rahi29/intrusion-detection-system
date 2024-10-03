import pandas as pd

def filters(data):
    

    columns_to_remove = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'is_guest_login','su_attempted','wrong_fragment','urgent','hot','num_failed_logins','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','srv_diff_host_rate']

    data = data.drop(columns=columns_to_remove)

    data = data.loc[:, (data != 0).any(axis=0)]

    data.to_csv('./Data/filtered_data.csv', index=False)

    return data