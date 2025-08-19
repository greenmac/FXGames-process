# [macos]$ uv pip install pandas requests beautifulsoup4 torch torchvision torchaudio
from datetime import datetime
import os
import time

def timer(func):
    def wrap(*args, **kwargs):
        start_time_dt = record_time()
        start_time = calculate_time()
        print(divider())

        func(*args, **kwargs)

        end_time = calculate_time()
        end_time_dt = record_time()
        
        cost_time = end_time-start_time
        m, s = divmod(cost_time, 60)
        h, m = divmod(m, 60)
        cost_time_dt = f'{int(h)}h:{int(m)}m:{round(s, 2)}s'
        
        print(divider())
        print('start_time:', start_time_dt)
        print('end_time  :', end_time_dt)
        print('cost_time :', cost_time_dt)
  
    def record_time():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def calculate_time():
        return time.time()
    
    def divider(num=30):
        return '='*num
    return wrap

def get_cuda_status():
    import torch; print(torch.cuda.is_available())
    
def get_mps_status():
    import torch; print(torch.mps.is_available())
    

def get_device():
    import torch
    match True:
        case _ if torch.mps.is_available():
            return torch.device('mps')
        case _ if torch.cuda.is_available():
            return torch.device('cuda')
        case _:
            return torch.device('cpu')

    
def get_folder_path(pre_path, folder_name):
    folder_path = f'{pre_path}/{folder_name}'
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_archive_path(pre_path, archive_columns, source, today=None):
    import pandas as pd
    archive_path = f'{pre_path}/archive_{source}_{today}.csv' if today else f'{pre_path}/archive_{source}.csv'
    if not os.path.exists(archive_path):
        df = pd.DataFrame(archive_columns)
        df.to_csv(archive_path, index=False)
    return archive_path

def get_today():
    return datetime.strftime(datetime.now(), '%Y%m%d')

def get_now_time():
    return datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')

def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_check_exist_list(archive_path, iloc=-1):
    import pandas as pd
    check_rows = []
    if os.path.isfile(archive_path):
        df = pd.read_csv(archive_path)
        check_rows = df.iloc[:, iloc].to_list()
    return check_rows

def get_soup(url, cookies=None):
    from bs4 import BeautifulSoup
    import requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    }
    resp = requests.get(url, headers=headers)
    resp_results = resp.text
    return BeautifulSoup(resp_results, 'lxml')

def get_split_file_name_and_ext(file_path):
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    return file_name, file_ext

if __name__ == '__main__':
    print(get_device())
