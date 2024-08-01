import json

def preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # แปลงข้อมูลตามที่ต้องการ
    return data

train_data = preprocess_data('data/train.json')
