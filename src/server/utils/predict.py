import torch
from .model import Net
from .preprocess import to_numpy_array
import pandas as pd
import numpy as np
import sys


net = Net(n_feature=99, n_hidden=50, n_output=8)
net.load_state_dict(torch.load(sys.path[0] + '/data/model/net.pkl'))



def predict(data):
    input_data,target = to_numpy_array(data,is_dict=True)
    input_data = torch.from_numpy(input_data).float()
    out = net(input_data)
    out = out.detach().numpy()
    result = {
        "period": out[0],
        "interest_rate": out[1],
        "percent": out[2],
        "value": out[3],
        "grace_period":  ['0-1年','1-2年','2-3年'][int(round( 2 if out[4] > 2 else out[4] ))],
        "situation": ['無延遲繳款', '逾期30天以上', '逾期90天以上'][np.argmax(out[5:])]
    }
    return result
    

if __name__ == '__main__':
#    data = {'uuid': '000173d87419493abc6eb0fba1e3d1b0', 'age_in_days': 11601, 'income': 1100000, 'marital': '未婚', 'career': '醫療／專業事務所', 'education': '專科/大學', 'credit_level': 1, 'dependents': 0, 'residence': '台南市', 'house_age': 3, 'property': '一般房貸', 'location': '台北市', 'proximity_attr': '住宅區', 'building_type': '公寓(5樓含以下無電梯)', 'amount_mortgage': 9000000, 'period': 12, 'balance_mortgage': 25500000, 'usage': '購置不動產-自用', 'appraisal': 30000000, 'interest_rate': 1.7, 'percent': 85, 'value': 8970000, 'payment_sources': '薪資', 'grace_period': 2, 'situation': '無延遲繳款', 'amount_transaction': 5000.0, 'balance_transaction': 5972193.0, 'trans_channel': '行銀', 'trans_type': '台幣轉帳'}
    data = { 
        "age_in_days": 11601, 
        "income": 1100000, 
        "marital": "未婚", 
        "career": "醫療／專業事務所", 
        "education": "專科/大學", 
        "credit_level": 1, 
        "dependents": 0, 
        "residence": "台南市", 
        "house_age": 3, 
        "property": "一般房貸", 
        "location": "台北市", 
        "proximity_attr": "住宅區", 
        "building_type": "公寓(5樓含以下無電梯)", 
        "amount_mortgage": 9000000 ,
        "balance_mortgage": 25500000, 
        "usage": "購置不動產-自用", 
        "appraisal": 30000000, 
        "value": 8970000, 
        "payment_sources": "薪資", 
        "amount_transaction": 5000.0, 
        "balance_transaction": 5972193.0, 
        "trans_channel": "行銀", 
        "trans_type": "台幣轉帳"}
    print(predict(data))