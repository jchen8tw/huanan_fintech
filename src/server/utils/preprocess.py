import pandas as pd
import numpy as np
import sys
df = pd.read_csv(sys.path[0] + '/data/rawdata/age_in_days.csv')
#df = pd.read_csv('test_train.csv')
df['marital'] = df['marital'].astype("category")
df['career'] = df['career'].astype("category")
df['education'] = df['education'].astype("category")
df['credit_level'] = df['credit_level'].astype("category").cat.rename_categories([0,1,2])
df['residence'] = df['residence'].astype("category")
df['house_age'] = df['house_age'].astype("category").cat.rename_categories([0,1,2,3,4])
df['property'] = df['property'].astype("category")
df['location'] = df['location'].astype("category")
df['proximity_attr'] = df['proximity_attr'].astype("category")
df['building_type'] = df['building_type'].astype("category")
df['usage'] = df['usage'].astype("category")
df['payment_sources'] = df['payment_sources'].astype("category")
df['grace_period'] = df['grace_period'].astype("category").cat.rename_categories([0,1,2])
df['situation'] = df['situation'].astype("category")
df['trans_channel'] = df['trans_channel'].astype("category")
df['trans_type'] = df['trans_type'].astype("category")

max_val = {}

def get_max(column):
    if(max_val.get(column,False) == False):
        if(column == 'credit_level' or column == 'house_age' or column == 'grace_period'):
            max_val[column] = df[column].cat.as_ordered().max()
        else:
            max_val[column] = df[column].max()
        return max_val[column]
    else:
        return max_val[column]

def renormalize(out):
    out[0] = out[0]*df['period'].max()
    out[1] = out[1]*df["interest_rate"].max()
    out[2] = out[2]*df["percent"].max()
    out[3] = out[3]*df["value"].max()
    out[4] = out[4]*df['grace_period'].cat.as_ordered().max()
    return out
def to_numpy_array(data,is_dict=False):
#    print(data.to_numpy())
    data_dict = {}
    if(is_dict == False):
        data_dict = data.to_dict()
    else:
        data_dict = data
#    print(data_dict)
#    input_arr = np.array([])
#    target_arr = np.array([])
    input_arr = []
    target_arr = []
    for column in data_dict:
        if(column == 'credit_level' or column == 'house_age' or column == 'grace_period'):
            if(column == 'grace_period'):
                #normalize
                #target_arr = np.concatenate((target_arr,np.array([data_dict[column]])/df[column].describe().max()))
                target_arr.append(np.array([data_dict[column]])/get_max(column))
            else:
                #normalize
                #input_arr = np.concatenate((input_arr,np.array([data_dict[column]])/df[column].describe().max()))
                input_arr.append(np.array([data_dict[column]])/get_max(column))
#            print(df[column].cat.categories)
#            print(data_dict[column])
        elif(column == 'uuid'):
            continue
        elif(df[column].dtype.name == 'category'):
            arr_size = df[column].cat.categories.size
            new_arr = np.zeros(arr_size)
#            print(list(df[column].cat.categories))
            pos = list(df[column].cat.categories).index(data_dict[column])
#            print(pos)
            new_arr[pos] = 1
#            print(new_arr)
            if(column == 'situation'):
                #target_arr = np.concatenate((target_arr,new_arr))
                target_arr.append(new_arr)
            else:
                #input_arr = np.concatenate((input_arr,new_arr))
                input_arr.append(new_arr)
        else:
            if(column == 'amount' or column == 'percent' or column == 'period' or column == 'balance' or column == 'value' or column == 'interest_rate'):
                #target_arr = np.concatenate((target_arr,np.array([data_dict[column]])/df[column].describe().max()))
                target_arr.append(np.array([data_dict[column]])/get_max(column))
            else:
                #input_arr = np.concatenate((input_arr,np.array([data_dict[column]])/df[column].describe().max()))
                #print(input_arr)
                input_arr.append(np.array([data_dict[column]])/get_max(column))
#    print(input_arr,target_arr)
    input_arr = np.concatenate(input_arr,axis=-1)
    target_arr = np.concatenate(target_arr,axis=-1)
#    print(input_arr,target_arr)
    #input_arr = input_arr.reshape(1,-1)
    #target_arr = target_arr.reshape(1,-1)
#    print(input_arr.shape,target_arr.shape)
    return input_arr,target_arr

    
def get_train_data():
    input_data,target_data = to_numpy_array(df.iloc[0])
#    print(input_data,target_data)
    input_data = [input_data]
    target_data = [target_data]
    for i in range(1,len(df)):
        if i % 1000 == 0:
            print(i)
        input_row,target_row = to_numpy_array(df.iloc[i])
        input_data.append(input_row)
        target_data.append(target_row)
    input_data = np.stack(input_data)
    target_data = np.stack(target_data)
#        print(i)
    return input_data,target_data


if __name__ == '__main__':
    input_data,target_data = get_train_data()
    print(input_data.shape,target_data.shape)