import pandas as pd
import fire
import numpy as np


SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
TRAIN_DISTRIBUTION = np.array([1633,1074,1301,1181,1378,3453,668,735,534,605])
def main(result_path):
    result = pd.read_csv(result_path)
    subject = result['subject']
    content_id = result['content_id']
    count = len(subject)
    ids = set()
    for i in content_id:
        ids.add(i)
    print('There are %d subjects in %d samples' % (count, len(ids)))
    print('average topic number: %.2f'%(count *1.0 / len(ids)))
    print('Trainning set average: %.2f' % (TRAIN_DISTRIBUTION.sum()*1.0/10654))
    # print(5594./5117)
    sub_list = [0] * 10
    for i in subject:
        sub_list[SUBJECT_MASK[i]] += 1
    sub_list = np.array(sub_list)
    sub_dis = sub_list * 1.0 / sub_list.sum()
    train_dis = TRAIN_DISTRIBUTION  * 1.0 / TRAIN_DISTRIBUTION.sum() 
    diss = sub_dis - train_dis
    print('主题', '个数', '与训练集分布相比%')
    for i in range(10):
        print(SUBJECT_LIST[i], sub_list[i], '%.2f'%(diss[i]*100))
    
# 价格 1633
# 配置 1074
# 操控 1301
# 舒适性 1181
# 油耗 1378
# 动力 3453
# 内饰 668
# 安全性 735
# 空间 534
# 外观 605


if __name__ == '__main__':
    fire.Fire()
