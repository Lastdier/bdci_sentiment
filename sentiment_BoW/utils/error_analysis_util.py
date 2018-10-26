# coding: utf-8
# created by Pengpeng on 2018/7/31
import collections
from utils.path_util import from_project_root

# 统计错误标签分布
def get_the_error_label_distribution(predict_result_file,distribution_file):

    error_label_dict = {}
    with open(predict_result_file,'r',encoding='utf-8') as f,open(distribution_file,'w',encoding='utf-8') as wf:
        for line in f.readlines()[1:]:
            line_list = line.strip().split(',')
            real_label = line_list[0]
            predict_label = line_list[1]

            if real_label not in error_label_dict.keys():
                error_label_dict[real_label] = {}

            if predict_label not in error_label_dict[real_label].keys():
                error_label_dict[real_label][predict_label] = 0

            error_label_dict[real_label][predict_label] += 1

        for i in range(19):
            label_index = str(i + 1)
            if label_index not in error_label_dict.keys():
                wf.write("\n")
            for j in range(19):
                pre_label_index = str(j + 1)
                if pre_label_index not in error_label_dict[label_index].keys():
                    wf.write("0")
                else:
                    wf.write(str(error_label_dict[label_index][pre_label_index]))

                if j < 18:
                    wf.write(",")

            wf.write("\n")
    pass

def main():
    predict_result_file = from_project_root("lstm_model/result/result_predict-1700.csv")
    distribution_file = from_project_root("lstm_model/result/erro_distribution-1700.csv")
    get_the_error_label_distribution(predict_result_file, distribution_file)
    pass


if __name__ == '__main__':
    main()