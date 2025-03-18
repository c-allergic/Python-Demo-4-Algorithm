from Dataset import Dataset
import numpy as np

def attack(dataset:Dataset,before,after):
    salary_of_new_employee = 0 
    # prior knowledge
    num_emp = len(dataset.data)
    new_num_emp = num_emp + 1
    salary_of_new_employee = new_num_emp * after - num_emp * before 
    print(f"新员工的薪水: {salary_of_new_employee}")
    
def main():
    # 生成数据集
    dataset = Dataset("salary_data.csv")
    dataset.generate_random_data()
    dataset.save_to_csv()
    dataset.load_data()
    return dataset


dataset = main()
original_result = dataset.query()
print(f"Original Result: {original_result}")
# 添加新员工
print("添加新员工")
dataset.add_data({'ID': len(dataset.data), 'salary': 9900})
new_result = dataset.query()
print(f"New Result: {new_result}")


# 隐私攻击
print("Attack before using differential privacy protection")
attack(dataset,original_result,new_result)
print("Attack after using differential privacy protection")
dataset.set_epsilon(.08)
new_result = dataset.query()
attack(dataset,original_result,new_result)

    