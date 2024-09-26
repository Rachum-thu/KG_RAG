import pandas as pd
import json
import numpy as np
from scipy import stats

# 模式名称
MODE_NAMES = {
    0: "KG-RAG",
    1: "KG-RAG + JSON",
    2: "KG-RAG + rules",
    3: "KG-RAG + rules + JSON",
}

# 文件路径
gemini_paths = [f"data/my_results/full-data/gemini_1.5_flash_kg_rag_based_mcq_{mode}.csv" for mode in range(0, 4)]
gpt_paths = [f"data/my_results/full-data/gpt_4o_mini_kg_rag_based_mcq_{mode}.csv" for mode in range(0, 4)]

# 加载CSV文件到DataFrame
gemini_dfs = [pd.read_csv(path) for path in gemini_paths]
gpt_dfs = [pd.read_csv(path) for path in gpt_paths]

# 定义函数检查是否包含正确答案
def contains_correct_answer(row):
    try:
        # 清理并解析'LLM_answer'中的JSON字符串
        cleaned_answer = (
            row['llm_answer']
            .replace('```', '')
            .replace('\n', '')
            .replace('json', '')
            .replace('{{', '{')
            .replace('}}', '}')
        )
        # 提取第一个完整的JSON对象
        json_part = cleaned_answer.split('}')[0] + '}'
        answer_json = json.loads(json_part)
        # 检查解析后的'answer'是否与'correct_answer'匹配
        return row['correct_answer'] == answer_json['answer']
    except (json.JSONDecodeError, KeyError, IndexError):
        # 如果解析失败或缺少键，返回False
        return False

# 应用函数到所有DataFrame以创建'is_correct'列
for df in gemini_dfs + gpt_dfs:
    df['is_correct'] = df.apply(contains_correct_answer, axis=1)

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 假设所有DataFrame具有相同的索引，选择300条随机数据
reference_df = gemini_dfs[0]
selected_indices = np.random.choice(reference_df.index, 300, replace=False)

# 定义模型和模式
models = {
    'Gemini': gemini_dfs,
    'GPT': gpt_dfs
}

# 初始化准确率结果的嵌套字典
accuracy_results = {
    'Gemini': {mode: [] for mode in MODE_NAMES.values()},
    'GPT': {mode: [] for mode in MODE_NAMES.values()}
}

# 进行200次重采样
iterations = 200
sample_size = 100

for i in range(iterations):
    # 从300条数据中随机抽取100条（不放回）
    sample_indices = np.random.choice(selected_indices, sample_size, replace=False)
    
    # 为配对检验，确保所有模型和模式使用相同的样本
    for model_name, dfs in models.items():
        for mode_idx, mode_name in MODE_NAMES.items():
            df = dfs[mode_idx]
            # 选择当前样本并计算准确率
            sample = df.loc[sample_indices]
            accuracy = sample['is_correct'].mean()
            accuracy_results[model_name][mode_name].append(accuracy)

# 定义函数进行配对统计检验
def compute_p_value(acc_base, acc_compare, method='t-test'):
    if method == 't-test':
        stat, p_value = stats.ttest_rel(acc_base, acc_compare)
    elif method == 'wilcoxon':
        # Wilcoxon检验要求数据对不全相同
        # 如果所有差异为0，会导致检验失败
        # 因此需要检查差异是否全部为0
        differences = np.array(acc_compare) - np.array(acc_base)
        if np.all(differences == 0):
            p_value = 1.0  # 没有差异
        else:
            stat, p_value = stats.wilcoxon(acc_base, acc_compare)
    else:
        raise ValueError("Unsupported method. Use 't-test' or 'wilcoxon'.")
    return p_value

# 初始化结果列表
results = []

# 对每个模型进行比较
for model_name in models.keys():
    for mode_idx, mode_name in MODE_NAMES.items():
        if mode_idx == 0:
            continue  # 跳过模式0
        # 模式0的准确率作为基准
        acc_base = accuracy_results[model_name][MODE_NAMES[0]]
        # 当前模式的准确率
        acc_compare = accuracy_results[model_name][mode_name]
        
        # 计算p-value（选择合适的检验方法）
        p_val_ttest = compute_p_value(acc_base, acc_compare, method='t-test')
        p_val_wilcoxon = compute_p_value(acc_base, acc_compare, method='wilcoxon')
        
        # 计算均值和标准差
        base_mean = np.mean(acc_base)
        base_std = np.std(acc_base)
        compare_mean = np.mean(acc_compare)
        compare_std = np.std(acc_compare)
        
        # 计算准确率提升
        mean_diff = compare_mean - base_mean
        
        # 添加到结果列表
        results.append({
            'Model': model_name,
            'Comparison': f"{mode_name} vs {MODE_NAMES[0]}",
            'Base Mean Accuracy': base_mean,
            'Base Std Dev': base_std,
            'Compare Mean Accuracy': compare_mean,
            'Compare Std Dev': compare_std,
            'Mean Difference': mean_diff,
            't-test p-value': p_val_ttest,
            'Wilcoxon p-value': p_val_wilcoxon
        })

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 手动进行Bonferroni校正
number_of_tests = len(results_df)
results_df['t-test p-value Corrected'] = results_df['t-test p-value'] * number_of_tests
results_df['t-test p-value Corrected'] = results_df['t-test p-value Corrected'].apply(lambda x: min(x, 1.0))
results_df['Wilcoxon p-value Corrected'] = results_df['Wilcoxon p-value'] * number_of_tests
results_df['Wilcoxon p-value Corrected'] = results_df['Wilcoxon p-value Corrected'].apply(lambda x: min(x, 1.0))

# 添加是否拒绝原假设的列
results_df['Reject H0 (t-test)'] = results_df['t-test p-value Corrected'] < 0.05
results_df['Reject H0 (Wilcoxon)'] = results_df['Wilcoxon p-value Corrected'] < 0.05

# 定义显示格式
header = (
    f"{'Model':<10}{'Comparison':<35}"
    f"{'Base Mean':<12}{'Base SD':<10}"
    f"{'Compare Mean':<15}{'Compare SD':<12}"
    f"{'Mean Diff':<12}{'t-test p-value':<20}"
    f"{'t-test p-val Corr':<20}{'Reject H0 (t)':<15}"
    f"{'Wilcoxon p-value':<20}{'Wilcoxon p-val Corr':<20}{'Reject H0 (W)':<15}"
)
print(header)
print("-" * len(header))

# 打印每一行结果
for _, row in results_df.iterrows():
    print(
        f"{row['Model']:<10}{row['Comparison']:<35}"
        f"{row['Base Mean Accuracy']:<12.4f}{row['Base Std Dev']:<10.4f}"
        f"{row['Compare Mean Accuracy']:<15.4f}{row['Compare Std Dev']:<12.4f}"
        f"{row['Mean Difference']:<12.4f}{row['t-test p-value']:<20.10f}"
        f"{row['t-test p-value Corrected']:<20.10f}{str(row['Reject H0 (t-test)']):<15}"
        f"{row['Wilcoxon p-value']:<20.10f}{row['Wilcoxon p-value Corrected']:<20.10f}"
        f"{str(row['Reject H0 (Wilcoxon)']):<15}"
    )
