import re
import sys

def compute_average_kernel(log_file, timing_target):
    kernel_times = []
    # 正则表达式匹配 ${timing_target}=<数字>
    pattern = re.compile(timing_target + r'=([\d\.]+)')
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '[MUSA_KERNEL_TIMING]' in line:
                match = pattern.search(line)
                if match:
                    kernel_times.append(float(match.group(1)))
                    
    if not kernel_times:
        print(f"未找到任何 {timing_target} 数据。")
        return

    avg_kernel = sum(kernel_times) / len(kernel_times)
    print(f"成功提取了 {len(kernel_times)} 条数据")
    print(f"{timing_target} 平均值: {avg_kernel:.6f} ms")

if __name__ == "__main__":
    log_path = sys.argv[1]
    timing_target = sys.argv[2]
    compute_average_kernel(log_path, timing_target)