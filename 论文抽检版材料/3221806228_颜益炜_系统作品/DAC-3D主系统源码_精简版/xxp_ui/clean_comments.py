"""清理debug-BACKUP.py中的注释代码"""
import re

def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # 跳过纯注释行（以#开头，但保留缩进内的有效注释）
        if stripped.startswith('#'):
            i += 1
            continue

        # 删除行尾注释（但保留字符串中的#）
        # 简单处理：如果行中有#，检查是否在字符串外
        if '#' in line:
            # 更安全的方法：只删除明显的行尾注释
            # 保留代码中的#字符（如在字符串、格式化等中）
            # 检测行尾注释模式：空格 + # + 注释内容
            match = re.search(r'\s+#\s+.*$', line)
            if match:
                # 只在确定是行尾注释时才删除
                # 检查#之前的内容是否闭合引号
                before_hash = line[:match.start()]
                single_quotes = before_hash.count("'") - before_hash.count("\\'")
                double_quotes = before_hash.count('"') - before_hash.count('\\"')

                # 如果引号都是成对的，这是一个行尾注释
                if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                    line = before_hash.rstrip() + '\n'

        cleaned_lines.append(line)
        i += 1

    # 删除多余的空行（连续超过2个空行变成2个）
    final_lines = []
    empty_count = 0
    for line in cleaned_lines:
        if line.strip() == '':
            empty_count += 1
            if empty_count <= 2:
                final_lines.append(line)
        else:
            empty_count = 0
            final_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

    print(f"清理完成！")
    print(f"原始行数: {len(lines)}")
    print(f"清理后行数: {len(final_lines)}")
    print(f"删除了 {len(lines) - len(final_lines)} 行")

if __name__ == '__main__':
    input_file = r'D:\zycgit\ZDevelop_Confocal\xxp_ui\window\debug-BACKUP.py'
    output_file = r'D:\zycgit\ZDevelop_Confocal\xxp_ui\window\debug-BACKUP-cleaned.py'
    clean_file(input_file, output_file)
