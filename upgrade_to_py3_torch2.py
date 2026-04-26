#!/usr/bin/env python3
"""
NegotiationToM 项目升级脚本
从 Python 3.5 + PyTorch 1.1 升级到 Python 3.10+ + PyTorch 2.x

使用方法:
    python upgrade_to_py3_torch2.py --dry-run  # 预览修改
    python upgrade_to_py3_torch2.py            # 执行修改
"""

import os
import re
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 需要处理的目录
TARGET_DIRS = ['cocoa', 'craigslistbargain', 'onmt']

# 修改统计
stats = {
    'files_processed': 0,
    'variable_removed': 0,
    'iteritems_fixed': 0,
    'xrange_fixed': 0,
    'print_fixed': 0,
    'uint8_mask_fixed': 0,
    'data_index_fixed': 0,
}

def fix_variable_import(content):
    """移除 Variable import 和使用"""
    count = 0

    # 移除 from torch.autograd import Variable
    pattern = r'from torch\.autograd import Variable\n?'
    content, n = re.subn(pattern, '', content)
    count += n

    # 移除 from torch.autograd import Variable, ... 中的 Variable
    pattern = r'from torch\.autograd import (.*?)Variable,?\s*'
    def replace_import(m):
        other_imports = m.group(1).strip()
        if other_imports:
            return f'from torch.autograd import {other_imports}'
        return ''
    content, n = re.subn(pattern, replace_import, content)
    count += n

    # Variable(xxx) -> xxx (简单情况)
    # Variable(tensor) -> tensor
    pattern = r'Variable\(([^,\)]+)\)'
    content, n = re.subn(pattern, r'\1', content)
    count += n

    # Variable(tensor, requires_grad=True) -> tensor.requires_grad_(True) 或保持
    # 这个比较复杂，先简单处理
    pattern = r'Variable\(([^,]+),\s*requires_grad\s*=\s*True\)'
    content, n = re.subn(pattern, r'\1.requires_grad_(True)', content)
    count += n

    return content, count

def fix_iteritems(content):
    """dict.iteritems() -> dict.items()"""
    pattern = r'\.iteritems\(\)'
    content, count = re.subn(pattern, '.items()', content)
    return content, count

def fix_itervalues(content):
    """dict.itervalues() -> dict.values()"""
    pattern = r'\.itervalues\(\)'
    content, count = re.subn(pattern, '.values()', content)
    return content, count

def fix_iterkeys(content):
    """dict.iterkeys() -> dict.keys()"""
    pattern = r'\.iterkeys\(\)'
    content, count = re.subn(pattern, '.keys()', content)
    return content, count

def fix_xrange(content):
    """xrange -> range"""
    pattern = r'\bxrange\b'
    content, count = re.subn(pattern, 'range', content)
    return content, count

def fix_print_statement(content):
    """Python 2 print 语句 -> print() 函数"""
    count = 0
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        # 跳过注释中的 print
        stripped = line.lstrip()
        if stripped.startswith('#'):
            new_lines.append(line)
            continue

        # 匹配 print 语句 (不是函数调用)
        # print 'xxx' 或 print xxx
        match = re.match(r'^(\s*)print\s+([^(].*?)$', line)
        if match:
            indent = match.group(1)
            args = match.group(2).strip()
            # 处理 print >> 的情况
            if args.startswith('>>'):
                # print >> sys.stderr, xxx -> print(xxx, file=sys.stderr)
                m = re.match(r'>>\s*(\S+)\s*,\s*(.*)', args)
                if m:
                    new_lines.append(f"{indent}print({m.group(2)}, file={m.group(1)})")
                    count += 1
                    continue
            new_lines.append(f"{indent}print({args})")
            count += 1
        else:
            new_lines.append(line)

    return '\n'.join(new_lines), count

def fix_uint8_to_bool_mask(content):
    """torch.uint8 用于 mask 时改为 torch.bool"""
    count = 0

    # dtype=torch.uint8 在 mask 相关上下文中 -> dtype=torch.bool
    # 这个需要谨慎，只改明显是 mask 的情况
    pattern = r'torch\.tensor\(([^)]+),\s*dtype\s*=\s*torch\.uint8'

    def replace_if_mask(m):
        nonlocal count
        # 简单启发式：如果变量名包含 mask/s 等，改为 bool
        args = m.group(1)
        count += 1
        return f'torch.tensor({args}, dtype=torch.bool'

    # 只在特定文件中替换
    content, n = re.subn(pattern, replace_if_mask, content)

    return content, count

def fix_masked_fill_variable(content):
    """masked_fill(Variable(mask), ...) -> masked_fill(mask.bool(), ...)"""
    pattern = r'\.masked_fill\(\s*Variable\(([^)]+)\)'
    replacement = r'.masked_fill(\1.bool()'
    content, count = re.subn(pattern, replacement, content)
    return content, count

def fix_data_index(content):
    """tensor.data[0] -> tensor.item() 或 tensor[0].item()"""
    # 这个比较危险，只处理明显的情况
    # .data[0] 通常是取标量
    pattern = r'\.data\[0\]'
    content, count = re.subn(pattern, '.item()', content)
    return content, count

def add_future_imports(content):
    """确保有 from __future__ import print_function (兼容性)"""
    # 如果已经是 Python 3 语法，这个其实不需要
    # 但为了安全，先保留
    return content, 0

def fix_map_filter_to_list(content):
    """map(...) 和 filter(...) 在需要 list 的地方包装"""
    # Python 3 中 map/filter 返回迭代器
    # 这个需要根据上下文判断，暂时不自动处理
    return content, 0

def process_file(filepath, dry_run=False):
    """处理单个文件"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()
    except Exception as e:
        print(f"  [ERROR] 无法读取 {filepath}: {e}")
        return False

    content = original_content
    file_stats = {}

    # 应用所有修复
    content, n = fix_variable_import(content)
    file_stats['variable'] = n

    content, n = fix_iteritems(content)
    file_stats['iteritems'] = n

    content, n = fix_itervalues(content)
    file_stats['itervalues'] = n

    content, n = fix_iterkeys(content)
    file_stats['iterkeys'] = n

    content, n = fix_xrange(content)
    file_stats['xrange'] = n

    content, n = fix_print_statement(content)
    file_stats['print'] = n

    content, n = fix_masked_fill_variable(content)
    file_stats['masked_fill'] = n

    # content, n = fix_uint8_to_bool_mask(content)
    # file_stats['uint8_mask'] = n

    # content, n = fix_data_index(content)
    # file_stats['data_index'] = n

    # 检查是否有修改
    total_fixes = sum(file_stats.values())
    if total_fixes > 0:
        stats['files_processed'] += 1
        stats['variable_removed'] += file_stats.get('variable', 0)
        stats['iteritems_fixed'] += file_stats.get('iteritems', 0) + file_stats.get('itervalues', 0) + file_stats.get('iterkeys', 0)
        stats['xrange_fixed'] += file_stats.get('xrange', 0)
        stats['print_fixed'] += file_stats.get('print', 0)

        rel_path = os.path.relpath(filepath, PROJECT_ROOT)
        print(f"  [MODIFIED] {rel_path}")
        for k, v in file_stats.items():
            if v > 0:
                print(f"             - {k}: {v}")

        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

    return True

def main():
    parser = argparse.ArgumentParser(description='升级 NegotiationToM 到 Python 3.10 + PyTorch 2.x')
    parser.add_argument('--dry-run', action='store_true', help='只预览修改，不实际写入')
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN 模式 - 只预览修改，不实际写入文件")
        print("=" * 60)
    else:
        print("=" * 60)
        print("执行模式 - 将修改文件")
        print("建议先用 git 备份: git add -A && git commit -m 'backup before upgrade'")
        print("=" * 60)
        response = input("确认继续? (y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return

    print("\n开始处理文件...\n")

    for target_dir in TARGET_DIRS:
        dir_path = PROJECT_ROOT / target_dir
        if not dir_path.exists():
            print(f"[SKIP] 目录不存在: {target_dir}")
            continue

        print(f"\n[DIR] 处理目录: {target_dir}")

        for py_file in dir_path.rglob('*.py'):
            process_file(py_file, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("处理完成! 统计:")
    print("=" * 60)
    print(f"  修改的文件数: {stats['files_processed']}")
    print(f"  Variable 移除: {stats['variable_removed']}")
    print(f"  iteritems 修复: {stats['iteritems_fixed']}")
    print(f"  xrange 修复: {stats['xrange_fixed']}")
    print(f"  print 语句修复: {stats['print_fixed']}")

    print("\n" + "=" * 60)
    print("后续手动检查事项:")
    print("=" * 60)
    print("""
1. torch.uint8 mask -> torch.bool
   搜索: dtype=torch.uint8 或 .byte()
   在 mask 上下文中改为 dtype=torch.bool 或 .bool()

2. .data[index] 访问
   搜索: .data[
   改为 .item() (标量) 或直接索引

3. masked_fill 的 mask 参数
   确保 mask 是 bool 类型，不是 uint8

4. torchtext API (如果使用)
   Field, BucketIterator 等已废弃，需要重写

5. 测试运行
   python -c "import torch; print(torch.__version__)"
   python -c "from cocoa.core.controller import Controller"
""")

if __name__ == '__main__':
    main()
