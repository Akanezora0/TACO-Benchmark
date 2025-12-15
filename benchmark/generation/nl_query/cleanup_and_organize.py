"""
清理和整理NL查询文件，确保每个数据库正好200条
- 保留索引0-199的文件
- 删除索引>=200的文件
- 确保50条重新生成的简单查询在0-199范围内
"""

import json
import os
import shutil
import argparse

def cleanup_database(nl_dir: str, database: str, dry_run: bool = False):
    """清理数据库的NL查询文件"""
    if not os.path.exists(nl_dir):
        print(f"目录不存在: {nl_dir}")
        return
    
    all_files = sorted([f for f in os.listdir(nl_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')],
                      key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    files_to_keep = []
    files_to_remove = []
    regenerated_count = 0
    
    for filename in all_files:
        idx_str = filename.split('_')[-1].split('.')[0]
        if not idx_str.isdigit():
            continue
        
        idx = int(idx_str)
        filepath = os.path.join(nl_dir, filename)
        
        # 检查是否重新生成
        is_regenerated = False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            is_regenerated = data.get('regenerated', False)
        except:
            pass
        
        if idx < 200:
            files_to_keep.append((idx, filename, is_regenerated))
            if is_regenerated:
                regenerated_count += 1
        else:
            files_to_remove.append((idx, filename))
    
    print(f"{database}:")
    print(f"  总文件数: {len(all_files)}")
    print(f"  保留文件数（索引0-199）: {len(files_to_keep)}")
    print(f"  删除文件数（索引>=200）: {len(files_to_remove)}")
    print(f"  重新生成的简单查询: {regenerated_count}")
    
    if files_to_remove:
        print(f"\n  将删除的文件（前10个）:")
        for idx, filename in files_to_remove[:10]:
            print(f"    索引 {idx}: {filename}")
        if len(files_to_remove) > 10:
            print(f"    ... 还有 {len(files_to_remove) - 10} 个文件")
    
    if not dry_run:
        # 删除索引>=200的文件
        removed_count = 0
        for idx, filename in files_to_remove:
            filepath = os.path.join(nl_dir, filename)
            try:
                os.remove(filepath)
                removed_count += 1
            except Exception as e:
                print(f"  删除失败 {filename}: {e}")
        
        print(f"\n  已删除 {removed_count} 个文件")
    
    print()
    return len(files_to_keep), regenerated_count

def main():
    parser = argparse.ArgumentParser(description='清理和整理NL查询文件')
    parser.add_argument('--nl_query_base_dir', type=str, 
                       default='benchmark/data/beijing/output/nl_query',
                       help='NL查询文件基础目录')
    parser.add_argument('--databases', type=str, nargs='+',
                       default=['企业服务', '社会保障', '医疗健康'],
                       help='要处理的数据库列表')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示将要执行的操作，不实际删除文件')
    
    args = parser.parse_args()
    
    print("="*80)
    if args.dry_run:
        print("清理和整理NL查询文件（预览模式）")
    else:
        print("清理和整理NL查询文件")
    print("="*80)
    print()
    
    total_kept = 0
    total_regenerated = 0
    
    for db in args.databases:
        nl_dir = os.path.join(args.nl_query_base_dir, db)
        kept, regenerated = cleanup_database(nl_dir, db, args.dry_run)
        total_kept += kept
        total_regenerated += regenerated
    
    print("="*80)
    print("总结:")
    print(f"  保留的文件总数: {total_kept}")
    print(f"  重新生成的简单查询总数: {total_regenerated}")
    print("="*80)
    
    if args.dry_run:
        print("\n这是预览模式，没有实际删除文件。")
        print("运行时不加 --dry-run 参数来实际执行清理。")

if __name__ == '__main__':
    main()


