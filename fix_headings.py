#!/usr/bin/env python3
"""
Fix heading hierarchy in markdown lesson files.
Changes the first ## Title to # Title (lesson title should be h1, not h2)
"""
import os
import re
import json
from pathlib import Path

PROJECT_ROOT = Path('/Users/znding04/Work/learn-some-ai')
CONTENT_DIR = PROJECT_ROOT / 'public' / 'content'

def fix_heading_in_file(filepath):
    """Fix the first ## heading to # heading in a markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Check if file starts with YAML frontmatter
    if content.startswith('---'):
        # Find the end of frontmatter (second ---)
        match = re.search(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if match:
            frontmatter = match.group(1)
            body_start = match.end()
            body = content[body_start:]
            
            # Find first heading in body and fix it
            lines = body.split('\n')
            for i, line in enumerate(lines):
                if re.match(r'^##\s+', line):
                    # This is the first heading - convert ## to #
                    lines[i] = '#' + line[2:]
                    break
            
            content = '---\n' + frontmatter + '\n---\n' + '\n'.join(lines)
    else:
        # No frontmatter, find first heading directly
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^##\s+', line):
                    lines[i] = '#' + line[2:]
                    break
        content = '\n'.join(lines)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    # Load lessons.json to get all content paths
    lessons_file = CONTENT_DIR / 'lessons.json'
    with open(lessons_file, 'r') as f:
        lessons = json.load(f)
    
    fixed_count = 0
    error_count = 0
    
    for lesson_id, lesson_data in lessons.items():
        content_path = lesson_data.get('contentPath', '')
        if not content_path:
            continue
        
        filepath = CONTENT_DIR / content_path
        if not filepath.exists():
            print(f"  MISSING: {filepath}")
            error_count += 1
            continue
        
        try:
            if fix_heading_in_file(filepath):
                print(f"  FIXED: {content_path}")
                fixed_count += 1
        except Exception as e:
            print(f"  ERROR: {content_path} - {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"  Fixed: {fixed_count} files")
    print(f"  Errors: {error_count} files")

if __name__ == '__main__':
    main()
