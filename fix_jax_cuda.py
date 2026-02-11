#!/usr/bin/env python3
"""
Fix JAX 0.4.35 crash: pathlib.Path(cuda_nvcc.__file__) where __file__ is None.

Run this BEFORE importing jax:
    python fix_jax_cuda.py
    python -c "import jax; print(jax.devices())"
"""
import site
import os
import re
import shutil

# Find jax/_src/lib/__init__.py without importing jax
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    candidate = os.path.join(sp, 'jax', '_src', 'lib', '__init__.py')
    if os.path.isfile(candidate):
        target = candidate
        break
else:
    print("ERROR: Cannot find jax/_src/lib/__init__.py")
    print("Make sure jax is installed: pip show jax")
    exit(1)

print(f"Found: {target}")

with open(target, 'r') as f:
    content = f.read()

# The buggy line:
#     cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__).parent
buggy = "cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__).parent"

if buggy not in content:
    print("Line not found — already patched or different JAX version.")
    print("Testing import...")
    os.execvp('python', ['python', '-c', 'import jax; print("JAX OK:", jax.devices())'])
    exit(0)

# Backup
backup = target + '.bak'
if not os.path.exists(backup):
    shutil.copy2(target, backup)
    print(f"Backup: {backup}")

# Patch: add a None guard before the pathlib call
# Find the exact indentation
match = re.search(r'^(\s+)(cuda_nvcc_path = pathlib\.Path\(cuda_nvcc\.__file__\)\.parent)', 
                  content, re.MULTILINE)
if match:
    indent = match.group(1)
    old_line = match.group(0)
    new_lines = (
        f"{indent}if cuda_nvcc.__file__ is None:\n"
        f"{indent}    return None\n"
        f"{indent}cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__).parent"
    )
    content = content.replace(old_line, new_lines, 1)
    
    with open(target, 'w') as f:
        f.write(content)
    print("PATCHED successfully!")
else:
    print("ERROR: regex match failed. Manual patch needed:")
    print(f"  Open {target}")
    print(f"  Find: {buggy}")
    print(f"  Add before it: if cuda_nvcc.__file__ is None: return None")
    exit(1)

# Test
print("\nTesting JAX import...")
os.execvp('python', ['python', '-c', 
    'import jax; print("JAX devices:", jax.devices()); print("Backend:", jax.default_backend())'])
