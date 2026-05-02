with open("src/torchsky/__init__.py", "r") as f:
    content = f.read()

# I need to ensure `handle_cache_capacity` inside `_read_batch_paths` is fully updated!
# Let's inspect `_read_batch_paths` again.

import ast

class MethodVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        if node.name in ['_read_batch_paths', '_read_batch_hdus']:
            print(f"Found {node.name}")

tree = ast.parse(content)
MethodVisitor().visit(tree)
