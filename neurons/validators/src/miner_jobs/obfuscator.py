import ast
from ast import Name, FunctionDef
import random
import string
import sys
import os

python_keywords = [
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
    'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import',
    'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
    'print', 'input', 'open', 'close', 'read', 'write', 'seek', 'tell', 'truncate', 'flush', 'readlines', 'writelines',
    'readline', 'write', 'write', 'write', 'write', 'write', 'write', 'write', 'write', 'write', 'write', 'write', 'write',
    'range', 'enumerate', 'zip', 'map', 'filter', 'reduce', 'sorted', 'reversed', 'len', 'min', 'max', 'sum', 'any', 'all',
    'abs', 'round', 'pow', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'exp', 'log', 'log10', 'log2', 'log1p',
    'logb', 'floor', 'ceil', 'trunc', 'fabs', 'factorial', 'gcd', 'lcm', 'isinstance', 'issubclass', 'type', 'isinstance',
    'issubclass', 'type', 'isinstance', 'issubclass', 'type', 'isinstance', 'issubclass', 'type', 'isinstance', 'issubclass',
    'type', 'isinstance', 'issubclass', 'type', 'isinstance', 'issubclass', 'type', 'isinstance', 'issubclass', 'type',
    'random', "main", 'threading', 'getattr', 'super', 'bytes', 'str', 'int', 'float', 'complex', 'list', 'tuple', 'set', 'dict',
    'NVML_ERROR_UNINITIALIZED', 'NVML_ERROR_INVALID_ARGUMENT', 'NVML_ERROR_NOT_SUPPORTED', 'NVML_ERROR_NO_PERMISSION',
    'NVML_ERROR_ALREADY_INITIALIZED', 'NVML_ERROR_NOT_FOUND', 'NVML_ERROR_INSUFFICIENT_SIZE', 'NVML_ERROR_INSUFFICIENT_POWER',
    'NVML_ERROR_DRIVER_NOT_LOADED', 'NVML_ERROR_TIMEOUT', 'NVML_ERROR_IRQ_ISSUE', 'NVML_ERROR_LIBRARY_NOT_FOUND', 'NVML_ERROR_FUNCTION_NOT_FOUND',
    'NVML_ERROR_CORRUPTED_INFOROM', 'NVML_ERROR_GPU_IS_LOST', 'NVML_ERROR_RESET_REQUIRED', 'NVML_ERROR_OPERATING_SYSTEM',
    'NVML_ERROR_LIB_RM_VERSION_MISMATCH', 'NVML_ERROR_MEMORY', 'NVML_ERROR_UNKNOWN',
    'NVML_ERROR_RESET_REQUIRED', 'NVML_ERROR_OPERATING_SYSTEM', 'NVML_ERROR_LIB_RM_VERSION_MISMATCH',
    'NVML_ERROR_MEMORY', 'NVML_ERROR_UNKNOWN', 'Exception', 'wrapper', 'args', 'kwargs', 'wraps', 'sys',
    'c_char_p', 'AttributeError', '_nvmlLib_refcount', 'nvmlLib', 'CDLL', 'os', 'tempfile', 'NVMLError',
    'OSError', 'c_uint', 'byref', 'create_string_buffer', 'c_int',  'setattr', 'subprocess', 'RuntimeError', 'hashlib',
    'iter', 'repr', 'exc', 'e', 're', 'psutil', 'shutil', 'b64encode', 'Fernet', 'json', 'machine_specs', 'hasattr', 'bool'
]


def random_for_block():
    # Create a new for loop node
    new_for = ast.For(
        target=ast.Name(id='i', ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id='range', ctx=ast.Load()),
            args=[ast.Constant(value=random.randint(1, 10))],
            keywords=[]
        ),
        body=[
            ast.Pass()
        ],
        lineno=0,
        col_offset=0,
        orelse=[]
    )
    return new_for


def random_if_generator():
    test_if = ast.If(
        test=ast.Compare(
            left=ast.Call(
                func=ast.Name(id='range', ctx=ast.Load()),
                args=[ast.Num(n=random.randint(1, 1000))],
                keywords=[]
            ),
            ops=[ast.Eq()],
            comparators=[ast.Num(n=0)]
        ),
        body=[
            ast.Pass()
        ],
        orelse=[
            ast.Pass()
        ]
    )

    return test_if


def random_code_block():
    random_int = random.randint(1, 100)
    if random_int % 3 == 0:
        return random_for_block()
    elif random_int % 3 == 1:
        return random_if_generator()
    else:
        return None


def generate_random_name():
    length = random.randint(5, 15)
    return '_' + ''.join(random.choices(string.ascii_letters, k=length))


class FunctionContentObfuscator(ast.NodeTransformer):
    def __init__(self, args_mapping: dict):
        self.name_mapping = {**args_mapping}

    def visit_Name(self, node: Name):
        if not node.id.startswith('__') and node.id not in python_keywords:
            if node.id not in self.name_mapping:
                self.name_mapping[node.id] = generate_random_name()
            node.id = self.name_mapping[node.id]

        return node


class CodeObfuscator(ast.NodeTransformer):
    def __init__(self):
        self.name_mapping = {}

    def visit_Name(self, node: Name):
        # Only rename if it's not a built-in or special name

        prev_id = node.id

        if not node.id.startswith('__'):
            if node.id not in self.name_mapping:
                self.name_mapping[node.id] = generate_random_name()
            node.id = self.name_mapping[node.id]

        return node


class ModuleObfuscator(ast.NodeTransformer):
    def __init__(self):
        self.func_name_mapping = {}

    def handle_class_def(self, node: ast.ClassDef):
        if node.name not in python_keywords:
            self.func_name_mapping[node.name] = generate_random_name()
            node.name = self.func_name_mapping[node.name]

        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in self.func_name_mapping:
                base.id = self.func_name_mapping[base.id]

        return node

    def handle_assign(self, node: ast.Assign):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id not in python_keywords:
            self.func_name_mapping[node.targets[0].id] = generate_random_name()
            node.targets[0].id = self.func_name_mapping[node.targets[0].id]
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id in self.func_name_mapping:
                node.value.func.id = self.func_name_mapping[node.value.func.id]
            else:
                print(f"Skipping assign", ast.unparse(node), node)

            for arg in node.value.args:
                if isinstance(arg, ast.Name) and arg.id in self.func_name_mapping:
                    arg.id = self.func_name_mapping[arg.id]
        elif isinstance(node.value, ast.Name) and node.value.id in self.func_name_mapping:
            node.value.id = self.func_name_mapping[node.value.id]
        else:
            print(f"Skipping assign - 2", ast.unparse(node), node)

    def visit_Module(self, node):
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name not in python_keywords:
                self.func_name_mapping[stmt.name] = generate_random_name()
                stmt.name = self.func_name_mapping[stmt.name]
            elif isinstance(stmt, ast.Assign):
                self.handle_assign(stmt)
            elif isinstance(stmt, ast.ClassDef):
                self.handle_class_def(stmt)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if func.id in self.func_name_mapping:
                    func.id = self.func_name_mapping[func.id]

                for arg in stmt.value.args:
                    if isinstance(arg, ast.Name) and arg.id in self.func_name_mapping:
                        arg.id = self.func_name_mapping[arg.id]

        random_block = random_code_block()
        if random_block:
            node.body.append(random_block)

        return node


class FunctionObfuscator(ast.NodeTransformer):
    def __init__(self, func_name_mapping: dict):
        self.func_name_mapping = func_name_mapping

    def visit_FunctionDef(self, node: FunctionDef):
        # Find usages of arguments inside the code content
        args_mapping = {}
        for arg in node.args.args:
            args_mapping[arg.arg] = generate_random_name()
            arg.arg = args_mapping[arg.arg]

        # generating some random code blocks
        for i in range(len(node.body)):
            random_block = random_code_block()
            if random_block:
                node.body.insert(i + 1, random_block)

        # Apply transformations to the new module
        obfuscator = FunctionContentObfuscator({**self.func_name_mapping, **args_mapping})
        return obfuscator.visit(node)


def obfuscate_code(source_code):
    # Parse the source code into an AST
    tree = ast.parse(source_code)

    module_obfuscator = ModuleObfuscator()
    tree = module_obfuscator.visit(tree)

    statement_reorderer = FunctionObfuscator(module_obfuscator.func_name_mapping)
    tree = statement_reorderer.visit(tree)

    # Generate new source code
    return ast.unparse(tree)


# Usage example
if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "r") as f:
        source = f.read()

    obfuscated = obfuscate_code(source)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_directory, "obfuscated_machine_scrape.py")

    with open(output_file_path, "w") as f:
        f.write(obfuscated)
