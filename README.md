# PythonInterpreter

## Description

My own realization of the interpreter of Python 3.6.3, written in Python 3.6.3. It doesn't cover all Python 3 syntax.
Based on many other ideas.

Done as a homework at the Yandex Data Science School in Oct-Nov 2018.

## How to run

To run a python program using this interpreter, simply type it in the file interpreter.py in the section

```if __name__ == '__main__':
    VirtualMachine().run(compile("""
with open('sample_program.py') as f:
    print(f.read())
""", "<stdin>", "exec"))
```

instead of the existing code. Then just run

```$ python3 interpreter.py```
