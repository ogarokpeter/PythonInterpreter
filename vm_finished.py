import types
import sys
import dis
import imp
import operator


class Function:
    __slots__ = [
        'function_code', 'function_name', 'function_defaults',
        'function_kwdefaults', 'function_globals',
        'function_locals', 'function_closure',
        'function_annotations',
        '__name__', '__dict__', '__doc__', '_vm'
    ]

    def __init__(self, name, code, globs, defaults,
                 kwdefaults, closure, annotations, vm):
        self._vm = vm
        self.function_code = code
        self.function_globals = globs
        self.function_name = self.__name__ = name if name else code.co_name
        self.function_defaults = defaults
        self.function_kwdefaults = kwdefaults
        self.function_closure = closure
        self.function_annotations = annotations
        self.function_locals = self._vm.frame.frame_locals
        self.__dict__ = {}
        self.__doc__ = code.co_consts[0] if code.co_consts else None

    def __get__(self, instance, owner):
        return self

    def __call__(self, *args, **kwargs):
        callargs = {}
        argcount = self.function_code.co_argcount
        argnames = self.function_code.co_varnames
        positional = tuple(argnames[:argcount])
        kwonlyargcount = self.function_code.co_kwonlyargcount
        if len(argnames) > 1 + argcount + kwonlyargcount:
            dsname = argnames[1 + argcount + kwonlyargcount]
        else:
            dsname = None
        if self.function_kwdefaults:
            for key, value in self.function_kwdefaults.items():
                if dsname not in callargs:
                    callargs[dsname] = {}
                callargs[dsname][key] = value
        if self.function_kwdefaults:
            for i in range(0, len(self.function_defaults)):
                callargs[
                         positional[i + len(positional)
                                    - len(self.function_defaults)]
                        ] = self.function_defaults[i]
        for i in range(len(args)):
            if i >= len(positional):
                ssname = argnames[len(positional)]
                if ssname not in callargs:
                    callargs[ssname] = (args[i],)
                else:
                    callargs[ssname] += (args[i],)
            else:
                callargs[positional[i]] = args[i]
        for key, value in kwargs.items():
            if dsname not in callargs:
                callargs[dsname] = {}
            callargs[dsname][key] = value
        frame = self._vm.make_frame(
            self.function_code, callargs, self.function_globals, {}
        )
        return self._vm.run_frame(frame)


class Dispatcher:
    def __init__(self, frame, vm):
        self.frame = frame
        self.vm = vm

    binary_ops = {
        'ADD': operator.add,
        'AND': operator.and_,
        'FLOOR_DIVIDE': operator.floordiv,
        'LSHIFT': operator.lshift,
        'MODULO': operator.mod,
        'MULTIPLY': operator.mul,
        'OR': operator.or_,
        'POWER': pow,
        'RSHIFT': operator.rshift,
        'SUBSCR': operator.getitem,
        'SUBTRACT': operator.sub,
        'TRUE_DIVIDE': operator.truediv,
        'XOR': operator.xor,
        'MATRIX_MULTIPLY': operator.matmul
    }

    unary_ops = {
        'INVERT': operator.invert,
        'NEGATIVE': operator.neg,
        'NOT': operator.not_,
        'POSITIVE': operator.pos,
    }

    compare_ops = {
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
        "is": lambda x, y: x is y,
        "is not": lambda x, y: x is not y,
        "in": lambda x, y: x in y,
        "not in": lambda x, y: x not in y,
    }

    def dispatch(self, instruction):
        if instruction is None:
            return True
        opname = instruction.opname
        argval = instruction.argval
        offset = instruction.offset
        if opname.startswith("BINARY_"):
            operName = opname[7:]
            self.execute_binary_operator(operName)
        elif opname.startswith('UNARY_'):
            operName = opname[6:]
            self.execute_unary_operator(operName)
        elif opname.startswith('INPLACE_'):
            operName = opname[8:]
            self.execute_inplace_operator(operName)
        else:
            bytecode_fn = getattr(self, '_{}'.format(opname), None)
            if not bytecode_fn:
                sys.exit(1)
            if instruction.arg is not None:
                return bytecode_fn(argval)
            else:
                return bytecode_fn()
        return False

    def top(self):
        return self.frame.stack[-1]

    def pop(self, i=0):
        return self.frame.stack.pop(-i - 1)

    def popn(self, n):
        return [self.pop() for a in range(n)][::-1]

    def push(self, *values):
        self.frame.stack.extend(values)

    def lastn(self, n):
        return self.frame.stack[-n]

    def jump(self, tojump):
        self.frame.frame_offsetCur = tojump

    def clear_stack_for_block(self, block):
        while len(self.frame.stack) > block[2]:
            self.pop()

    def secure_blocks(self, endReason):
        block = self.frame.blocks[-1]
        if block[0] == 'loop' and endReason == 'continue':
            self.jump(self.returnValue)
            return None
        self.pop_block()
        self.clear_stack_for_block(block)
        if block[0] == 'loop' and endReason == 'break':
            self.jump(block[1])
            return None
        if block[0] == 'finally':
            if endReason == 'return' or endReason == 'continue':
                sekf.push(self.returnValue)
            self.push(endReason)
            endReason = None
            self.jump(block[1])
        return endReason

    def push_block(self, type, handler=None, level=None):
        if level is None:
            level = len(self.frame.stack)
        self.frame.blocks.append((type, handler, level))

    def pop_block(self):
        return self.frame.blocks.pop()

    def execute_binary_operator(self, op):
        x, y = self.popn(2)
        self.push(self.binary_ops[op](x, y))

    def execute_unary_operator(self, operator):
        x = self.pop()
        self.push(self.unary_ops[operator](x))

    def execute_inplace_operator(self, operator):
        x, y = self.popn(2)
        if operator == 'ADD':
            x += y
        elif operator == 'AND':
            x &= y
        elif operator == 'FLOOR_DIVIDE':
            x //= y
        elif operator == 'LSHIFT':
            x <<= y
        elif operator == 'MODULO':
            x %= y
        elif operator == 'MULTIPLY':
            x *= y
        elif operator == 'OR':
            x |= y
        elif operator == 'POWER':
            x **= y
        elif operator == 'RSHIFT':
            x >>= y
        elif operator == 'SUBTRACT':
            x -= y
        elif operator == 'TRUE_DIVIDE':
            x /= y
        elif operator == 'XOR':
            x ^= y
        elif operator == 'MATRIX_MULTIPLY':
            x @= y
        self.push(x)

    def _call_function(self, flags, args, kwargs):
        kwlen, lenp = divmod(flags, 256)
        namedArguments = {}
        for i in range(kwlen):
            key, value = self.popn(2)
            namedArguments[key] = value
        namedArguments.update(kwargs)
        positionalArguments = self.popn(lenp)
        positionalArguments.extend(args)
        func = self.pop()
        if str(type(func)) == "<class 'builtin_function_or_method'>":
            def wrapped(_unused_posargs, _unused_namedargs,
                        _unused_frame, _unused_func):
                var = None
                val = None
                for var, val in _unused_frame.frame_locals.items():
                    locals()[var] = val
                for var, val in _unused_frame.frame_globs.items():
                    globals()[var] = val
                del var
                del val
                return _unused_func(*_unused_posargs, **_unused_namedargs)

            returnValue = wrapped(positionalArguments,
                                  namedArguments, self.frame, func)
        else:
            returnValue = func(*positionalArguments, **namedArguments)
        self.push(returnValue)

    def _CALL_FUNCTION(self, argval):
        return self._call_function(argval, [], {})

    def _CALL_FUNCTION_EX(self, argval):
        varkw = self.pop() if (argval & 0x1) else {}
        varpos = self.pop()
        return self._call_function(0, varpos, varkw)

    def _CALL_FUNCTION_KW(self, argval):
        kwargnames = self.pop()
        lenkwargs = len(kwargnames)
        kwargs = self.popn(lenkwargs)
        arg = argval - lenkwargs
        return self._call_function(arg, [], dict(zip(kwargnames, kwargs)))

    def _LOAD_BUILD_CLASS(self, arg):
        self.frame.stack.append(builtins.__build_class__)

    def _LOAD_CONST(self, argval):
        self.push(argval)

    def _LOAD_ATTR(self, argval):
        obj = self.pop()
        val = getattr(obj, argval)
        self.push(val)

    def _POP_TOP(self):
        self.pop()

    def _DUP_TOP(self):
        self.push(self.top())

    def _DUP_TOP_TWO(self):
        a, b = self.popn(2)
        self.push(a, b, a, b)

    def _ROT_TWO(self):
        a, b = self.popn(2)
        self.push(b, a)

    def _ROT_THREE(self):
        a, b, c = self.popn(3)
        self.push(c, a, b)

    def _RAISE_VARARGS(self, argval):
        raise self.frame.stack.pop()

    def _LOAD_NAME(self, argval):
        frame = self.frame
        if argval in frame.frame_locals:
            val = frame.frame_locals[argval]
        elif argval in frame.frame_globs:
            val = frame.frame_globs[argval]
        elif argval in frame.frame_builtins:
            val = frame.frame_builtins[argval]
        else:
            raise NameError("name '{}' is not defined".format(argval))
        self.push(val)

    def _STORE_NAME(self, argval):
        self.frame.frame_locals[argval] = self.pop()

    def _STORE_ATTR(self, argval):
        val, obj = self.popn(2)
        setattr(obj, argval, val)

    def _DELETE_NAME(self, argval):
        del self.frame.frame_locals[argval]

    def _DELETE_ATTR(self, argval):
        obj = self.pop()
        delattr(obj, argval)

    def _LOAD_FAST(self, arg):
        if arg in self.frame.frame_locals:
            val = self.frame.frame_locals[arg]
        else:
            raise UnboundLocalError(
                "local variable '{}' referenced before assignment".format(arg)
            )
        self.push(val)

    def _STORE_FAST(self, argval):
        self.frame.frame_locals[argval] = self.pop()

    def _STORE_SUBSCR(self):
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def _DELETE_FAST(self, argval):
        del self.frame.frame_locals[argval]

    def _DELETE_GLOBAL(self, argval):
        del self.frame.frame_globs[argval]

    def _DELETE_SUBSCR(self):
        obj, subscr = self.popn(2)
        del obj[subscr]

    def _STORE_ANNOTATION(self, argval):
        TOS = self.frame.stack.pop()
        self.frame.frame_locals["__annotations__"][argval] = TOS

    def _SETUP_ANNOTATIONS(self, argval=None):
        if "__annotations__" not in self.frame.frame_locals:
            self.frame.frame_locals["__annotations__"] = dict()

    def _LOAD_GLOBAL(self, argval):
        fr = self.frame
        if argval in fr.frame_globs:
            val = fr.frame_globs[argval]
        elif argval in fr.frame_builtins:
            val = fr.frame_builtins[argval]
        else:
            raise NameError("name '{}' is not defined".format(argval))
        self.push(val)

    def _STORE_GLOBAL(self, argval):
        fr = self.frame
        fr.frame_globs[argval] = self.pop()

    def _LOAD_LOCALS(self):
        self.push(self.frame.frame_locals)

    def _RETURN_VALUE(self, argval=None):
        value = self.pop()
        self.frame.returnValue = value

    def _MAKE_FUNCTION(self, argval=0):
        name = self.pop()
        code = self.pop()
        globs = self.frame.frame_globs
        closure = self.pop() if (argval & 0x8) else None
        annot = self.pop() if (argval & 0x4) else None
        kwdefaults = self.pop() if (argval & 0x2) else None
        defaults = self.pop() if (argval & 0x1) else None
        fn = Function(name, code, globs, defaults,
                      kwdefaults, closure, annot, self.vm)
        self.push(fn)

    def _BUILD_LIST(self, argval):
        lst = self.popn(argval)
        self.push(lst)

    def _BUILD_LIST_UNPACK(self, argval):
        lst = []
        for i in range(argval):
            cur = self.frame.stack[i - argval]
            lst.extend(cur)
        for a in range(argval):
            self.pop()
        self.push(lst)

    def _BUILD_SET(self, argval):
        lst = self.popn(argval)
        self.push(set(lst))

    def _BUILD_SET_UNPACK(self, argval):
        st = set()
        for i in range(argval):
            cur = self.frame.stack[i - argval]
            st.update(cur)
        for a in range(argval):
            self.pop()
        self.push(st)

    def _BUILD_TUPLE(self, argval):
        lst = self.popn(argval)
        self.push(tuple(lst))

    def _BUILD_TUPLE_UNPACK(self, argval):
        lst = self.popn(argval)
        self.push(tuple(e for l in lst for e in l))

    def _BUILD_TUPLE_UNPACK_WITH_CALL(self, argval):
        lst = self.popn(argval)
        self.push(tuple(e for l in lst for e in l))

    def _BUILD_CONST_KEY_MAP(self, argval):
        keys = self.pop()
        values = self.popn(argval)
        res = dict(zip(keys, values))
        self.push(res)

    def _BUILD_MAP(self, argval):
        res = {}
        for i in range(0, argval):
            key, val = self.popn(2)
            res[key] = val
        self.push(res)

    def _BUILD_MAP_UNPACK(self, argval):
        dct = {}
        for i in range(argval):
            cur = self.frame.stack[i - argval]
            dct.update(cur)
        for a in range(argval):
            self.pop()
        self.push(dct)

    def _BUILD_MAP_UNPACK_WITH_CALL(self, argval):
        lst = {}
        for i in range(argval):
            cur = self.frame.stack[i - argval]
            lst.update(cur)
        for a in range(argval):
            self.pop()
        self.push(lst)

    def _BUILD_STRING(self, argval):
        s = ""
        for i in range(argval - 1, -1, -1):
            s += self.lastn(i - argval)
        for a in range(argval):
            self.pop()
        self.push(s)

    def _STORE_MAP(self):
        dct, val, key = self.popn(3)
        dct[key] = val
        self.push(dct)

    def _BUILD_SLICE(self, argval):
        if argval == 2:
            x, y = self.popn(2)
            self.push(slice(x, y))
        elif argval == 3:
            x, y, z = self.popn(3)
            self.push(slice(x, y, z))

    def _LIST_APPEND(self, argval):
        val = self.pop()
        lst = self.lastn(argval)
        lst.append(val)

    def _SET_ADD(self, argval):
        val = self.pop()
        st = self.lastn(argval)
        st.add(val)

    def _MAP_ADD(self, argval):
        val, key = self.popn(2)
        dct = self.lastn(argval)
        dct[key] = val

    def _JUMP_FORWARD(self, argval):
        self.jump(argval)

    def _JUMP_ABSOLUTE(self, argval):
        self.jump(argval)

    def _JUMP_IF_TRUE(self, argval):
        val = self.top()
        if val:
            self.jump(argval)

    def _JUMP_IF_FALSE(self, argval):
        val = self.top()
        if not val:
            self.jump(argval)

    def _POP_JUMP_IF_TRUE(self, argval):
        val = self.pop()
        if val:
            self.jump(argval)

    def _POP_JUMP_IF_FALSE(self, argval):
        val = self.pop()
        if not val:
            self.jump(argval)

    def _JUMP_IF_TRUE_OR_POP(self, argval):
        val = self.top()
        if val:
            self.jump(argval)
        else:
            self.pop()

    def _JUMP_IF_FALSE_OR_POP(self, argval):
        val = self.top()
        if not val:
            self.jump(argval)
        else:
            self.pop()

    def _EXTENDED_ARG(self, argval):
        pass

    def _UNPACK_EX(self, argval):
        TOS = self.pop()
        before = argval % 256
        after = argval // 256
        before_result = []
        while before > 0:
            before_result.append(TOS[0])
            before -= 1
            TOS = TOS[1:]
        after_result = []
        while after > 0:
            after_result.append(TOS[-1])
            after -= 1
            TOS = TOS[:-1]
        after_result = after_result[::-1]
        result = before_result + [list(TOS)] + after_result
        for i in result[::-1]:
            self.push(i)

    def _UNPACK_SEQUENCE(self, argval):
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def _COMPARE_OP(self, argval):
        x, y = self.popn(2)
        self.push(self.compare_ops[argval](x, y))

    def _SETUP_LOOP(self, argval):
        self.push_block('loop', argval)

    def _SETUP_EXCEPT(self, argval):
        self.push_block('setup-except', argval)

    def _SETUP_FINALLY(self, argval):
        self.push_block('finally', argval)

    def _SETUP_WITH(self, argval):
        contman = self.pop()
        self.push(contman.__exit__)
        contmanobj = contman.__enter__()
        self.push_block('finally', argval)
        self.push(contmanobj)

    def _END_FINALLY(self):
        obj = self.pop()
        endReason = None
        if obj:
            endReason = obj
            if endReason == 'return' or endReason == 'continue':
                self.returnValue = self.pop()
        return endReason

    def _WITH_CLEANUP_START(self):
        a = self.top()
        if a is None:
            exFunc = self.pop(1)
        else:
            if a == 'return' or a == 'continue':
                exFunc = self.pop(2)
            else:
                exFunc = self.pop(1)
        ans = exFunc(a, None, None)
        self.push(a)
        self.push(ans)

    def _WITH_CLEANUP_FINISH(self):
        self.pop(1)
        self.pop(1)

    def _GET_ITER(self):
        self.push(iter(self.pop()))

    def _FOR_ITER(self, argval):
        it = self.top()
        try:
            self.push(next(it))
        except StopIteration:
            self.pop()
            self.jump(argval)

    def _BREAK_LOOP(self):
        return 'break'

    def _CONTINUE_LOOP(self, argval):
        self.returnValue = argval
        return 'continue'

    def _POP_BLOCK(self):
        self.pop_block()

    def _POP_EXCEPT(self):
        self.pop_block()

    def _IMPORT_NAME(self, argval):
        level, fromlist = self.popn(2)
        frame = self.frame
        self.push(
            __import__(argval, frame.frame_globs,
                       frame.frame_locals, fromlist, level)
        )

    def _IMPORT_STAR(self):
        mod = self.pop()
        if hasattr(mod, "__all__"):
            d = mod.__all__
        else:
            d = dir(mod)
        for attr in d:
            if attr[0] != '_':
                self.frame.frame_locals[attr] = getattr(mod, attr)

    def _IMPORT_FROM(self, argval):
        mod = self.top()
        self.push(getattr(mod, argval))

    def _FORMAT_VALUE(self, argval):
        fmt_function = argval[0]
        fmt_spec = None
        if argval[1]:
            fmt_spec = self.pop()
        value = self.pop()
        if fmt_function:
            value = fmt_function(value)
        if fmt_spec:
            value = format(value, fmt_spec)
        self.push(str(value))


class Frame:
    def __init__(self, frame_code, frame_globs, frame_locals, frame_last):
        self.blocks = []
        self.stack = []
        self.frame_code = frame_code
        self.frame_globs = frame_globs
        self.frame_locals = frame_locals
        self.frame_offsetCur = 0
        self.instructions = dis.Bytecode(self.frame_code)
        self.returnValue = None
        if frame_last:
            self.frame_builtins = frame_last.frame_builtins
        else:
            self.frame_builtins = frame_locals['__builtins__']
            if hasattr(self.frame_builtins, '__dict__'):
                self.frame_builtins = self.frame_builtins.__dict__

    def get_next_instruction(self):
        for instruction in self.instructions:
            if instruction.offset >= self.frame_offsetCur:
                self.frame_offsetCur = instruction.offset + 1
                return instruction


class VirtualMachine():
    def __init__(self):
        self.frames = []
        self.returnValue = None

    @property
    def frame(self):
        return self.frames[-1] if len(self.frames) else None

    def make_frame(self, code, callargs=None,
                   frame_globs=None, frame_locals=None):
        if callargs is None:
            callargs = {}
        if frame_globs is not None:
            frame_globs = frame_globs
            if frame_locals is None:
                frame_locals = frame_globs
        elif self.frames:
            frame_globs = self.frame.frame_globs
            frame_locals = {}
        else:
            frame_globs = frame_locals = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__doc__': None,
                '__package__': None,
            }
        frame_locals.update(callargs)
        frame = Frame(code, frame_globs, frame_locals, self.frame)
        return frame

    def push_frame(self, frame):
        self.frames.append(frame)

    def pop_frame(self):
        self.frames.pop()

    def get_next_instruction(self):
        next_instruction = self.frame.get_next_instruction()
        return next_instruction

    def run_frame(self, frame):
        dispatcher = Dispatcher(frame, self)
        self.push_frame(frame)
        while True:
            instruction = self.get_next_instruction()
            endReason = dispatcher.dispatch(instruction)
            while endReason and frame.blocks:
                endReason = dispatcher.secure_blocks(endReason)
            if endReason:
                break
        returnValue = self.frame.returnValue
        self.pop_frame()
        return returnValue

    def run_code(self, code, frame_globs=None, frame_locals=None):
        frame = self.make_frame(code, frame_globs=frame_globs,
                                frame_locals=frame_locals)
        val = self.run_frame(frame)
        return val

    def run(self, code: types.CodeType) -> None:
        main_mod = imp.new_module('__main__')
        sys.modules['__main__'] = main_mod
        main_mod.__builtins__ = sys.modules['builtins']
        return self.run_code(code, frame_globs=main_mod.__dict__)


if __name__ == '__main__':
    VirtualMachine().run(compile("""
with open('/home/ogarokpeter/Downloads/virtual2.py') as f:
    print(f.read())
""", "<stdin>", "exec"))
