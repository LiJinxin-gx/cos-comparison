"""
Value compiler & executor for the instruction protocols (shell / batch /
app default reader): Python built-in operations in literal form.

Grammar (stdlib-only, fully iterative - no recursion anywhere):

  literals           1, -3, 2.5, True / False / None
  quoted strings     'abc' / "a b"  (content verbatim, no escapes)
  containers         (1, 2) tuple, () empty tuple, (x,) one-tuple
                     [1, 2] list, [] empty list, {1: "a", 2: "b"} dict
                     (Python-style: elements are comma-separated; the old
                     space-separated forms are gone - "()" is the EMPTY
                     tuple and a bare `(a b c)` group is a NESTED CALL)
  nested call        (func arg1 arg2) calls func(arg1, arg2); func is any
                     name resolvable at run time, or an operator symbol
                     (+ - * / // % ** & | ^ ~ << >> == != < <= > >=)
  grouping           (x) one element without comma = x itself
  infix expression   data[0]+1, (a+b)*2, x<=3 and not (y==1) - one word
                     without spaces; built-in operators with precedence,
                     parentheses for sub-expressions, comparisons and the
                     words and / or / not (eager - the operand evaluation
                     is side-effect free, so short-circuiting is not
                     needed)
  data references    data[index] / data<N> resolve at run time
  unpack markers     %expr expands a sequence into arguments, %%expr a
                     mapping into keywords (argument level)
  variable/pointer   &name -> the variable index, *expr dereferences it
                     (shell namespace; infix operands that are plain words
                     resolve against the variable table first and fall
                     back to the string literal)

Values compile into a flat instruction stream executed by a value stack
machine; execution resolves data references and nested calls against the
caller-provided environment (data region, variables, name resolver).

No project modules are imported - only the standard library.
"""

import operator

__all__ = ("ValueCode", "compile_value", "execute_code", "ValueEnv",
           "DataRef", "SYMBOL_FUNCS", "SYMBOL_OPERATORS")

# ---------------------------- operator tables ---------------------------------

_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
}

_COMPARE = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_UNARY = {
    "-": operator.neg,
    "~": operator.invert,
    "not": operator.not_,
}

_LOGICAL = {
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
}

_PRECEDENCE = {
    "or": 4,
    "and": 6,
    "==": 8, "!=": 8, "<": 8, "<=": 8, ">": 8, ">=": 8,
    "|": 10,
    "^": 12,
    "&": 14,
    "<<": 16, ">>": 16,
    "+": 18, "-": 18,
    "*": 20, "/": 20, "//": 20, "%": 20,
    "_unary": 24,          # -, ~, not (right associative)
    "**": 30,
}

_RIGHT = frozenset(("**", "_unary"))
_LOGICAL_OPS = frozenset(_LOGICAL)
_COMPARE_OPS = frozenset(_COMPARE)

# symbol forms usable as a call head inside a nested call group
SYMBOL_OPERATORS = tuple(
    sorted((set(_BINARY) | set(_COMPARE) | set(_UNARY)) - {"not"},
           key=len, reverse=True))

_UNARY_WORDS = frozenset(("not",))

# --------------------------- data reference types -----------------------------


class DataRef:
    """Variable reference to a data index, resolved at run time against the
    executor's data region (``data[index]`` / ``data<N>`` in the text)."""

    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return "DataRef(%r)" % (self.index,)

    def resolve(self, data):
        return data[self.index]


# ------------------------------- value machine --------------------------------

class ValueEnv:
    """Execution environment: the data region, a variable mapping, the
    name resolver for nested calls, and the variable-index table for
    pointer forms."""

    __slots__ = ("data", "vars", "resolve", "var_index")

    def __init__(self, data=None, vars=None, resolve=None, var_index=None):
        self.data = data
        self.vars = vars if vars is not None else {}
        self.resolve = resolve
        self.var_index = var_index if var_index is not None else {}

    def resolve_name(self, name):
        if self.resolve is not None:
            return self.resolve(name)
        raise ValueError("call needs a name resolver: " + name)


class ValueCode:
    """Compiled value: a flat instruction stream plus operands.  kind is
    "plain", "unpack" (``%expr`` - the value is a sequence, expanded into
    positional arguments) or "unpack_map" (``%%expr`` - a mapping,
    expanded into keyword arguments)."""

    __slots__ = ("code", "text", "kind")

    def __init__(self, code, text, kind="plain"):
        self.code = code
        self.text = text
        self.kind = kind

    def __repr__(self):
        return "ValueCode(%r)" % (self.text,)

    def execute(self, env):
        return execute_code(self.code, env)


def compile_value(text):
    """Compile one value word into a ValueCode (iterative, no recursion).
    ``text`` must be a single protocol word (spaces inside quotes are
    fine)."""
    if not isinstance(text, str):
        return ValueCode((("push", text),), repr(text))
    if text.startswith("%%"):
        inner = compile_value(text[2:])
        return ValueCode(inner.code, text, "unpack_map")
    if text.startswith("%") and len(text) > 1:
        inner = compile_value(text[1:])
        return ValueCode(inner.code, text, "unpack")
    kind = _container_kind(text)
    if kind:
        return ValueCode(_compile_container(text, kind), text)
    return ValueCode(_compile_atom(text), text)


# ------------------------------ character scanner -----------------------------

def _container_kind(text):
    if len(text) >= 2 and text[0] in "([{" and text[-1] in ")]}":
        head, tail = text[0], text[-1]
        if (head, tail) in (("(", ")"), ("[", "]"), ("{", "}")):
            return head
    return None


def _compile_container(text, kind):
    """Compile a container/group word by single character scan with an
    explicit frame stack (nested groups become frames - never recursion).
    Every frame carries its full state:
    [kind, items, comma_seen, trailing, first_name, first_start, pending]"""
    closer = {"(": ")", "[": "]", "{": "}"}
    code = []
    frame = [kind, 0, False, False, None, 0, False]
    frames = [frame]
    buf = []
    quote = None
    word_depth = 0             # brackets inside a word (data[0], ...)
    index = 1                  # skip the opening bracket
    length = len(text)
    while index < length:
        ch = text[index]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            index += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            index += 1
            continue
        f = frames[-1]
        fk = f[0]
        if ch in "([{":
            if buf:
                # inside a word: part of the token (e.g. data[0])
                buf.append(ch)
                word_depth += 1
            else:
                frames.append(
                    [ch, 0, False, False, None, len(code), False])
            index += 1
            continue
        if ch in ")]}":
            if word_depth:
                # closing a bracket that opened inside the current word
                buf.append(ch)
                word_depth -= 1
                index += 1
                continue
            if ch != closer[fk]:
                buf.append(ch)
                index += 1
                continue
            if buf:
                if word_depth:
                    raise ValueError("unbalanced literal: " + text)
                _flush_element(code, "".join(buf).strip(), fk, f[6],
                               f[4])
                buf = []
                if fk == "{":
                    if f[6]:
                        f[1] += 1
                else:
                    f[1] += 1
                f[6] = False
            _close_frame(code, frames)
            if not frames:
                return code
            index += 1
            continue
        if fk == "{":
            if ch == ":":
                if not buf:
                    raise ValueError("missing dictionary key")
                _flush_element(code, "".join(buf).strip(), "key", False,
                               None)
                buf = []
                f[6] = True
                index += 1
                continue
            if ch == ",":
                if not buf and not f[6]:
                    if f[1] == 0:
                        raise ValueError("empty dictionary element")
                    f[3] = True
                else:
                    _flush_element(code, "".join(buf).strip(),
                                   "value", False, None)
                    buf = []
                    f[1] += 1
                    f[6] = False
                index += 1
                continue
            if ch in " \t":
                index += 1
                continue
            buf.append(ch)
            index += 1
            continue
        if ch in ", \t":
            if ch == ",":
                f[2] = True
            if buf:
                flush_text = "".join(buf).strip()
                buf = []
                if flush_text:
                    if f[1] == 0 and fk == "(" and f[4] is None:
                        name = _head_name(flush_text)
                        if name is not None:
                            f[4] = name
                            f[5] = len(code)
                    _flush_element(code, flush_text, fk, False, f[4])
                    f[1] += 1
                    if fk == "[" and not f[2] and f[1] > 1:
                        raise ValueError(
                            "list elements must be comma-separated "
                            "(Python-style literals)")
                elif ch == ",":
                    if f[1] == 0:
                        raise ValueError("empty group element")
                    f[3] = True
            index += 1
            continue
        buf.append(ch)
        index += 1
    raise ValueError("unbalanced literal: " + text)


def _close_frame(code, frames):
    """Finish the top frame: emit the closing instruction and merge the
    state back into the parent frame."""
    f = frames[-1]
    fk = f[0]
    items = f[1]
    comma = f[2]
    trailing = f[3]
    first_name = f[4]
    first_start = f[5]
    if fk == "(":
        if comma or trailing or items == 0:
            code.append(("mk", "tuple", items))
        elif items == 1:
            pass                       # grouping: the value stays
        else:
            name = first_name
            if name is None:
                raise ValueError(
                    "invalid group: a multi-element ( ) without commas "
                    "must start with a function name or operator symbol")
            del code[first_start]      # the head word is the name
            code.append(("call", name, items - 1))
    elif fk == "[":
        if items > 1 and not comma and not trailing:
            raise ValueError(
                "list elements must be comma-separated "
                "(Python-style literals)")
        if items == 0 and not comma and not trailing:
            code.append(("mk", "list", 0))
        else:
            code.append(("mk", "list", items))
    else:  # "{"
        if f[6]:
            raise ValueError("missing value after ':' in dictionary")
        if items == 0 and not comma:
            code.append(("mk", "dict", 0))
        else:
            code.append(("mk", "dict", items))
    frames.pop()
    if frames:
        parent = frames[-1]
        parent[1] += 1                  # the group counts as one element
        parent[6] = False


def _compile_atom(text):
    """Top-level single word without a container: same element logic."""
    code = []
    _flush_element(code, text, "top", False, None)
    return code


def _head_name(text):
    """The word of a call head, or None when it is not a plain name /
    symbol word (numbers, quoted strings, groups)."""
    if not text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return None
    try:
        int(text)
        return None
    except ValueError:
        pass
    if text[0].isdigit():
        return None
    return text


def _flush_element(code, text, where, pending_value, first_name):
    """Emit the instruction stream of one element (a value word)."""
    del first_name, pending_value, where
    if text == "":
        return
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        code.append(("push", text[1:-1]))
        return
    if text in ("True", "False", "None"):
        code.append(("push", {"True": True, "False": False,
                              "None": None}[text]))
        return
    ref = _data_ref(text)
    if ref is not None:
        code.append(("push", ref))
        return
    ptr = _pointer_ref(text)
    if ptr is not None:
        code.append(ptr)
        return
    num = _number(text)
    if num is not None:
        code.append(("push", num))
        return
    if _is_expression(text):
        code.extend(_compile_expression(text))
        return
    code.append(("pushn", text))       # plain word: variable or string


def _number(text):
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _data_ref(text):
    if text.startswith("data[") and text.endswith("]") \
            and text[5:-1].isdigit():
        return DataRef(int(text[5:-1]))
    if text.startswith("data") and text[4:].isdigit():
        return DataRef(int(text[4:]))
    return None


def _pointer_ref(text):
    if text.startswith("&") and len(text) > 1:
        return ("ptr", text[1:])
    if text.startswith("*") and len(text) > 1:
        rest = text[1:]
        if rest.isdigit():
            return ("deref", rest, None)
        if rest.startswith("&") and len(rest) > 1:
            return ("deref", rest[1:], "&")
        return ("deref", rest, "v")
    return None


# ---------------------------- infix expressions ------------------------------

def _is_expression(text):
    """Whether a word contains infix operators (or a leading bracket that
    makes it a parenthesised sub-expression at element level).  A word
    made only of operator symbols (==, +, ...) is a callable name, not an
    expression."""
    if not text:
        return False
    if all(ch in "+-*/%&|^~<>!=~" for ch in text):
        return False
    for ch in text:
        if ch in "+-*/%&|^~<>!=(":
            return True
    for word in (" and ", " or ", " not "):
        if word in text:
            return True
    if text.startswith("(") and text.endswith(")"):
        return True
    return False


def _tokenize(text):
    """Split an expression word into tokens (iterative): numbers, quoted
    strings, words (data[i], variables), operator symbols and brackets."""
    tokens = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch in " \t":
            i += 1
            continue
        if ch in "\"'":
            j = i + 1
            while j < length and text[j] != ch:
                j += 1
            if j >= length:
                raise ValueError("unterminated string in expression")
            tokens.append(("str", text[i + 1:j]))
            i = j + 1
            continue
        if ch == "(":
            tokens.append(("(",))
            i += 1
            continue
        if ch == ")":
            tokens.append((")",))
            i += 1
            continue
        if ch.isdigit() or ch == "." and i + 1 < length \
                and text[i + 1].isdigit():
            j = i
            while j < length:
                c = text[j]
                if c.isdigit():
                    j += 1
                    continue
                if c == "." and not (j + 1 < length
                                     and text[j + 1] == "."):
                    j += 1
                    continue
                if c in "eE" and j + 1 < length and (
                        text[j + 1].isdigit()
                        or (text[j + 1] in "+-"
                            and j + 2 < length
                            and text[j + 2].isdigit())):
                    j += 1
                    continue
                break
            tokens.append(("num", text[i:j]))
            i = j
            continue
        matched = None
        for symbol in SYMBOL_OPERATORS:
            if text.startswith(symbol, i):
                matched = symbol
                break
        if matched is not None:
            tokens.append(("op", matched))
            i += len(matched)
            continue
        if text.startswith("not ", i) or text[i:i + 4] in ("not)", "not(") \
                or text.startswith("not", i) and (
                    i + 3 == length
                    or text[i + 3] in " \t(*"):
            tokens.append(("op", "not"))
            i += 3
            continue
        word_matched = False
        for word in ("and", "or"):
            if text.startswith(word, i) and (
                    i + len(word) == length
                    or text[i + len(word)] in " \t"):
                tokens.append(("op", word))
                i += len(word)
                word_matched = True
                break
        if word_matched:
            continue
        j = i
        while j < length and text[j] not in " \t()" \
                and not _at_operator(text, j):
            j += 1
        tokens.append(("word", text[i:j]))
        i = j
    return tokens


def _at_operator(text, i):
    for symbol in SYMBOL_OPERATORS:
        if text.startswith(symbol, i):
            return True
    return False


def _compile_expression(text):
    """Infix expression -> instruction stream (shunting-yard, iterative).
    and / or / not are eager (side-effect-free operands)."""
    tokens = _tokenize(text)
    out = []
    ops = []
    expect_operand = True
    i = 0
    length = len(tokens)
    while i < length:
        token = tokens[i]
        kind = token[0]
        if kind in ("num", "str"):
            value = token[1]
            if kind == "num":
                parsed = _number(value)
                if parsed is None:
                    raise ValueError("bad number in expression: " + value)
                out.append(("push", parsed))
            else:
                out.append(("push", value))
            expect_operand = False
        elif kind == "word":
            text_word = token[1]
            ref = _data_ref(text_word)
            if ref is not None:
                out.append(("push", ref))
            else:
                pointer = _pointer_ref(text_word)
                if pointer is not None:
                    out.append(pointer)
                else:
                    out.append(("pushn", text_word))
            expect_operand = False
        elif kind == "(":
            ops.append("(")
            expect_operand = True
        elif kind == ")":
            while ops and ops[-1] != "(":
                out.append(("op", ops.pop()))
            if not ops:
                raise ValueError("unbalanced ')' in expression")
            ops.pop()
            expect_operand = False
        else:
            symbol = token[1]
            if symbol == "not":
                if not expect_operand:
                    raise ValueError("'not' needs an operand")
                ops.append("_unary_not")
                i += 1
                continue
            if symbol in _LOGICAL_OPS:
                if expect_operand:
                    raise ValueError(
                        "'%s' needs a left operand" % symbol)
                op_token = symbol
            elif symbol in _COMPARE_OPS or symbol in _BINARY:
                if expect_operand:
                    if symbol in _UNARY and symbol not in _COMPARE_OPS:
                        op_token = "_unary"
                        ops.append(op_token)
                        i += 1
                        continue
                    raise ValueError(
                        "operator '%s' needs a left operand" % symbol)
                op_token = symbol
            else:
                raise ValueError("unknown operator: " + symbol)
            precedence = _PRECEDENCE[op_token]
            while ops and ops[-1] != "(":
                top = ops[-1]
                top_prec = _PRECEDENCE.get(top, 0)
                if top in _RIGHT:
                    if top_prec > precedence:
                        out.append(("op", ops.pop()))
                        continue
                else:
                    if top_prec >= precedence:
                        out.append(("op", ops.pop()))
                        continue
                break
            ops.append(op_token)
            expect_operand = True
        i += 1
    while ops:
        if ops[-1] == "(":
            raise ValueError("unbalanced '(' in expression")
        out.append(("op", ops.pop()))
    return _rewrite_unary(out)


def _rewrite_unary(stream):
    """Post-process the RPN stream: resolve unary markers and compound
    ops into op instructions carrying the operator function.  The helper
    stack fully simulates the RPN (operand folding for negated numeric
    literals); every real instruction is kept."""
    rewritten = []
    stack = []
    for ins in stream:
        kind = ins[0]
        if kind == "push":
            rewritten.append(ins)
            stack.append(ins)
        elif kind == "op":
            token = ins[1]
            if token == "_unary":
                if not stack:
                    raise ValueError("missing operand for unary operator")
                top = stack.pop()
                if top[0] == "push":
                    value = top[1]
                    if isinstance(value, (int, float)) \
                            and not isinstance(value, bool):
                        rewritten.pop()
                        rewritten.append(("push", -value))
                        stack.append(("push", -value))
                        continue
                rewritten.append(("op1", _UNARY["-"]))
                stack.append(None)
            elif token == "_unary_not":
                if not stack:
                    raise ValueError("missing operand for 'not'")
                stack.pop()
                rewritten.append(("op1", _UNARY["not"]))
                stack.append(None)
            elif token in _LOGICAL_OPS:
                if len(stack) < 2:
                    raise ValueError("missing operand for '%s'" % token)
                stack.pop()
                stack[-1] = None
                rewritten.append(("op2", _LOGICAL[token]))
            elif token in _COMPARE_OPS:
                if len(stack) < 2:
                    raise ValueError("missing operand for '%s'" % token)
                stack.pop()
                stack[-1] = None
                rewritten.append(("op2", _COMPARE[token]))
            else:
                if len(stack) < 2:
                    raise ValueError("missing operand for '%s'" % token)
                stack.pop()
                stack[-1] = None
                rewritten.append(("op2", _BINARY[token]))
        elif kind in ("pushn", "ptr", "deref"):
            rewritten.append(ins)
            stack.append(ins)
        else:
            raise ValueError("unexpected instruction in expression")
    if len(stack) != 1:
        raise ValueError("invalid expression (operands without operator)")
    return rewritten


# --------------------------------- execution ----------------------------------

def execute_code(code, env):
    """Run a compiled instruction stream on the value stack (iterative).
    Returns the single produced value."""
    stack = []
    for ins in code:
        kind = ins[0]
        if kind == "push":
            value = ins[1]
            stack.append(_resolve_leaf(value, env))
        elif kind == "pushn":
            name = ins[1]
            if env.vars is not None and name in env.vars:
                stack.append(env.vars[name])
            else:
                stack.append(name)         # fall back to the string
        elif kind == "ptr":
            try:
                stack.append(env.var_index[ins[1]])
            except KeyError:
                raise ValueError("unknown variable: " + ins[1]) from None
        elif kind == "deref":
            index = _deref_index(ins, env)
            stack.append(_var_value(index, env))
        elif kind == "op2":
            right = stack.pop()
            left = stack.pop()
            stack.append(ins[1](left, right))
        elif kind == "op1":
            stack.append(ins[1](stack.pop()))
        elif kind == "mk":
            count = ins[2]
            if ins[1] == "dict":
                items = stack[-2 * count:] if count else []
                if count:
                    del stack[-2 * count:]
                stack.append(dict(zip(items[::2], items[1::2])))
            elif count == 0:
                stack.append(() if ins[1] == "tuple" else [])
            else:
                items = stack[-count:]
                del stack[-count:]
                stack.append(tuple(items) if ins[1] == "tuple" else items)
        elif kind == "call":
            name = ins[1]
            count = ins[2]
            args = stack[-count:] if count else []
            if count:
                del stack[-count:]
            stack.append(env.resolve_name(name)(*args))
        else:
            raise ValueError("unknown instruction: " + kind)
    if not stack:
        raise ValueError("empty value")
    return stack[-1]


def _resolve_leaf(value, env):
    if isinstance(value, DataRef):
        if env.data is None:
            raise ValueError("data reference outside a data region")
        return value.resolve(env.data)
    return value


def _deref_index(ins, env):
    form = ins[2]
    if form is None:
        return int(ins[1])
    if form == "&":
        try:
            return env.var_index[ins[1]]
        except KeyError:
            raise ValueError("unknown variable: " + ins[1]) from None
    name = ins[1]
    if env.vars is not None and name in env.vars \
            and isinstance(env.vars[name], int):
        return env.vars[name]
    raise ValueError("invalid dereference: *" + name)


def _var_value(index, env):
    for name, i in (env.var_index or {}).items():
        if i == index:
            return env.vars[name]
    raise ValueError("no variable at index " + str(index))

# symbol -> callable: usable as a function name in calls and instructions
SYMBOL_FUNCS = {}
for _sym, _fn in _BINARY.items():
    SYMBOL_FUNCS[_sym] = _fn
for _sym, _fn in _COMPARE.items():
    SYMBOL_FUNCS[_sym] = _fn
for _sym, _fn in _UNARY.items():
    SYMBOL_FUNCS.setdefault(_sym, _fn)
