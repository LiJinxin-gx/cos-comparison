"""
Operator tools: Python's built-in operations in functional (callable) form.

Every language operator (+ - * / // % ** & | ^ ~ << >> == != < <= > >= [],
in, not, len, ...) is exposed as a first-class function with the standard
`operator` module naming (zero learning cost): pass them around, feed them
to core callbacks (data_mapping / data_filter), or compose them inside
ComposalFunction steps.

All operations that the stdlib `operator` module provides are imported
directly from it (the C implementation, stable across Python versions);
only what operator does not offer is written independently:

  * the reflected r* family (rsub, rmul, rmatmul, rtruediv, rfloordiv,
    rmod, rpow, rand, ror, rxor, rlshift, rrshift) - a generic reflected
    function cannot exist in operator because it needs the right
    operand's type, so these follow the data model: r*(a, b) calls
    b.__r*(a) (the `a op b` fallback path; on symmetric operators the
    value coincides with the forward op)
  * pow with the modular form (operator.pow takes only two arguments)
  * len_ (the builtin len in function form)
  * call on Python < 3.11 (the stdlib operator.call is 3.11+)

Not provided: logical `and` / `or` (they cannot short-circuit in function
form) and attrgetter / itemgetter / methodcaller (owned by
func_tool.FuncHelper - single source of truth, same rule as no_done).
"""

from operator import (
    add,
    sub,
    mul,
    matmul,
    truediv,
    floordiv,
    mod,
    iadd,
    isub,
    imul,
    imatmul,
    itruediv,
    ifloordiv,
    imod,
    ipow,
    and_,
    or_,
    xor,
    lshift,
    rshift,
    iand,
    ior,
    ixor,
    ilshift,
    irshift,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    neg,
    pos,
    abs,
    invert,
    not_,
    getitem,
    setitem,
    delitem,
    contains,
    concat,
    countOf,
    indexOf,
    truth,
    length_hint,
)

try:
    from operator import call
except ImportError:  # Python < 3.11
    def call(obj, *args, **kwargs):
        """Call obj(*args, **kwargs) - the stdlib operator.call is 3.11+."""
        return obj(*args, **kwargs)

# ------------------ reflected variants (not in operator) --------------------

def pow(a, b, mod=None):
    """a ** b, or pow(a, b, mod) - operator.pow takes only two arguments."""
    if mod is None:
        return a ** b
    return a.__pow__(b, mod)


def radd(a, b):
    return b.__radd__(a)


def rsub(a, b):
    return b.__rsub__(a)


def rmul(a, b):
    return b.__rmul__(a)


def rmatmul(a, b):
    return b.__rmatmul__(a)


def rtruediv(a, b):
    return b.__rtruediv__(a)


def rfloordiv(a, b):
    return b.__rfloordiv__(a)


def rmod(a, b):
    return b.__rmod__(a)


def rpow(a, b):
    return b.__rpow__(a)


def rand(a, b):
    return b.__rand__(a)


def ror(a, b):
    return b.__ror__(a)


def rxor(a, b):
    return b.__rxor__(a)


def rlshift(a, b):
    return b.__rlshift__(a)


def rrshift(a, b):
    return b.__rrshift__(a)


def len_(a):
    return len(a)


"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    # binary arithmetic: add(a, b) == a + b, truediv == a / b, ...
    "add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "pow",
    # reflected (b.__r*(a), the `a op b` fallback path; not in operator)
    "radd", "rsub", "rmul", "rmatmul", "rtruediv", "rfloordiv", "rmod",
    "rpow",
    # in-place (a.__i*__(b), else plain a op b)
    "iadd", "isub", "imul", "imatmul", "itruediv", "ifloordiv", "imod",
    "ipow",
    # binary bitwise: and_(a, b) == a & b, or_ == a | b, xor == a ^ b
    "and_", "or_", "xor", "lshift", "rshift",
    # reflected bitwise (not in operator)
    "rand", "ror", "rxor", "rlshift", "rrshift",
    # in-place bitwise
    "iand", "ior", "ixor", "ilshift", "irshift",
    # comparisons: eq(a, b) == (a == b), lt == (a < b), ...
    "eq", "ne", "lt", "le", "gt", "ge",
    # unary: neg == -a, pos == +a, invert == ~a, not_ == not a
    "neg", "pos", "abs", "invert", "not_",
    # container / call protocols: getitem(a, i) == a[i],
    # contains(c, x) == (x in c), concat == a + b, call(f, *a) == f(*a)
    "getitem", "setitem", "delitem", "contains", "concat", "countOf",
    "indexOf", "call", "truth", "len_", "length_hint",
)
