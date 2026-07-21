# Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=missing-function-docstring,invalid-name

import contextlib
import itertools
import os
import pickle
import platform
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import weakref
from collections import OrderedDict, UserList, defaultdict, deque

import pytest

import helpers
import optree
from helpers import (
    GLOBAL_NAMESPACE,
    NAMESPACED_TREE,
    PYPY,
    STANDARD_DICT_TYPES,
    TEST_ROOT,
    TREE_STRINGS,
    TREES,
    MyAnotherDict,
    MyDict,
    Py_DEBUG,
    check_script_in_subprocess,
    disable_systrace,
    gc_collect,
    parametrize,
    recursionlimit,
    skipif_android,
    skipif_ios,
    skipif_pypy,
    skipif_wasm,
)


@pytest.mark.skipif(
    platform.machine().lower() not in ('x86_64', 'amd64'),
    reason='Only run on x86_64 and AMD64 architectures',
)
@skipif_wasm
@skipif_android
@skipif_ios
@skipif_pypy
@disable_systrace
def test_treespec_construct():
    with pytest.raises(TypeError, match=re.escape('No constructor defined!')):
        optree.PyTreeSpec()
    treespec = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
    with pytest.raises(TypeError, match=re.escape('No constructor defined!')):
        treespec.__init__()
    del treespec

    gc_collect()

    returncode = 0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            check_script_in_subprocess(
                r"""
                import signal
                import sys

                import optree
                import optree._C

                for _ in range(32):
                    treespec = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
                    try:
                        repr(treespec)
                    except optree._C.InternalError as ex:
                        assert 'src/treespec/serialization.cpp' in str(ex).replace('\\', '/')
                        sys.exit(0)
                """,
                cwd=tmpdir,
                output=None,
            )
    except subprocess.CalledProcessError as ex:
        returncode = abs(ex.returncode)
        if 128 < returncode < 256:
            returncode -= 128
    assert returncode in (
        0,
        signal.SIGSEGV,
        signal.SIGABRT,
        0xC0000005,  # STATUS_ACCESS_VIOLATION on Windows
    )


def test_treespec_equal_hash():
    for i, tree1 in enumerate(TREES):
        treespec1 = optree.tree_structure(tree1)
        treespec1_none_is_leaf = optree.tree_structure(tree1, none_is_leaf=True)
        assert treespec1 != treespec1_none_is_leaf
        assert hash(treespec1) != hash(treespec1_none_is_leaf)
        for j, tree2 in enumerate(TREES):
            treespec2 = optree.tree_structure(tree2)
            treespec2_none_is_leaf = optree.tree_structure(tree2, none_is_leaf=True)
            if i == j:
                assert treespec1 == treespec2
                assert treespec1_none_is_leaf == treespec2_none_is_leaf
            if treespec1 == treespec2:
                assert hash(treespec1) == hash(treespec2)
            else:
                assert hash(treespec1) != hash(treespec2)
            if treespec1_none_is_leaf == treespec2_none_is_leaf:
                assert hash(treespec1_none_is_leaf) == hash(treespec2_none_is_leaf)
            else:
                assert hash(treespec1_none_is_leaf) != hash(treespec2_none_is_leaf)
            assert hash(treespec1) != hash(treespec2_none_is_leaf)
            assert hash(treespec1_none_is_leaf) != hash(treespec2)


def test_treespec_equal_hash_with_namespace():
    # `optree.functools.partial` is registered in the global namespace, so it is recognized
    # under any namespace. Flattening the same object with and without an explicit namespace
    # yields structurally identical treespecs that compare equal, because an empty namespace is
    # treated as a wildcard compatible with any namespace (see `PyTreeSpec::EqualTo`). Equal
    # treespecs MUST hash equally, otherwise hash-based containers (`dict` / `set`) break.
    obj = optree.functools.partial(int, base=2)

    treespec_no_namespace = optree.tree_structure(obj)
    treespec_namespace = optree.tree_structure(obj, namespace='namespace')

    assert treespec_no_namespace.namespace == ''
    assert treespec_namespace.namespace == 'namespace'

    # The empty namespace is a wildcard compatible with any namespace: these compare equal.
    assert treespec_no_namespace == treespec_namespace

    # Hash/equality contract: equal objects must have equal hashes.
    assert hash(treespec_no_namespace) == hash(treespec_namespace)

    # Consequences for hash-based containers when the contract is honored.
    assert treespec_namespace in {treespec_no_namespace: 'value'}
    assert len({treespec_no_namespace, treespec_namespace}) == 1


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_rich_compare(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        count = itertools.count()

        def build_subtree(x):
            cnt = next(count)
            if cnt % 4 == 0:
                return (x,)
            if cnt % 4 == 1:
                return [x, x]
            if cnt % 4 == 2:
                return (x, [x])
            return {'a': x, 'b': [x]}

        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        suffix_treespec = optree.tree_structure(
            optree.tree_map(build_subtree, tree, none_is_leaf=none_is_leaf, namespace=namespace),
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert treespec == treespec
        assert not (treespec != treespec)
        assert not (treespec < treespec)
        assert not (treespec > treespec)
        assert treespec <= treespec
        assert treespec >= treespec
        assert optree.treespec_is_prefix(treespec, treespec, strict=False)
        assert not optree.treespec_is_prefix(treespec, treespec, strict=True)
        assert optree.treespec_is_suffix(treespec, treespec, strict=False)
        assert not optree.treespec_is_suffix(treespec, treespec, strict=True)

        if 'FlatCache' in str(treespec) or treespec == suffix_treespec:
            return

        assert treespec != suffix_treespec
        assert not (treespec == suffix_treespec)
        assert treespec != suffix_treespec
        assert treespec < suffix_treespec
        assert not (treespec > suffix_treespec)
        assert treespec <= suffix_treespec
        assert not (treespec >= suffix_treespec)
        assert suffix_treespec != treespec
        assert not (suffix_treespec == treespec)
        assert suffix_treespec > treespec
        assert not (suffix_treespec < treespec)
        assert suffix_treespec >= treespec
        assert not (suffix_treespec <= treespec)


@parametrize(
    data=list(
        itertools.chain(
            zip(TREES, TREE_STRINGS[False], itertools.repeat(False)),
            zip(TREES, TREE_STRINGS[True], itertools.repeat(True)),
        ),
    ),
)
def test_treespec_string_representation(data):
    tree, expected_string, none_is_leaf = data
    treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf)
    assert str(treespec) == expected_string
    assert repr(treespec) == expected_string

    assert expected_string.startswith('PyTreeSpec(')
    assert expected_string.endswith(')')
    if none_is_leaf:
        assert expected_string.endswith(', NoneIsLeaf)')
        representation = expected_string[len('PyTreeSpec(') : -len(', NoneIsLeaf)')]
    else:
        representation = expected_string[len('PyTreeSpec(') : -len(')')]

    if (
        'CustomTreeNode' not in representation
        and 'sys.float_info' not in representation
        and 'time.struct_time' not in representation
    ):
        representation = re.sub(
            r"<class '([\w\.]+)'>",
            lambda match: match.group(1),
            representation,
        )
        counter = itertools.count()
        representation = re.sub(r'\*', lambda _: str(next(counter)), representation)
        new_tree = optree.tree_unflatten(treespec, range(treespec.num_leaves))
        reconstructed_tree = eval(representation, helpers.__dict__.copy())
        assert new_tree == reconstructed_tree


def test_treespec_with_empty_tuple_string_representation():
    assert str(optree.tree_structure(())) == r'PyTreeSpec(())'


def test_treespec_with_single_element_tuple_string_representation():
    assert str(optree.tree_structure((1,))) == r'PyTreeSpec((*,))'


def test_treespec_with_empty_list_string_representation():
    assert str(optree.tree_structure([])) == r'PyTreeSpec([])'


def test_treespec_with_empty_dict_string_representation():
    assert str(optree.tree_structure({})) == r'PyTreeSpec({})'


@disable_systrace
def test_treespec_self_referential():
    class Holder:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Holder) and self.value == other.value

        def __hash__(self):
            return hash(self.value)

        def __repr__(self):
            return f'Holder({self.value!r})'

    key = Holder('a')

    hashes = set()
    treespec = optree.tree_structure({key: 0})
    assert str(treespec) == "PyTreeSpec({Holder('a'): *})"
    assert hash(treespec) == hash(treespec)
    hashes.add(hash(treespec))

    key.value = 'b'
    assert str(treespec) == "PyTreeSpec({Holder('b'): *})"
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))

    key.value = treespec
    assert str(treespec) == 'PyTreeSpec({Holder(...): *})'
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))

    key.value = ('a', treespec, treespec)
    assert str(treespec) == "PyTreeSpec({Holder(('a', ..., ...)): *})"
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))

    other = optree.tree_structure({Holder(treespec): 1})
    assert str(other) == "PyTreeSpec({Holder(PyTreeSpec({Holder(('a', ..., ...)): *})): *})"
    assert hash(other) == hash(other)
    assert hash(other) not in hashes
    hashes.add(hash(other))

    key.value = other
    assert str(treespec) == 'PyTreeSpec({Holder(PyTreeSpec({Holder(...): *})): *})'
    assert str(other) == 'PyTreeSpec({Holder(PyTreeSpec({Holder(...): *})): *})'
    assert hash(treespec) == hash(treespec)
    assert hash(treespec) not in hashes
    hashes.add(hash(treespec))
    assert hash(other) == hash(other)
    assert hash(treespec) == hash(other)

    gc_collect()
    if not PYPY:
        with recursionlimit(64):
            with pytest.raises(RecursionError):
                assert treespec != other

        wr = weakref.ref(treespec)
        del treespec, key, other
        gc_collect()
        assert wr() is None


@disable_systrace
def test_treeiter_self_referential():
    sentinel = object()

    d = {'a': 1}
    it = optree.tree_iter(d)
    assert next(it) == 1
    d['b'] = 2
    assert next(it, sentinel) is sentinel

    d = {'a': 1, 'b': {'c': 2}}
    it = optree.tree_iter(d)
    assert next(it) == 1
    d['b']['d'] = it
    assert next(it) == 2
    assert next(it) is it
    assert next(it, sentinel) is sentinel

    d = {'a': 1, 'b': {'c': 2}}
    it = optree.tree_iter(d)
    wr = weakref.ref(it)
    assert next(it) == 1
    d['b']['d'] = it
    assert next(it) == 2

    del it, d
    gc_collect()
    if not PYPY:
        assert wr() is None


def test_treeiter_leaf_predicate_no_reference_leak():
    # A reference cycle that runs through the `leaf_predicate` callback must be collectable.
    # Regression: `PyTreeIter` tp_traverse / tp_clear previously ignored `m_leaf_predicate`, so a
    # cycle through the predicate was invisible to the cyclic garbage collector and leaked.
    def is_leaf(x):
        return False

    it = optree.tree_iter({'a': 1, 'b': {'c': 2}}, is_leaf)
    wr = weakref.ref(it)
    assert next(it) == 1
    is_leaf.self_ref = it  # cycle: it -> m_leaf_predicate (is_leaf) -> is_leaf.self_ref -> it

    del it, is_leaf
    gc_collect()
    if not PYPY:
        assert wr() is None


def test_treespec_with_namespace():
    tree = NAMESPACED_TREE

    for namespace in ('', 'undefined'):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=False, namespace=namespace)
        assert leaves == [tree]
        assert str(treespec) == 'PyTreeSpec(*)'
        paths, leaves, treespec = optree.tree_flatten_with_path(
            tree,
            none_is_leaf=False,
            namespace=namespace,
        )
        assert paths == [()]
        assert leaves == [tree]
        assert paths == treespec.paths()
        assert str(treespec) == 'PyTreeSpec(*)'
        accessors, leaves, treespec = optree.tree_flatten_with_accessor(
            tree,
            none_is_leaf=False,
            namespace=namespace,
        )
        assert accessors == [optree.PyTreeAccessor()]
        assert leaves == [tree]
        assert accessors == treespec.accessors()
        assert str(treespec) == 'PyTreeSpec(*)'
    for namespace in ('', 'undefined'):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True, namespace=namespace)
        assert leaves == [tree]
        assert str(treespec) == 'PyTreeSpec(*, NoneIsLeaf)'
        paths, leaves, treespec = optree.tree_flatten_with_path(
            tree,
            none_is_leaf=True,
            namespace=namespace,
        )
        assert paths == [()]
        assert leaves == [tree]
        assert paths == treespec.paths()
        assert str(treespec) == 'PyTreeSpec(*, NoneIsLeaf)'
        accessors, leaves, treespec = optree.tree_flatten_with_accessor(
            tree,
            none_is_leaf=True,
            namespace=namespace,
        )
        assert accessors == [optree.PyTreeAccessor()]
        assert leaves == [tree]
        assert accessors == treespec.accessors()
        assert str(treespec) == 'PyTreeSpec(*, NoneIsLeaf)'

    expected_string = "PyTreeSpec(CustomTreeNode(MyAnotherDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [None, *, *]), *]), namespace='namespace')"
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=False, namespace='namespace')
    assert leaves == [2, 1, 101]
    assert str(treespec) == expected_string
    paths, leaves, treespec = optree.tree_flatten_with_path(
        tree,
        none_is_leaf=False,
        namespace='namespace',
    )
    assert paths == [('foo', 'b'), ('foo', 'a'), ('baz',)]
    assert leaves == [2, 1, 101]
    assert paths == treespec.paths()
    assert str(treespec) == expected_string
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(
        tree,
        none_is_leaf=False,
        namespace='namespace',
    )
    assert accessors == [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', MyAnotherDict, optree.PyTreeKind.CUSTOM),),
        ),
    ]
    assert leaves == [2, 1, 101]
    assert accessors == treespec.accessors()
    assert str(treespec) == expected_string

    expected_string = "PyTreeSpec(CustomTreeNode(MyAnotherDict[['foo', 'baz']], [CustomTreeNode(MyDict[['c', 'b', 'a']], [*, *, *]), *]), NoneIsLeaf, namespace='namespace')"
    leaves, treespec = optree.tree_flatten(tree, none_is_leaf=True, namespace='namespace')
    assert leaves == [None, 2, 1, 101]
    assert str(treespec) == expected_string
    paths, leaves, treespec = optree.tree_flatten_with_path(
        tree,
        none_is_leaf=True,
        namespace='namespace',
    )
    assert paths == [('foo', 'c'), ('foo', 'b'), ('foo', 'a'), ('baz',)]
    assert leaves == [None, 2, 1, 101]
    assert paths == treespec.paths()
    assert str(treespec) == expected_string
    accessors, leaves, treespec = optree.tree_flatten_with_accessor(
        tree,
        none_is_leaf=True,
        namespace='namespace',
    )
    assert accessors == [
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('c', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('b', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (
                optree.MappingEntry('foo', MyAnotherDict, optree.PyTreeKind.CUSTOM),
                optree.MappingEntry('a', MyDict, optree.PyTreeKind.CUSTOM),
            ),
        ),
        optree.PyTreeAccessor(
            (optree.MappingEntry('baz', MyAnotherDict, optree.PyTreeKind.CUSTOM),),
        ),
    ]
    assert leaves == [None, 2, 1, 101]
    assert accessors == treespec.accessors()
    assert str(treespec) == expected_string


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_pickle_roundtrip(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        expected = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        try:
            pickle.loads(pickle.dumps(tree))
        except pickle.PicklingError:
            with pytest.raises(pickle.PicklingError, match=r"Can't pickle .*:"):
                pickle.loads(pickle.dumps(expected))
        else:
            actual = pickle.loads(pickle.dumps(expected))
            assert actual == expected
            if expected.type in STANDARD_DICT_TYPES:
                assert list(optree.tree_unflatten(actual, range(len(actual)))) == list(
                    optree.tree_unflatten(expected, range(len(expected))),
                )


class Foo:
    def __init__(self, x, y):
        self.x = x
        self.y = y


@skipif_wasm
@skipif_android
@skipif_ios
def test_treespec_pickle_missing_registration():
    if sys.version_info[:2] == (3, 11) and platform.system() == 'Windows' and Py_DEBUG:
        pytest.skip('Python 3.11 on Windows has a bug during PyStructSequence type deallocation.')

    optree.register_pytree_node(
        Foo,
        lambda foo: ((foo.x, foo.y), None),
        lambda _, children: Foo(*children),
        namespace='foo',
    )

    treespec = optree.tree_structure(Foo(0, 1), namespace='foo')
    serialized = pickle.dumps(treespec)

    try:
        output = subprocess.run(
            [
                sys.executable,
                '-c',
                textwrap.dedent(
                    f"""
                    import pickle
                    import sys

                    sys.path.insert(0, {str(TEST_ROOT)!r})

                    try:
                        treespec = pickle.loads({serialized!r})
                    except Exception as ex:
                        print(ex)
                    else:
                        print('No exception was raised.', file=sys.stderr)
                        sys.exit(1)
                    """,
                ).strip(),
            ],
            capture_output=True,
            check=True,
            text=True,
            encoding='utf-8',
            cwd=TEST_ROOT,
            env={
                key: value
                for key, value in os.environ.items()
                if (
                    not key.startswith(('PYTHON', 'PYTEST', 'COV_'))
                    or key in ('PYTHON_GIL', 'PYTHONDEVMODE', 'PYTHONHASHSEED')
                )
            },
            timeout=120.0,
        )
        message = output.stdout.strip()
    except subprocess.CalledProcessError as ex:
        raise RuntimeError(ex.stderr) from ex

    assert re.match(
        r"^Unknown custom type in pickled PyTreeSpec: <class '.*'> in namespace 'foo'\.$",
        string=message,
    )

    optree.unregister_pytree_node(Foo, namespace='foo')
    with pytest.raises(
        RuntimeError,
        match=r"^Unknown custom type in pickled PyTreeSpec: <class '.*'> in namespace 'foo'\.$",
    ):
        treespec = pickle.loads(serialized)


def test_treespec_setstate_rejects_malformed_state():
    # `PyTreeSpec.__setstate__` (used by `pickle`) must reject structurally malformed state rather
    # than build a corrupt spec that triggers out-of-bounds reads / crashes when later used. The
    # per-node tuple layout is (kind, arity, node_data, node_entries, custom, num_leaves, num_nodes,
    # original_keys); see `PyTreeSpec::FromPickleable`.
    def setstate(state):
        obj = optree.PyTreeSpec.__new__(optree.PyTreeSpec)
        obj.__setstate__(state)
        return obj

    # Sanity: well-formed states still round-trip.
    for spec in [
        optree.tree_structure((0, 0)),
        optree.tree_structure({'a': 0, 'b': 0}),
        optree.tree_structure(defaultdict(int, {'a': 0, 'b': 0})),
    ]:
        assert setstate(spec.__getstate__()) == spec

    malformed_exceptions = (RuntimeError, ValueError, TypeError)

    # Negative arity.
    with pytest.raises(malformed_exceptions):
        setstate((((3, -1, None, None, None, 0, 1, None),), False, ''))

    # DefaultDict metadata as a list where a 2-tuple is expected previously caused a raw tuple-item
    # read to segfault; it is now coerced to a tuple and used safely.
    restored = setstate(
        (
            (
                (1, 0, None, None, None, 1, 1, None),
                (1, 0, None, None, None, 1, 1, None),
                (8, 2, [int, ['a', 'b']], None, None, 2, 3, {'a': None, 'b': None}),
            ),
            False,
            '',
        ),
    )
    assert optree.tree_unflatten(restored, [10, 20]) == defaultdict(int, {'a': 10, 'b': 20})

    # DefaultDict metadata with the wrong tuple size is rejected.
    with pytest.raises(malformed_exceptions):
        setstate(
            (
                (
                    (1, 0, None, None, None, 1, 1, None),
                    (1, 0, None, None, None, 1, 1, None),
                    (8, 2, (int, ['a', 'b'], 'extra'), None, None, 2, 3, {'a': None, 'b': None}),
                ),
                False,
                '',
            ),
        )

    # Dict key list shorter than arity (MakeNode would index past the list end).
    with pytest.raises(malformed_exceptions):
        setstate(
            (
                (
                    (1, 0, None, None, None, 1, 1, None),
                    (1, 0, None, None, None, 1, 1, None),
                    (5, 2, ['a'], None, None, 2, 3, {'a': None, 'b': None}),
                ),
                False,
                '',
            ),
        )

    # Inconsistent intermediate num_nodes (previously only the last node was checked).
    with pytest.raises(malformed_exceptions):
        setstate(
            (
                (
                    (1, 0, None, None, None, 1, 5, None),  # leaf claims num_nodes == 5
                    (1, 0, None, None, None, 1, 1, None),
                    (3, 2, None, None, None, 2, 3, None),
                ),
                False,
                '',
            ),
        )


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_type(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.is_leaf():
            assert treespec.type is None
        else:
            assert type(tree) is treespec.type


@parametrize(
    tree=TREES,
    inner_tree=[
        None,
        '*',
        (),
        (None,),
        ('*',),
        ['*', '*', '*'],
        ['*', '*', None],
        {'a': '*', 'b': None},
        {'a': '*', 'b': ('*', '*')},
    ],
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_compose_children(
    tree,
    inner_tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        inner_treespec = optree.tree_structure(
            inner_tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        expected_treespec = optree.tree_structure(
            optree.tree_map(
                lambda _: inner_tree,
                tree,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            ),
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        composed_treespec = treespec.compose(inner_treespec)
        transformed_treespec = treespec.transform(None, lambda _: inner_treespec)
        expected_leaves = treespec.num_leaves * inner_treespec.num_leaves
        assert composed_treespec.num_leaves == treespec.num_leaves * inner_treespec.num_leaves
        assert transformed_treespec.num_leaves == expected_leaves
        expected_nodes = (treespec.num_nodes - treespec.num_leaves) + (
            inner_treespec.num_nodes * treespec.num_leaves
        )
        assert composed_treespec.num_nodes == expected_nodes
        assert transformed_treespec.num_nodes == expected_nodes
        leaves = list(range(expected_leaves))
        composed = optree.tree_unflatten(composed_treespec, leaves)
        transformed = optree.tree_unflatten(transformed_treespec, leaves)
        assert composed == transformed

        if 'FlatCache' in str(treespec):
            return

        assert (leaves, composed_treespec) == optree.tree_flatten(
            composed,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert (leaves, transformed_treespec) == optree.tree_flatten(
            transformed,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )

        assert composed_treespec == expected_treespec
        assert transformed_treespec == expected_treespec

        stack = [(composed_treespec.children(), expected_treespec.children())]
        while stack:
            composed_children, expected_children = stack.pop()
            for composed_child, expected_child in zip(composed_children, expected_children):
                assert composed_child == expected_child
                stack.append((composed_child.children(), expected_child.children()))

        if treespec == expected_treespec:
            assert not (treespec != expected_treespec)
            assert not (treespec < expected_treespec)
            assert treespec <= expected_treespec
            assert not (treespec > expected_treespec)
            assert treespec >= expected_treespec
            assert expected_treespec >= treespec
            assert not (expected_treespec > treespec)
            assert expected_treespec <= treespec
            assert not (expected_treespec < treespec)
            assert not optree.treespec_is_prefix(treespec, expected_treespec, strict=True)
            assert optree.treespec_is_prefix(treespec, expected_treespec, strict=False)
            assert not optree.treespec_is_suffix(treespec, expected_treespec, strict=True)
            assert optree.treespec_is_suffix(treespec, expected_treespec, strict=False)
            assert not optree.treespec_is_prefix(expected_treespec, treespec, strict=True)
            assert optree.treespec_is_prefix(expected_treespec, treespec, strict=False)
            assert not optree.treespec_is_suffix(expected_treespec, treespec, strict=True)
            assert optree.treespec_is_suffix(expected_treespec, treespec, strict=False)
        else:
            assert treespec != expected_treespec
            assert treespec < expected_treespec
            assert treespec <= expected_treespec
            assert not (treespec > expected_treespec)
            assert not (treespec >= expected_treespec)
            assert expected_treespec >= treespec
            assert expected_treespec > treespec
            assert not (expected_treespec <= treespec)
            assert not (expected_treespec < treespec)
            assert optree.treespec_is_prefix(treespec, expected_treespec, strict=True)
            assert optree.treespec_is_prefix(treespec, expected_treespec, strict=False)
            assert not optree.treespec_is_suffix(treespec, expected_treespec, strict=True)
            assert not optree.treespec_is_suffix(treespec, expected_treespec, strict=False)
            assert not optree.treespec_is_prefix(expected_treespec, treespec, strict=True)
            assert not optree.treespec_is_prefix(expected_treespec, treespec, strict=False)
            assert optree.treespec_is_suffix(expected_treespec, treespec, strict=True)
            assert optree.treespec_is_suffix(expected_treespec, treespec, strict=False)


def test_treespec_compose_rejects_incompatible_namespace_merge():
    # Regression: composing an empty-namespace spec (whose custom nodes are resolved globally) with a
    # spec in another namespace adopted that namespace but kept the global registrations. When the
    # same type is registered differently in the two namespaces, the composed spec silently used the
    # wrong flatten/unflatten (spurious flatten_up_to errors; corrupt pickle). The merge is rejected.
    class Pair:
        def __init__(self, a, b):
            self.a, self.b = a, b

    class Single:
        def __init__(self, x):
            self.x = x

    optree.register_pytree_node(
        Pair,
        lambda t: ((t.a, t.b), None, None),
        lambda m, c: Pair(c[0], c[1]),
        namespace=GLOBAL_NAMESPACE,
    )
    optree.register_pytree_node(  # behavior differs from the global registration
        Pair,
        lambda t: ((t.b, t.a), None, None),
        lambda m, c: Pair(c[1], c[0]),
        namespace='behavior_change',
    )
    optree.register_pytree_node(
        Single,
        lambda t: ((t.x,), None, None),
        lambda m, c: Single(c[0]),
        namespace='behavior_change',
    )
    try:
        outer = optree.tree_structure(Pair(0, 0))
        inner = optree.tree_structure(Single(0), namespace='behavior_change')
        assert outer.namespace == ''
        assert inner.namespace == 'behavior_change'
        with pytest.raises(ValueError, match='different registration'):
            outer.compose(inner)

        # `tree_transpose` builds its expected structure with `compose`, so the rejection surfaces
        # through the public API too (here via its structure-mismatch diagnostic path).
        with pytest.raises(ValueError, match='different registration'):
            optree.tree_transpose(outer, inner, [1, 2, 3])

        # `broadcast_to_common_suffix` adopts the namespace the same way.
        self_spec = optree.tree_structure({'k': Pair(0, 0)})
        other_spec = optree.tree_structure({'k': Single(0)}, namespace='behavior_change')
        with pytest.raises(ValueError, match='different registration'):
            self_spec.broadcast_to_common_suffix(other_spec)
    finally:
        optree.unregister_pytree_node(Pair, namespace=GLOBAL_NAMESPACE)
        optree.unregister_pytree_node(Pair, namespace='behavior_change')
        optree.unregister_pytree_node(Single, namespace='behavior_change')


def test_treespec_compose_allows_compatible_namespace_merge():
    # The namespace-merge rejection must not over-reject: a custom type registered only globally
    # resolves identically under any namespace (via global fallback), so merging an empty-namespace
    # spec that uses it into another namespace is allowed and the result stays consistent.
    class GlobalOnly:
        def __init__(self, a, b):
            self.a, self.b = a, b

    optree.register_pytree_node(
        GlobalOnly,
        lambda t: ((t.a, t.b), None, None),
        lambda m, c: GlobalOnly(c[0], c[1]),
        namespace=GLOBAL_NAMESPACE,
    )
    try:
        outer = optree.tree_structure(GlobalOnly(0, 0))
        assert outer.namespace == ''

        # Both empty -> the merge stays in the global namespace.
        assert outer.compose(optree.tree_structure(0)).namespace == ''

        # Empty side (global-only custom) merged into a namespace: allowed, adopts the namespace,
        # and unflattens consistently (the global registration is used throughout).
        with optree.dict_insertion_ordered(True, namespace='no_override'):
            inner = optree.tree_structure({'x': 0}, namespace='no_override')
        assert inner.namespace == 'no_override'
        composed = outer.compose(inner)
        assert composed.namespace == 'no_override'
        result = optree.tree_unflatten(composed, [1, 2])
        assert isinstance(result, GlobalOnly)
        assert result.a == {'x': 1}
        assert result.b == {'x': 2}

        # The cross-namespace merge equals building the composed structure directly with `tree_map`
        # in the adopted namespace -- compose's defining identity.
        expected = optree.tree_structure(
            optree.tree_map(lambda _: {'x': 0}, GlobalOnly(0, 0), namespace='no_override'),
            namespace='no_override',
        )
        assert composed == expected

        # broadcast_to_common_suffix likewise allows the compatible merge.
        broadcasted = outer.broadcast_to_common_suffix(
            optree.tree_structure(GlobalOnly(0, 0), namespace='no_override'),
        )
        assert broadcasted.namespace == 'no_override'
    finally:
        optree.unregister_pytree_node(GlobalOnly, namespace=GLOBAL_NAMESPACE)


def test_treespec_broadcast_to_common_suffix_does_not_mutate_argument_on_key_mismatch():
    # Regression: BroadcastToCommonSuffixImpl built the "got key(s)" part of its key-mismatch error
    # message by sorting the ARGUMENT spec's live dict-node key list IN PLACE -- `other_keys` was a
    # borrow of `node_data`, not a copy. For an OrderedDict the child subtrees stay in insertion
    # order while the keys get permuted, silently corrupting a spec the caller still holds: repr,
    # equality, hash, and unflatten all go wrong. The message must be built from a sorted COPY.
    other = optree.tree_structure(OrderedDict([('c', 1), ('b', 2)]))
    before_repr = str(other)
    before_hash = hash(other)
    this = optree.tree_structure({'a': 1})
    with pytest.raises(ValueError, match='dictionary key mismatch'):
        this.broadcast_to_common_suffix(other)
    # The argument spec must be byte-for-byte unchanged by the failed call.
    assert str(other) == before_repr
    assert hash(other) == before_hash
    # And it must still unflatten in its ORIGINAL insertion order (c, b), not a sorted (b, c) order.
    assert other.unflatten([10, 20]) == OrderedDict([('c', 10), ('b', 20)])


def test_treespec_compose_rejects_namespace_override_with_different_arity():
    # A type registered globally flattens both members as children (arity 2); a namespace override
    # flattens one member as a child and stores the other as node metadata (arity 1). Both
    # registrations round-trip, but merging an empty-namespace spec (global, arity 2) into that
    # namespace must be rejected: the composed spec would claim the namespace while carrying an
    # arity-2 node that the namespace's registration cannot unflatten.
    class TwoMember:
        def __init__(self, a, b):
            self.a, self.b = a, b

        def __eq__(self, other):
            return isinstance(other, TwoMember) and (self.a, self.b) == (other.a, other.b)

        __hash__ = None

    optree.register_pytree_node(
        TwoMember,
        lambda t: ((t.a, t.b), None, None),  # global: both members are children
        lambda metadata, children: TwoMember(children[0], children[1]),
        namespace=GLOBAL_NAMESPACE,
    )
    optree.register_pytree_node(
        TwoMember,
        lambda t: ((t.a,), t.b, None),  # override: one child, the other is metadata
        lambda metadata, children: TwoMember(children[0], metadata),
        namespace='arity_change',
    )
    try:
        obj = TwoMember(1, 2)

        # Both registrations round-trip on their own.
        global_leaves, global_spec = optree.tree_flatten(obj)
        assert global_leaves == [1, 2]
        assert optree.tree_unflatten(global_spec, global_leaves) == obj
        custom_leaves, custom_spec = optree.tree_flatten(obj, namespace='arity_change')
        assert custom_leaves == [1]
        assert optree.tree_unflatten(custom_spec, custom_leaves) == obj

        assert global_spec.namespace == ''
        assert global_spec.num_leaves == 2
        assert custom_spec.namespace == 'arity_change'
        assert custom_spec.num_leaves == 1
        with pytest.raises(ValueError, match='different registration'):
            global_spec.compose(custom_spec)
    finally:
        optree.unregister_pytree_node(TwoMember, namespace=GLOBAL_NAMESPACE)
        optree.unregister_pytree_node(TwoMember, namespace='arity_change')


def test_treespec_transform_rejects_incompatible_namespace_merge():
    # `transform` unifies the namespace across the input spec and the transform outputs. If that
    # unified (non-empty) namespace rebinds a custom node -- e.g. the input's globally-resolved
    # custom node -- to a different registration, the transform must be rejected (same class as the
    # compose / broadcast merge rejection). A globally-only-registered type is still allowed via
    # fallback.
    class TransformT:
        def __init__(self, a, b):
            self.a, self.b = a, b

    class TransformS:
        def __init__(self, x):
            self.x = x

    class TransformGlobal:
        def __init__(self, a, b):
            self.a, self.b = a, b

    optree.register_pytree_node(
        TransformT,
        lambda t: ((t.a, t.b), None, None),
        lambda m, c: TransformT(c[0], c[1]),
        namespace=GLOBAL_NAMESPACE,
    )
    optree.register_pytree_node(
        TransformT,
        lambda t: ((t.b, t.a), None, None),
        lambda m, c: TransformT(c[1], c[0]),
        namespace='transform_change',
    )
    optree.register_pytree_node(
        TransformS,
        lambda t: ((t.x,), None, None),
        lambda m, c: TransformS(c[0]),
        namespace='transform_change',
    )
    optree.register_pytree_node(
        TransformGlobal,
        lambda t: ((t.a, t.b), None, None),
        lambda m, c: TransformGlobal(c[0], c[1]),
        namespace=GLOBAL_NAMESPACE,
    )

    def to_namespaced_leaf(_):
        return optree.tree_structure(TransformS(0), namespace='transform_change')

    def to_namespaced_node(nodespec):
        # Outer tuple -> globally-registered TransformT; inner lists -> 'transform_change' TransformS.
        if nodespec.type is tuple:
            return optree.tree_structure(TransformT(0, 0))
        return to_namespaced_leaf(None)

    try:
        # The rejection must fire for every `(f_node, f_leaf)` combination that puts a
        # globally-resolved custom node under the non-empty unified namespace.

        # (None, f_leaf): the input's global TransformT is kept, f_leaf injects the namespace.
        outer = optree.tree_structure(TransformT(0, 0))
        assert outer.namespace == ''
        with pytest.raises(ValueError, match='different registration'):
            outer.transform(None, to_namespaced_leaf)

        # (f_node, None): f_node alone yields a global TransformT above 'transform_change' children.
        with pytest.raises(ValueError, match='different registration'):
            optree.tree_structure(([0], [0])).transform(to_namespaced_node, None)

        # (f_node, f_leaf): f_node injects the global TransformT, f_leaf injects the namespace.
        with pytest.raises(ValueError, match='different registration'):
            optree.tree_structure([0, 0]).transform(
                lambda _: optree.tree_structure(TransformT(0, 0)),
                to_namespaced_leaf,
            )

        # Compatible: TransformGlobal resolves identically under any namespace via fallback.
        global_outer = optree.tree_structure(TransformGlobal(0, 0))
        transformed = global_outer.transform(None, to_namespaced_leaf)
        assert transformed.namespace == 'transform_change'
    finally:
        optree.unregister_pytree_node(TransformT, namespace=GLOBAL_NAMESPACE)
        optree.unregister_pytree_node(TransformT, namespace='transform_change')
        optree.unregister_pytree_node(TransformS, namespace='transform_change')
        optree.unregister_pytree_node(TransformGlobal, namespace=GLOBAL_NAMESPACE)


def test_treespec_is_prefix_nested_dict_key_reorder():
    # Regression: `IsPrefix` reorders a dict node's children in a working copy of the traversal to
    # make key order irrelevant. When a NESTED dict also needed reordering, it indexed the pristine
    # traversal by an offset into the already-mutated working copy, corrupting it -> a spurious
    # `optree._C.InternalError` or a wrong boolean. Two treespecs that describe the SAME tree
    # (differing only in dict key insertion order, at nested levels) must be mutual non-strict
    # prefixes / suffixes.

    # Top-level AND nested dict keys reordered; the top-level reorder relocates the nested dict.
    tree_a = OrderedDict([('a', 0), ('b', OrderedDict([('e', 0), ('g', 0)])), ('d', 0)])
    tree_b = OrderedDict([('b', OrderedDict([('g', 0), ('e', 0)])), ('a', 0), ('d', 0)])
    a = optree.tree_structure(tree_a)
    b = optree.tree_structure(tree_b)
    assert optree.treespec_is_prefix(a, b, strict=False)
    assert optree.treespec_is_prefix(b, a, strict=False)
    assert optree.treespec_is_suffix(a, b, strict=False)
    assert optree.treespec_is_suffix(b, a, strict=False)
    assert a <= b
    assert b <= a
    assert a >= b
    assert b >= a

    # A nested dict whose reorder relocates a subtree containing another out-of-order dict.
    tree_a2 = OrderedDict([('a', 0), ('d', 0), ('b', OrderedDict([('e', 0), ('f', 0)]))])
    tree_b2 = OrderedDict([('b', OrderedDict([('f', 0), ('e', 0)])), ('d', 0), ('a', 0)])
    a2 = optree.tree_structure(tree_a2)
    b2 = optree.tree_structure(tree_b2)
    assert optree.treespec_is_prefix(a2, b2, strict=False)
    assert optree.treespec_is_prefix(b2, a2, strict=False)
    assert a2 <= b2
    assert b2 <= a2


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_entries(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        expected_paths, _, treespec = optree.tree_flatten_with_path(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert optree.treespec_paths(treespec) == expected_paths

        def gen_path(spec):
            entries = optree.treespec_entries(spec)
            children = optree.treespec_children(spec)
            assert len(entries) == spec.num_children
            assert len(children) == spec.num_children
            assert entries is not optree.treespec_entries(spec)
            assert children is not optree.treespec_children(spec)
            optree.treespec_entries(spec).clear()
            optree.treespec_children(spec).clear()

            if spec.is_leaf():
                assert spec.num_children == 0
                yield ()
                return

            for entry, child in zip(entries, children):
                for suffix in gen_path(child):
                    yield (entry, *suffix)

        paths = list(gen_path(treespec))
        assert paths == expected_paths

        expected_accessors, _, other_treespec = optree.tree_flatten_with_accessor(
            tree,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        assert optree.treespec_accessors(treespec) == expected_accessors
        assert optree.treespec_accessors(other_treespec) == expected_accessors
        assert treespec == other_treespec

        def gen_typed_path(spec):
            entries = optree.treespec_entries(spec)
            children = optree.treespec_children(spec)
            assert len(entries) == spec.num_children
            assert len(children) == spec.num_children

            if spec.is_leaf():
                assert spec.num_children == 0
                yield ()
                return

            node_type = spec.type
            node_kind = spec.kind
            for entry, child in zip(entries, children):
                for suffix in gen_typed_path(child):
                    yield ((entry, node_type, node_kind), *suffix)

        typed_paths = list(gen_typed_path(treespec))
        expected_typed_paths = [
            tuple((e.entry, e.type, e.kind) for e in accessor) for accessor in expected_accessors
        ]
        assert typed_paths == expected_typed_paths


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_entry(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None or treespec.type is type(None):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 1)
        if treespec.is_leaf(strict=False):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Entry() index out of range.'),
            ):
                optree.treespec_entry(treespec, 1)
        expected_entries = optree.treespec_entries(treespec)
        for i, entry in enumerate(expected_entries):
            assert entry == optree.treespec_entry(treespec, i)
            assert entry == optree.treespec_entry(treespec, i - len(expected_entries))
            assert optree.treespec_entry(treespec, i) == optree.treespec_entry(treespec, i)
            assert optree.treespec_entry(
                treespec,
                i - len(expected_entries),
            ) == optree.treespec_entry(
                treespec,
                i - len(expected_entries),
            )
            assert optree.treespec_entry(treespec, i) == optree.treespec_entry(
                treespec,
                i - len(expected_entries),
            )
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, len(expected_entries))
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Entry() index out of range.')):
            optree.treespec_entry(treespec, -len(expected_entries) - 1)

        assert expected_entries == [
            optree.treespec_entry(treespec, i) for i in range(len(expected_entries))
        ]


def test_treespec_children():
    treespec = optree.tree_structure(((1, 2, 3), (4,)))
    c0 = optree.tree_structure((0, 0, 0))
    c1 = optree.tree_structure((7,))
    assert optree.treespec_children(treespec) == [c0, c1]

    treespec = optree.tree_structure(((1, 2, 3), (4,)))
    c0 = optree.tree_structure((0, 0, 0))
    c1 = optree.tree_structure((7,), none_is_leaf=True)
    assert optree.treespec_children(treespec) != [c0, c1]

    treespec = optree.tree_structure(((1, 2, None), (4,)), none_is_leaf=False)
    c0 = optree.tree_structure((0, 0, None), none_is_leaf=False)
    c1 = optree.tree_structure((7,), none_is_leaf=False)
    assert optree.treespec_children(treespec) == [c0, c1]

    treespec = optree.tree_structure(((1, 2, 3, None), (4,)), none_is_leaf=True)
    c0 = optree.tree_structure((0, 0, 0, 0), none_is_leaf=True)
    c1 = optree.tree_structure((7,), none_is_leaf=True)
    assert optree.treespec_children(treespec) == [c0, c1]


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_child(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None or treespec.type is type(None):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 1)
        if treespec.is_leaf(strict=False):
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 0)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, -1)
            with pytest.raises(
                IndexError,
                match=re.escape('PyTreeSpec::Child() index out of range.'),
            ):
                optree.treespec_child(treespec, 1)
        expected_children = optree.treespec_children(treespec)
        for i, child in enumerate(expected_children):
            assert child == optree.treespec_child(treespec, i)
            assert child == optree.treespec_child(treespec, i - len(expected_children))
            assert optree.treespec_child(treespec, i) == optree.treespec_child(treespec, i)
            assert optree.treespec_child(
                treespec,
                i - len(expected_children),
            ) == optree.treespec_child(
                treespec,
                i - len(expected_children),
            )
            assert optree.treespec_child(treespec, i) == optree.treespec_child(
                treespec,
                i - len(expected_children),
            )
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, len(expected_children))
        with pytest.raises(IndexError, match=re.escape('PyTreeSpec::Child() index out of range.')):
            optree.treespec_child(treespec, -len(expected_children) - 1)

        assert expected_children == [
            optree.treespec_child(treespec, i) for i in range(len(expected_children))
        ]


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_one_level(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None:
            assert treespec.is_leaf()
            assert optree.treespec_one_level(treespec) is None
            assert optree.treespec_children(treespec) == []
            assert treespec.num_children == 0
        else:
            one_level = optree.treespec_one_level(treespec)
            counter = itertools.count()
            expected_treespec = optree.tree_structure(
                tree,
                is_leaf=lambda x: next(counter) > 0,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            num_children = treespec.num_children
            assert not treespec.is_leaf()
            assert not one_level.is_leaf()
            assert not expected_treespec.is_leaf()
            assert one_level == expected_treespec
            assert optree.treespec_one_level(one_level) == one_level
            assert optree.treespec_one_level(expected_treespec) == expected_treespec
            assert one_level.num_nodes == num_children + 1
            assert one_level.num_leaves == num_children
            assert one_level.num_children == num_children
            assert len(one_level) == num_children
            assert optree.treespec_entries(one_level) == optree.treespec_entries(treespec)
            assert all(optree.treespec_child(one_level, i).is_leaf() for i in range(num_children))
            assert all(child.is_leaf() for child in optree.treespec_children(one_level))
            assert optree.treespec_is_prefix(one_level, treespec)
            assert optree.treespec_is_suffix(treespec, one_level)
            assert (
                optree.treespec_from_collection(
                    optree.tree_unflatten(one_level, treespec.children()),
                    none_is_leaf=none_is_leaf,
                    namespace=namespace,
                )
                == treespec
            )
            it = iter(treespec.children())
            assert optree.treespec_transform(one_level, None, lambda _: next(it)) == treespec


def test_treespec_transform():
    treespec = optree.tree_structure(((1, 2, 3), (4,)))
    assert optree.treespec_transform(treespec) == treespec
    assert optree.treespec_transform(treespec) is not treespec
    assert optree.treespec_transform(
        treespec,
        None,
        lambda _: optree.tree_structure((1, [2])),
    ) == optree.tree_structure((((0, [1]), (2, [3]), (4, [5])), ((6, [7]),)))
    assert optree.treespec_transform(
        treespec,
        lambda spec: optree.treespec_list(spec.children()),
    ) == optree.tree_structure([[1, 2, 3], [4]])
    assert optree.treespec_transform(
        treespec,
        lambda spec: optree.treespec_dict(zip('abcd', spec.children())),
    ) == optree.tree_structure({'a': {'a': 0, 'b': 1, 'c': 2}, 'b': {'a': 3}})
    assert optree.treespec_transform(
        treespec,
        lambda spec: optree.treespec_dict(zip('abcd', spec.children())),
        lambda spec: optree.tree_structure([0, None, 1]),
    ) == optree.tree_structure(
        {'a': {'a': [0, None, 1], 'b': [2, None, 3], 'c': [4, None, 5]}, 'b': {'a': [6, None, 7]}},
    )
    namespaced_treespec = optree.tree_structure(
        MyAnotherDict({1: MyAnotherDict({2: 1, 1: 2, 0: 3}), 0: MyAnotherDict({0: 4})}),
        namespace='namespace',
    )
    assert (
        optree.treespec_transform(
            treespec,
            lambda spec: optree.tree_structure(
                MyAnotherDict(zip(spec.entries(), spec.children())),
                namespace='namespace',
            ),
        )
        == namespaced_treespec
    )
    assert optree.treespec_transform(
        namespaced_treespec,
        lambda spec: optree.treespec_list(spec.children()),
    ) == optree.tree_structure([[1, 2, 3], [4]])

    with pytest.raises(
        TypeError,
        match=re.escape('Expected the PyTreeSpec transform function returns a PyTreeSpec'),
    ):
        optree.treespec_transform(treespec, lambda _: None)

    with pytest.raises(
        TypeError,
        match=re.escape('Expected the PyTreeSpec transform function returns a PyTreeSpec'),
    ):
        optree.treespec_transform(treespec, None, lambda _: None)

    with pytest.raises(
        ValueError,
        match=(
            r'Expected the PyTreeSpec transform function returns '
            r'a PyTreeSpec with the same value of `none_is_leaf=\w+` as the input'
        ),
    ):
        optree.treespec_transform(
            treespec,
            lambda spec: optree.treespec_list(
                [optree.treespec_leaf(none_is_leaf=True)] * spec.num_children,
                none_is_leaf=True,
            ),
        )

    def fn(spec):
        with optree.dict_insertion_ordered(True, namespace='undefined'):
            return optree.treespec_dict(zip('abcd', spec.children()), namespace='undefined')

    with pytest.raises(ValueError, match=r'Expected treespec\(s\) with namespace .*, got .*\.'):
        optree.treespec_transform(namespaced_treespec, fn)

    with pytest.raises(
        ValueError,
        match=re.escape(
            'Expected the PyTreeSpec transform function returns '
            'a PyTreeSpec with the same number of arity as the input',
        ),
    ):
        optree.treespec_transform(treespec, lambda _: optree.tree_structure([0, 1]))

    with pytest.raises(
        ValueError,
        match=re.escape(
            'Expected the PyTreeSpec transform function returns '
            'a one-level PyTreeSpec as the input',
        ),
    ):
        optree.treespec_transform(
            treespec,
            lambda spec: optree.tree_structure([None] + [0] * spec.num_children),
        )


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_nodes(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        nodes = []
        stack = [treespec]
        while stack:
            spec = stack.pop()
            nodes.append(spec)
            children = spec.children()
            stack.extend(reversed(children))
            assert spec.num_nodes == sum(child.num_nodes for child in children) + 1
        assert treespec.num_nodes == len(nodes)


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_leaves(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        leaves, treespec = optree.tree_flatten(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        assert treespec.num_leaves == len(leaves)
        assert treespec.num_leaves == len(treespec)
        assert treespec.num_leaves == len(treespec.paths())
        assert treespec.num_leaves == len(treespec.accessors())


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_num_children(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        assert treespec.num_children == len(treespec.entries())
        assert treespec.num_children == len(treespec.children())


def test_treespec_is_leaf():
    assert optree.treespec_is_strict_leaf(optree.tree_structure(1))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure((1, 2)))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure(None))
    assert optree.treespec_is_strict_leaf(optree.tree_structure(None, none_is_leaf=True))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure(()))
    assert not optree.treespec_is_strict_leaf(optree.tree_structure([]))
    assert optree.treespec_is_leaf(optree.tree_structure(1))
    assert not optree.treespec_is_leaf(optree.tree_structure((1, 2)))
    assert not optree.treespec_is_leaf(optree.tree_structure(None))
    assert optree.treespec_is_leaf(optree.tree_structure(None, none_is_leaf=True))
    assert not optree.treespec_is_leaf(optree.tree_structure(()))
    assert not optree.treespec_is_leaf(optree.tree_structure([]))
    assert optree.tree_structure(1).is_leaf(strict=True)
    assert not optree.tree_structure((1, 2)).is_leaf(strict=True)
    assert not optree.tree_structure(None).is_leaf(strict=True)
    assert optree.tree_structure(None, none_is_leaf=True).is_leaf(strict=True)
    assert not optree.tree_structure(()).is_leaf(strict=True)
    assert not optree.tree_structure([]).is_leaf(strict=True)

    assert optree.treespec_is_leaf(optree.tree_structure(1), strict=False)
    assert not optree.treespec_is_leaf(optree.tree_structure((1, 2)), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure(None), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure(None, none_is_leaf=True), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure(()), strict=False)
    assert optree.treespec_is_leaf(optree.tree_structure([]), strict=False)
    assert optree.tree_structure(1).is_leaf(strict=False)
    assert not optree.tree_structure((1, 2)).is_leaf(strict=False)
    assert optree.tree_structure(None).is_leaf(strict=False)
    assert optree.tree_structure(None, none_is_leaf=True).is_leaf(strict=False)
    assert optree.tree_structure(()).is_leaf(strict=False)
    assert optree.tree_structure([]).is_leaf(strict=False)


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_is_one_level(
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        treespec = optree.tree_structure(tree, none_is_leaf=none_is_leaf, namespace=namespace)
        if treespec.type is None:
            assert treespec.is_leaf()
            assert optree.treespec_one_level(treespec) is None
            assert not optree.treespec_is_one_level(treespec)
        else:
            one_level = optree.treespec_one_level(treespec)
            counter = itertools.count()
            expected_treespec = optree.tree_structure(
                tree,
                is_leaf=lambda x: next(counter) > 0,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            assert not treespec.is_leaf()
            assert not one_level.is_leaf()
            assert not expected_treespec.is_leaf()
            assert one_level == expected_treespec
            assert optree.treespec_one_level(one_level) == one_level
            assert optree.treespec_one_level(expected_treespec) == expected_treespec
            assert optree.treespec_is_one_level(one_level)
            assert optree.treespec_is_one_level(expected_treespec)
            assert optree.treespec_is_one_level(treespec) == (treespec == one_level)
            assert optree.treespec_is_one_level(treespec) == (treespec == expected_treespec)


@parametrize(
    namespace=['', 'undefined', 'namespace'],
)
def test_treespec_leaf_none(namespace):
    assert optree.treespec_leaf(none_is_leaf=False, namespace=namespace) != optree.treespec_leaf(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(namespace=namespace) == optree.tree_structure(
        1,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) == optree.tree_structure(
        1,
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) == optree.tree_structure(
        None,
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) != optree.tree_structure(
        None,
        none_is_leaf=False,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) == optree.treespec_none(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=True, namespace=namespace) != optree.treespec_none(
        none_is_leaf=False,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=False, namespace=namespace) != optree.treespec_none(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_leaf(none_is_leaf=False, namespace=namespace) != optree.treespec_none(
        none_is_leaf=False,
        namespace=namespace,
    )

    assert optree.treespec_none(none_is_leaf=False, namespace=namespace) != optree.treespec_none(
        none_is_leaf=True,
        namespace=namespace,
    )
    assert optree.treespec_none(namespace=namespace) == optree.tree_structure(
        None,
        namespace=namespace,
    )
    assert optree.treespec_none(namespace=namespace) != optree.tree_structure(
        1,
        namespace=namespace,
    )
    assert optree.treespec_none(none_is_leaf=True, namespace=namespace) == optree.tree_structure(
        1,
        none_is_leaf=True,
        namespace=namespace,
    )

    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert optree.treespec_from_collection(
            1,
            namespace=namespace,
        ) == optree.treespec_leaf(
            namespace=namespace,
        )
    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert optree.treespec_from_collection(
            1,
            none_is_leaf=True,
            namespace=namespace,
        ) == optree.treespec_leaf(
            none_is_leaf=True,
            namespace=namespace,
        )
    assert optree.treespec_from_collection(
        None,
        namespace=namespace,
    ) == optree.treespec_none(
        namespace=namespace,
    )
    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert optree.treespec_from_collection(
            None,
            none_is_leaf=True,
            namespace=namespace,
        ) == optree.treespec_none(
            none_is_leaf=True,
            namespace=namespace,
        )


@parametrize(
    tree=TREES,
    none_is_leaf=[False, True],
    namespace=['', 'undefined', 'namespace'],
    dict_should_be_sorted=[False, True],
    dict_session_namespace=['', 'undefined', 'namespace'],
)
def test_treespec_constructor(  # noqa: C901
    tree,
    none_is_leaf,
    namespace,
    dict_should_be_sorted,
    dict_session_namespace,
):
    use_sorted_keys = dict_should_be_sorted or dict_session_namespace not in {'', namespace}
    with optree.dict_insertion_ordered(
        not dict_should_be_sorted,
        namespace=dict_session_namespace or GLOBAL_NAMESPACE,
    ):
        for passed_namespace in sorted({'', namespace}):
            stack = [tree]
            while stack:
                node = stack.pop()
                counter = itertools.count()
                expected_treespec = optree.tree_structure(
                    node,
                    none_is_leaf=none_is_leaf,
                    namespace=namespace,
                )
                children, one_level_treespec = optree.tree_flatten(
                    node,
                    is_leaf=lambda x: next(counter) > 0,  # noqa: B023
                    none_is_leaf=none_is_leaf,
                    namespace=namespace,
                )
                node_type = type(node)
                if one_level_treespec.is_leaf():
                    assert len(children) == 1
                    with pytest.warns(
                        UserWarning,
                        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
                    ):
                        assert (
                            optree.treespec_from_collection(
                                node,
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                    assert (
                        optree.treespec_leaf(
                            none_is_leaf=none_is_leaf,
                            namespace=passed_namespace,
                        )
                        == expected_treespec
                    )
                else:
                    children_treespecs = [
                        optree.tree_structure(
                            child,
                            none_is_leaf=none_is_leaf,
                            namespace=namespace,
                        )
                        for child in children
                    ]
                    collection_of_treespecs = optree.tree_unflatten(
                        one_level_treespec,
                        children_treespecs,
                    )
                    assert (
                        optree.treespec_from_collection(
                            collection_of_treespecs,
                            none_is_leaf=none_is_leaf,
                            namespace=namespace,
                        )
                        == expected_treespec
                    )

                    if node_type in {type(None), tuple, list}:
                        if node_type is tuple:
                            assert (
                                optree.treespec_tuple(
                                    children_treespecs,
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    tuple(children_treespecs),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                        elif node_type is list:
                            assert (
                                optree.treespec_list(
                                    children_treespecs,
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    list(children_treespecs),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                        else:
                            assert len(children_treespecs) == 0
                            assert (
                                optree.treespec_none(
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    None,
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                    elif node_type is dict:
                        if use_sorted_keys:
                            assert (
                                optree.treespec_dict(
                                    zip(sorted(node), children_treespecs),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    dict(zip(sorted(node), children_treespecs)),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                        else:
                            context = (
                                optree.dict_insertion_ordered(
                                    True,
                                    namespace=passed_namespace or GLOBAL_NAMESPACE,
                                )
                                if dict_session_namespace != passed_namespace
                                else contextlib.nullcontext()
                            )
                            with context:
                                assert (
                                    optree.treespec_dict(
                                        zip(node, children_treespecs),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                                assert (
                                    optree.treespec_from_collection(
                                        dict(zip(node, children_treespecs)),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                    elif node_type is OrderedDict:
                        assert (
                            optree.treespec_ordereddict(
                                zip(node, children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                OrderedDict(zip(node, children_treespecs)),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                    elif node_type is defaultdict:
                        if use_sorted_keys:
                            assert (
                                optree.treespec_defaultdict(
                                    node.default_factory,
                                    zip(sorted(node), children_treespecs),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                            assert (
                                optree.treespec_from_collection(
                                    defaultdict(
                                        node.default_factory,
                                        zip(sorted(node), children_treespecs),
                                    ),
                                    none_is_leaf=none_is_leaf,
                                    namespace=passed_namespace,
                                )
                                == expected_treespec
                            )
                        else:
                            context = (
                                optree.dict_insertion_ordered(
                                    True,
                                    namespace=passed_namespace or GLOBAL_NAMESPACE,
                                )
                                if dict_session_namespace != passed_namespace
                                else contextlib.nullcontext()
                            )
                            with context:
                                assert (
                                    optree.treespec_defaultdict(
                                        node.default_factory,
                                        zip(node, children_treespecs),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                                assert (
                                    optree.treespec_from_collection(
                                        defaultdict(
                                            node.default_factory,
                                            zip(node, children_treespecs),
                                        ),
                                        none_is_leaf=none_is_leaf,
                                        namespace=passed_namespace,
                                    )
                                    == expected_treespec
                                )
                    elif node_type is deque:
                        assert (
                            optree.treespec_deque(
                                children_treespecs,
                                maxlen=node.maxlen,
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                deque(children_treespecs, maxlen=node.maxlen),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                    elif optree.is_structseq(node):
                        assert (
                            optree.treespec_structseq(
                                node_type(children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                node_type(children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        with pytest.raises(
                            ValueError,
                            match=r'Expected a namedtuple of PyTreeSpec\(s\), got .*\.',
                        ):
                            optree.treespec_namedtuple(
                                node_type(children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                    elif optree.is_namedtuple(node):
                        assert (
                            optree.treespec_namedtuple(
                                node_type(*children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        assert (
                            optree.treespec_from_collection(
                                node_type(*children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )
                            == expected_treespec
                        )
                        with pytest.raises(
                            ValueError,
                            match=r'Expected a PyStructSequence of PyTreeSpec\(s\), got .*\.',
                        ):
                            optree.treespec_structseq(
                                node_type(*children_treespecs),
                                none_is_leaf=none_is_leaf,
                                namespace=passed_namespace,
                            )

                    stack.extend(reversed(children))


def test_treespec_constructor_namespace():
    @optree.register_pytree_node_class(namespace='mylist')
    class MyList(UserList):
        def __tree_flatten__(self):
            return self.data, None, None

        @classmethod
        def __tree_unflatten__(cls, metadata, children):
            return cls(children)

    with pytest.warns(
        UserWarning,
        match=re.escape('PyTreeSpec::MakeFromCollection() is called on a leaf.'),
    ):
        assert (
            optree.treespec_from_collection(
                MyList([optree.treespec_leaf(), optree.treespec_leaf(), optree.treespec_leaf()]),
            )
            == optree.treespec_leaf()
        )

    expected_treespec = optree.tree_structure(MyList([1, 2, 3]), namespace='mylist')
    actual_treespec = optree.treespec_from_collection(
        MyList([optree.treespec_leaf(), optree.treespec_leaf(), optree.treespec_leaf()]),
        namespace='mylist',
    )
    assert actual_treespec == expected_treespec
    assert actual_treespec.type is MyList
    assert actual_treespec.namespace == 'mylist'

    children_treespecs = actual_treespec.children()
    assert all(child.namespace == 'mylist' for child in children_treespecs)
    treespec1 = optree.treespec_from_collection(list(children_treespecs), namespace='')
    assert treespec1.type is list
    assert treespec1.namespace == 'mylist'

    treespec2 = optree.treespec_from_collection(
        [optree.treespec_leaf(), optree.treespec_leaf(), optree.treespec_leaf()],
        namespace='mylist',
    )
    assert treespec2.type is list
    assert treespec2.namespace == ''

    assert treespec1 == treespec2


def test_treespec_dict_constructor_preserves_insertion_ordered_namespace():
    # Regression: under `dict_insertion_ordered` mode the key order of a dict spec depends on the
    # namespace, so `treespec_dict(..., namespace=...)` must keep that namespace (like `tree_flatten`)
    # instead of resetting it to '' -- an empty-namespace spec with unsorted keys is otherwise
    # unreachable via `tree_flatten` and breaks equality/consistency.
    leaf = optree.tree_structure(0)

    with optree.dict_insertion_ordered(True, namespace='namespace'):
        constructed = optree.treespec_dict({'b': leaf, 'a': leaf}, namespace='namespace')
        _, flattened = optree.tree_flatten({'b': 1, 'a': 2}, namespace='namespace')

    assert constructed.entries() == ['b', 'a']  # insertion order preserved
    assert flattened.namespace == 'namespace'
    assert constructed.namespace == 'namespace'  # was '' before the fix
    assert constructed == flattened

    # Without the mode, keys are sorted and the namespace is dropped -- same as `tree_flatten`.
    outside = optree.treespec_dict({'b': leaf, 'a': leaf}, namespace='namespace')
    _, flattened_outside = optree.tree_flatten({'b': 1, 'a': 2}, namespace='namespace')
    assert outside.entries() == ['a', 'b']
    assert outside.namespace == ''
    assert outside == flattened_outside


def test_treespec_constructor_none_treespec_inputs():
    with pytest.raises(ValueError, match=r'Expected a\(n\) list of PyTreeSpec\(s\), got .*\.'):
        optree.treespec_list([optree.treespec_leaf(), 1])

    with pytest.raises(ValueError, match=r'Expected a\(n\) list of PyTreeSpec\(s\), got .*\.'):
        optree.treespec_from_collection([optree.treespec_leaf(), 1])

    with pytest.raises(ValueError, match=r'Expected a\(n\) list of PyTreeSpec\(s\), got .*\.'):
        optree.treespec_from_collection(
            [
                optree.treespec_leaf(),
                (optree.treespec_leaf(), optree.treespec_leaf()),
            ],
        )

    assert optree.treespec_from_collection(
        [
            optree.treespec_leaf(),
            optree.treespec_tuple((optree.treespec_leaf(), optree.treespec_leaf())),
        ],
    ) == optree.tree_structure([0, (1, 2)])
