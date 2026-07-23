/*
Copyright 2022-2026 MetaOPT Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
================================================================================
*/

#include "optree/optree.h"

inline namespace {
#if PYBIND11_VERSION_HEX >= 0x030000F0  // pybind11 3.0.0
using pybind11::detail::is_holder_constructed;
#else
[[nodiscard]] inline bool is_holder_constructed(PyObject *obj) {
    auto * const instance = reinterpret_cast<pybind11::detail::instance *>(obj);
    return instance->get_value_and_holder().holder_constructed();
}
#endif
}  // namespace

namespace optree {

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ int PyTreeSpec::PyTpTraverse(PyObject *self_base, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self_base));
    if (!::is_holder_constructed(self_base)) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto &self = thread_safe_cast<PyTreeSpec &>(py::handle{self_base});
    PYTREESPEC_SANITY_CHECK(self);
    for (const auto &node : self.m_traversal) {
        Py_VISIT(node.node_data.ptr());
        Py_VISIT(node.node_entries.ptr());
        Py_VISIT(node.original_keys.ptr());
        if (node.custom != nullptr) [[unlikely]] {
            // The custom registration is shared with (and pinned by) the registry, but the treespec
            // transitively references its Python members; report them so the cyclic GC can see and
            // collect reference cycles that pass through a registered custom type or its functions.
            // `PyTpClear` needs no counterpart: `m_traversal.clear()` drops the treespec's hold on
            // the registration (a `shared_ptr`), which is the correct way to break the cycle.
            Py_VISIT(node.custom->type.ptr());
            Py_VISIT(node.custom->flatten_func.ptr());
            Py_VISIT(node.custom->unflatten_func.ptr());
            Py_VISIT(node.custom->path_entry_type.ptr());
        }
    }
    return 0;
}

/*static*/ int PyTreeSpec::PyTpClear(PyObject *self_base) {
    if (!::is_holder_constructed(self_base)) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto &self = thread_safe_cast<PyTreeSpec &>(py::handle{self_base});
    PYTREESPEC_SANITY_CHECK(self);
    for (auto &node : self.m_traversal) {
        Py_CLEAR(node.node_data.ptr());
        Py_CLEAR(node.node_entries.ptr());
        Py_CLEAR(node.original_keys.ptr());
    }
    self.m_traversal.clear();
    return 0;
}

// NOLINTNEXTLINE[readability-function-cognitive-complexity]
/*static*/ int PyTreeIter::PyTpTraverse(PyObject *self_base, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self_base));
    if (!::is_holder_constructed(self_base)) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto &self = thread_safe_cast<PyTreeIter &>(py::handle{self_base});
    for (const auto &[obj, _] : self.m_agenda) {
        Py_VISIT(obj.ptr());
    }
    Py_VISIT(self.m_root.ptr());
    if (self.m_leaf_predicate) {
        // The leaf predicate is an owned Python callback; it must be visited so the cyclic GC can
        // see reference cycles that pass through it (otherwise such cycles leak).
        Py_VISIT(self.m_leaf_predicate->ptr());
    }
    return 0;
}

/*static*/ int PyTreeIter::PyTpClear(PyObject *self_base) {
    if (!::is_holder_constructed(self_base)) [[unlikely]] {
        // The holder has not been constructed yet. Skip the traversal to avoid segmentation faults.
        return 0;
    }
    auto &self = thread_safe_cast<PyTreeIter &>(py::handle{self_base});
    for (auto &[obj, _] : self.m_agenda) {
        Py_CLEAR(obj.ptr());
    }
    self.m_agenda.clear();
    Py_CLEAR(self.m_root.ptr());
    if (self.m_leaf_predicate) {
        // Drop the owned leaf predicate reference to break cycles that pass through it.
        Py_CLEAR(self.m_leaf_predicate->ptr());
    }
    return 0;
}

}  // namespace optree
