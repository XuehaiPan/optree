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

set(
    optree_csrc
    optree.cpp
    registry.cpp
    treespec/constructors.cpp
    treespec/treespec.cpp
    treespec/flatten.cpp
    treespec/unflatten.cpp
    treespec/traversal.cpp
    treespec/serialization.cpp
    treespec/hashing.cpp
    treespec/richcomparison.cpp
    treespec/gc.cpp
)

pybind11_add_module(_C MODULE "${optree_csrc}")

# Free-threaded (PEP 703) builds must compile the extension with `Py_GIL_DISABLED` so the
# `PyObject` / `PyTypeObject` header layout matches the free-threaded interpreter it is loaded into.
# On Windows `pyconfig.h` does NOT define it automatically (its comment says "a build backend will
# define it with the value"), and neither CMake's FindPython nor `pybind11_add_module` do. Without
# it the extension uses the 16-byte GIL `PyObject` header while linking the 32-byte free-threaded
# runtime, shifting every `PyTypeObject` slot down 16 bytes (e.g. the extension reads `tp_alloc`
# where the runtime stores `tp_dictoffset`) and segfaulting on `import`.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sysconfig; print(int(bool(sysconfig.get_config_var('Py_GIL_DISABLED'))))"
    OUTPUT_VARIABLE OPTREE_Py_GIL_DISABLED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(OPTREE_Py_GIL_DISABLED)
    message(STATUS "Free-threaded Python detected: defining Py_GIL_DISABLED=1 for target _C")
    target_compile_definitions(_C PRIVATE Py_GIL_DISABLED=1)
endif()

if(DEFINED Python_EXTRA_LIBRARIES)
    target_link_libraries(_C PRIVATE ${Python_EXTRA_LIBRARIES})
endif()
