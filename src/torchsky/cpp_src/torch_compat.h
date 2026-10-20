#pragma once

#include <nanobind/nanobind.h>
#include "torchsky_torch.h"
#include <ATen/DLConvertor.h>
#include <Python.h>

namespace nb = nanobind;

// Isolated compatibility bridge for query_disc and experimental basis only.
// Manual declaration avoids importing pybind11 into this nanobind module.
extern PyObject* THPVariable_Wrap(const at::TensorBase& var);

inline nb::object tensor_to_python(const torch::Tensor& tensor) {
    PyObject* tensor_obj = THPVariable_Wrap(tensor);
    if (!tensor_obj) {
        throw std::runtime_error("Failed to wrap tensor");
    }
    return nb::steal(tensor_obj);
}

inline torch::Tensor python_to_tensor(nb::object obj) {
    if (!PyObject_HasAttrString(obj.ptr(), "__dlpack__")) {
        throw std::runtime_error("Object does not implement __dlpack__");
    }
    PyObject* capsule_obj = PyObject_CallMethod(obj.ptr(), "__dlpack__", nullptr);
    if (!capsule_obj) {
        throw nb::python_error();
    }
    nb::object capsule = nb::steal(capsule_obj);
    auto* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor")
    );
    if (!managed) {
        throw nb::python_error();
    }
    auto tensor = at::fromDLPack(managed);
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0) {
        PyErr_Clear();
    }
    return tensor;
}
