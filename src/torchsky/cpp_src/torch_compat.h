#pragma once

#include <nanobind/nanobind.h>
#include "torchsky_torch.h"
#include <ATen/DLConvertor.h>
#include <Python.h>

namespace nb = nanobind;

// Manual declaration avoids importing pybind11 into this nanobind module.
extern PyObject* THPVariable_Wrap(const at::TensorBase& var);

// Helper function to convert torch::Tensor to Python object - FAST PATH
inline nb::object tensor_to_python(const torch::Tensor& tensor) {
    PyObject* tensor_obj = THPVariable_Wrap(tensor);
    if (!tensor_obj) {
        throw std::runtime_error("Failed to wrap tensor");
    }
    return nb::steal(tensor_obj);
}

// Helper function to convert any DLPack producer to torch::Tensor.
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

inline torch::Tensor as_f64_contig(nb::object obj) {
    auto t = python_to_tensor(obj);
    if (t.scalar_type() != torch::kFloat64) {
        t = t.to(torch::kFloat64);
    }
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}
