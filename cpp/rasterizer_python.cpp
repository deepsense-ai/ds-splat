#include "ds_cuda_rasterizer/rasterizer_torch.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<TorchRasterizationSettings>(m, "RasterizationSettings")
        .def(py::init<>())
        .def_readwrite("image_height", &TorchRasterizationSettings::image_height)
        .def_readwrite("image_width", &TorchRasterizationSettings::image_width)
        .def_readwrite("tanfovx", &TorchRasterizationSettings::tanfovx)
        .def_readwrite("tanfovy", &TorchRasterizationSettings::tanfovy)
        .def_readwrite("bg", &TorchRasterizationSettings::bg)
        .def_readwrite("scale_modifier", &TorchRasterizationSettings::scale_modifier)
        .def_readwrite("view_matrix", &TorchRasterizationSettings::view_matrix)
        .def_readwrite("proj_matrix", &TorchRasterizationSettings::proj_matrix)
        .def_readwrite("sh_degree", &TorchRasterizationSettings::sh_degree)
        .def_readwrite("max_sh_degree", &TorchRasterizationSettings::max_sh_degree)
        .def_readwrite("campos", &TorchRasterizationSettings::campos)
        .def_readwrite("prefiltered", &TorchRasterizationSettings::prefiltered)
        .def_readwrite("debug", &TorchRasterizationSettings::debug);

    m.def("forward_deepsense", &rasterizer_forward_deepsense, "rasterizer forward (CUDA)");
    m.def("backward_deepsense", &rasterizer_backward_deepsense, "rasterizer backward (CUDA)");
}
