#include <type_traits>
#define CATCH_CONFIG_MAIN

#include "ds_cuda_rasterizer/rasterizer_cuda.hpp"
#include "ds_cuda_rasterizer/rasterizer_torch.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <span>
#include <torch/script.h>

torch::Tensor identify_tile_ranges(torch::Tensor keys, const TorchRasterizationSettings& settings);
std::tuple<torch::Tensor, torch::Tensor> prepare_keys(torch::Tensor means_2d, torch::Tensor radii,
                                                      const TorchRasterizationSettings& settings);
void sort_by_keys(torch::Tensor indices, torch::Tensor keys);

std::int64_t get_key(std::int32_t tile_index, std::int32_t depth)
{
    return std::int64_t(tile_index) << 32 | depth;
}

// TEST(Rasterizer, PrepareKeysIndicesSimple)
// {
//     static constexpr int N = 10;

//     auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

//     std::array<float, 3 * N> ndc_array = {1437, 185, 315,  1087, 507,  29,   591, 807, 1429, 209,
//                                           164,  979, 119,  687,  104,  1088, 968, 139, 730,  607,
//                                           1216, 679, 1368, 441,  1485, 995,  893, 690, 21,   835};

//     std::array<int32_t, N> radii_array = {20, 6, 10, 4, 4, 8, 11, 17, 11, 31};
//     torch::Tensor ndc = torch::from_blob(ndc_array.data(), {N, 4}, torch::dtype(torch::kFloat32)).cuda();
//     torch::Tensor radii = torch::from_blob(radii_array.data(), {N}, torch::dtype(torch::kInt32)).cuda();

//     const TorchRasterizationSettings raster_settings = {
//         .image_height = 1084, .image_width = 1920, .tanfovx = 0.7673294196293707, .tanfovy = 0.4332052624853496};

//     auto [keys, indices] = prepare_keys(ndc, radii, raster_settings);

//     constexpr auto KEYS_NUM = 40;
//     EXPECT_EQ(keys.numel(), KEYS_NUM);
//     EXPECT_EQ(keys.numel(), indices.numel());

//     const std::array<int32_t, KEYS_NUM> ref_indices = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
//                                                        2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8};

//     auto indices_cpu = indices.cpu();
//     const std::span<int32_t, KEYS_NUM> indices_span(indices_cpu.data_ptr<int32_t>(), KEYS_NUM);
//     EXPECT_THAT(indices_span, testing::ElementsAreArray(ref_indices));

//     const std::array<int64_t, KEYS_NUM> ref_keys = {
//         5533052272640,  5537347239936,  5541642207232,  5545937174528,  6048448348160,  6052743315456, 6057038282752,
//         6061333250048,  6563844423680,  6568139390976,  6572434358272,  6576729325568,  16266146873344,
//         16270441840640, 16781542948864, 16785837916160, 25410179080192, 25414474047488, 25925575155712,
//         25929870123008, 26440971231232, 26445266198528, 5206648864768,  5210943832064,  21677820870656,
//         22193216946176, 31212652134400, 31216947101696, 31728048209920, 31732343177216, 19259784167424,
//         19264079134720, 19268374102016, 19775180242944, 19779475210240, 19783770177536, 31835444690944,
//         31839739658240, 32350840766464, 32355135733760};

//     auto keys_cpu = keys.cpu();
//     const std::span<int64_t, KEYS_NUM> keys_span(keys_cpu.data_ptr<int64_t>(), KEYS_NUM);
//     EXPECT_THAT(keys_span, testing::ElementsAreArray(ref_keys));
// }

TEST(Rasterizer, IdentifyTileRanges)
{
    static constexpr int N = 10;
    std::array<int64_t, N> keys_array = {get_key(0, 40), get_key(1, 20), get_key(1, 50), get_key(2, 60),
                                         get_key(3, 10), get_key(3, 20), get_key(4, 20), get_key(5, 40),
                                         get_key(5, 30), get_key(5, 50)};

    torch::Tensor keys = torch::from_blob(keys_array.data(), {N}, torch::dtype(torch::kInt64)).cuda();

    TorchRasterizationSettings raster_settings = {};

    raster_settings.image_height = 30;
    raster_settings.image_width = 40;

    auto indices = identify_tile_ranges(keys, raster_settings);

    std::array<int32_t, 7> ref_indices = {0, 1, 3, 4, 6, 7, 10};
    auto cpu_indices = indices.cpu();
    std::span<int32_t> indices_span(cpu_indices.data_ptr<int32_t>(), cpu_indices.numel());

    EXPECT_THAT(indices_span, testing::ElementsAreArray(ref_indices));
}

TEST(Rasterizer, IdentifyTileRangesNoTileEntries)
{
    static constexpr int N = 10;
    std::array<int64_t, N> keys_array = {get_key(0, 40), get_key(1, 20), get_key(1, 50), get_key(3, 60),
                                         get_key(3, 10), get_key(3, 20), get_key(3, 20), get_key(5, 40),
                                         get_key(5, 30), get_key(5, 50)};

    torch::Tensor keys = torch::from_blob(keys_array.data(), {N}, torch::dtype(torch::kInt64)).cuda();

    TorchRasterizationSettings raster_settings = {};

    raster_settings.image_height = 30;
    raster_settings.image_width = 40;

    auto indices = identify_tile_ranges(keys, raster_settings);

    std::array<int32_t, 7> ref_indices = {0, 1, 3, 3, 7, 7, 10};
    auto cpu_indices = indices.cpu();
    std::span<int32_t> indices_span(cpu_indices.data_ptr<int32_t>(), cpu_indices.numel());

    EXPECT_THAT(indices_span, testing::ElementsAreArray(ref_indices));
}

template <typename Func, typename... Args>
auto with_time_report(
    typename std::enable_if<!std::is_void_v<std::invoke_result_t<Func, Args...>>, std::string_view>::type step_name,
    Func function, Args&&... args)
{
    const auto start_ts = std::chrono::high_resolution_clock::now();
    const auto result = function(std::forward<Args>(args)...);
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    const auto duration = (end_ts - start_ts) / 1.0ms;

    std::cout << "Timing step:" << step_name << " took: " << duration << "ms\n";

    return result;
}

template <typename Func, typename... Args>
void with_time_report(
    typename std::enable_if<std::is_void_v<std::invoke_result_t<Func, Args...>>, std::string_view>::type step_name,
    Func function, Args&&... args)
{

    const auto start_ts = std::chrono::high_resolution_clock::now();
    function(std::forward<Args>(args)...);
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    const auto duration = (end_ts - start_ts) / 1.0ms;

    std::cout << "Timing step:" << step_name << " took: " << duration << "ms\n";
}

TEST(Rasterizer, DeepsensePyTorchPipelineProfiling)
{
    torch::jit::script::Module input_data = torch::jit::load("input_data_garden.pt");
    torch::Tensor means_3d = input_data.attr("means_3d").toTensor();
    torch::Tensor means_2d = input_data.attr("means_2d").toTensor();
    torch::Tensor shs = input_data.attr("shs").toTensor();
    torch::Tensor opacities = input_data.attr("opacities").toTensor();
    torch::Tensor scales = input_data.attr("scales").toTensor();
    torch::Tensor rotations = input_data.attr("rotations").toTensor();
    torch::Tensor colors_precomp;
    torch::Tensor cov_3d_precomp;

    const TorchRasterizationSettings settings = {
        .image_height = static_cast<int>(input_data.attr("image_height").toInt()),
        .image_width = static_cast<int>(input_data.attr("image_width").toInt()),
        .tanfovx = static_cast<float>(input_data.attr("tanfovx").toDouble()),
        .tanfovy = static_cast<float>(input_data.attr("tanfovy").toDouble()),
        .bg = torch::zeros({3}, torch::device(torch::kCUDA)),
        .scale_modifier = static_cast<float>(input_data.attr("scale_modifier").toDouble()),
        .view_matrix = input_data.attr("viewmatrix").toTensor().t().contiguous(),
        .proj_matrix = input_data.attr("projmatrix").toTensor().t().contiguous(),
        .sh_degree = static_cast<int>(input_data.attr("sh_degree").toInt()),
        .campos = input_data.attr("campos").toTensor()};

    std::vector<torch::Tensor> result;
    for(auto i = 0; i < 10; ++i)
    {
        result = with_time_report("rasterizer_deepsense", &rasterizer_forward_deepsense, means_3d, means_2d, shs,
                                  colors_precomp, opacities, scales, rotations, cov_3d_precomp, settings);
    }
}

TEST(Rasterizer, DeepsenseCorePipelineProfiling)
{
    torch::jit::script::Module input_data = torch::jit::load("input_data_garden.pt");
    torch::Tensor means_3d = input_data.attr("means_3d").toTensor();
    torch::Tensor means_2d = input_data.attr("means_2d").toTensor();
    torch::Tensor shs = input_data.attr("shs").toTensor();
    torch::Tensor opacities = input_data.attr("opacities").toTensor();
    torch::Tensor scales = input_data.attr("scales").toTensor();
    torch::Tensor rotations = input_data.attr("rotations").toTensor();

    const TorchRasterizationSettings settings = {
        .image_height = static_cast<int>(input_data.attr("image_height").toInt()),
        .image_width = static_cast<int>(input_data.attr("image_width").toInt()),
        .tanfovx = static_cast<float>(input_data.attr("tanfovx").toDouble()),
        .tanfovy = static_cast<float>(input_data.attr("tanfovy").toDouble()),
        .bg = torch::zeros({3}, torch::device(torch::kCUDA)),
        .scale_modifier = static_cast<float>(input_data.attr("scale_modifier").toDouble()),
        .view_matrix = input_data.attr("viewmatrix").toTensor().t().contiguous(),
        .proj_matrix = input_data.attr("projmatrix").toTensor().t().contiguous(),
        .sh_degree = static_cast<int>(input_data.attr("sh_degree").toInt()),
        .campos = input_data.attr("campos").toTensor()};

    torch::Tensor output_image = torch::empty({settings.image_height, settings.image_width, 3},
                                              torch::device(torch::kCUDA).dtype(torch::kFloat32));

    std::vector<torch::Tensor> result;
    for(auto i = 0; i < 10; ++i)
    {
        with_time_report("rasteruizer_core_deepsense", &rasterizer_forward_core_deepsense, means_3d.data_ptr<float>(),
                         shs.data_ptr<float>(), opacities.data_ptr<float>(), scales.data_ptr<float>(),
                         rotations.data_ptr<float>(), means_3d.sizes()[0], settings.view_matrix.data_ptr<float>(),
                         settings.proj_matrix.data_ptr<float>(), settings.campos.data_ptr<float>(),
                         settings.image_width, settings.image_height, settings.tanfovx, settings.tanfovy,
                         settings.scale_modifier, 3, settings.sh_degree, output_image.data_ptr<float>());
    }
}
