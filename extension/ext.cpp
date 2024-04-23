#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <pybind11/numpy.h>
#include <boost/functional/hash.hpp>
#include "iostream"

namespace py = pybind11;


std::unordered_map<int, int> vistas2kitti_dict = {
    {0, 0},
    {55, 10},  // Car
    {52, 11},  // Bicycle
    {54, 13},  // Bus
    {57, 15},  // Motorcycle
    {58, 16},  // On Rails
    {61, 18},  // Truck
    {59, 20},  // Other Vehicle
    {19, 30},  // Person
    {20, 31},  // Bicyclist
    {21, 32},  // Motorcyclist
    {13, 40},  // Road
    {10, 44},  // Parking
    {15, 48},  // Sidewalk
    {7, 49},   // Other Ground
    {8, 49},
    {9, 49},
    {11, 49},
    {12, 49},
    {14, 49},
    {17, 50},  // Building
    {3, 51},   // Fence
    {16, 52},  // Other Structure
    {18, 52},
    {2, 52},
    {4, 52},
    {5, 52},
    {6, 52},
    {24, 60},  // Lane Marking
    {30, 70},  // Vegetation
    {29, 72},  // Terrain
    {45, 80},  // Pole
    {46, 81},  // Traffic Sign
    {49, 81},
    {50, 81}
};

// Function to convert VISTAS class ID to KITTI class ID
int vistas2kitti(int cls) {
    auto it = vistas2kitti_dict.find(cls);
    if (it != vistas2kitti_dict.end()) {
        return it->second;
    } else {
        return 99;  // Default value if class ID is not found
    }
}


std::unordered_map<std::pair<int64_t, int64_t>, std::vector<int>, boost::hash<std::pair<int64_t, int64_t>>> process_image(
    py::array_t<int64_t> x,
    py::array_t<int64_t> y,
    py::array_t<double> depths,
    py::array_t<double> cls,
    py::array_t<double> img,
    py::array_t<uint8_t> semimg) {

    auto x_accessor = x.unchecked<1>();
    auto y_accessor = y.unchecked<1>();
    auto depths_accessor = depths.unchecked<1>();
    auto cls_accessor = cls.unchecked<1>();
    auto img_mut = img.mutable_unchecked<2>();
    auto semimg_accessor = semimg.unchecked<2>();

    std::unordered_map<std::pair<int64_t, int64_t>, std::vector<int>, boost::hash<std::pair<int64_t, int64_t>>> result;

    for (ssize_t i = 0; i < x.size(); i++) {
        int64_t xi = x_accessor(i);
        int64_t yi = y_accessor(i);
        double di = depths_accessor(i);
        double ci = cls_accessor(i);
        if (di < img_mut(yi, xi)) {  // Assuming cls maps to int keys in kitti2vistas
            img_mut(yi, xi) = di;
            result[{yi, xi}] = {vistas2kitti(int(semimg_accessor(yi, xi))), int(ci)};  // Type casting to int
            //std::cout<<i<<"/"<<x.size()<<": "<<yi<<","<<xi<<" no crash"<<std::endl;
        }
    }
    return result;
}

PYBIND11_MODULE(hdmap_ext, m) {
    m.def("process_image", &process_image, "A function that processes images.");
}