cmake_minimum_required(VERSION 3.18)
project(integral_image)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(integral_image src/main.cpp)
add_executable(test_integral_image src/tests.cpp)

set(SOURCES
    src/serial_version.cpp
    src/parallel_version.cpp
    src/torch_version.cpp
)
target_sources(integral_image PRIVATE ${SOURCES})
target_sources(test_integral_image PRIVATE ${SOURCES})

target_link_libraries(integral_image ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(test_integral_image ${TORCH_LIBRARIES} ${OpenCV_LIBS})

enable_testing()
add_test(NAME test_integral_image COMMAND $<TARGET_FILE:test_integral_image>)
