project(loev_v_a)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(prj.labs)
add_subdirectory(coursework)
