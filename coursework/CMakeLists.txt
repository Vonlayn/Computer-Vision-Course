project(cw)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cw src.cpp)
target_link_libraries(cw ${OpenCV_LIBS})