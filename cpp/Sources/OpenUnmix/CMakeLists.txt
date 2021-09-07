set(target OpenUnmix)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${target} ${sources})

target_link_libraries(
    ${target} PRIVATE
    PFFFT
    ${TORCH_LIBRARIES}
)
