
file(GLOB SOURCE *.cpp)
file(GLOB HEADERS *.hpp)
add_executable(Lab1 ${SOURCE} ${HEADERS})
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Lab1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
