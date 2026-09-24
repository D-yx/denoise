cmake_minimum_required(VERSION 3.18)
project(rnnoise_official_windows C)

set(CMAKE_C_STANDARD 99)
set(RNNOISE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(dump_features
    src/dump_features.c src/denoise.c src/pitch.c src/celt_lpc.c
    src/kiss_fft.c src/parse_lpcnet_weights.c src/rnnoise_tables.c)
target_include_directories(dump_features PRIVATE ${RNNOISE_INCLUDE})
target_compile_definitions(dump_features PRIVATE TRAINING=1)
target_link_libraries(dump_features PRIVATE m)
if(MINGW)
  # dump_features keeps several 2,000-frame work buffers on the stack.
  target_link_options(dump_features PRIVATE "-Wl,--stack,33554432")
endif()

# The official repository downloads or generates these model files.  The
# training pipeline generates them from the selected .pth checkpoint.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/rnnoise_data.c AND
   EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/rnnoise_data.h)
  add_library(rnnoise STATIC
      src/denoise.c src/rnn.c src/pitch.c src/kiss_fft.c src/celt_lpc.c
      src/nnet.c src/nnet_default.c src/parse_lpcnet_weights.c
      src/rnnoise_data.c src/rnnoise_tables.c)
  target_include_directories(rnnoise PUBLIC include PRIVATE src)
  target_link_libraries(rnnoise PUBLIC m)
  add_executable(rnnoise_demo examples/rnnoise_demo.c)
  target_link_libraries(rnnoise_demo PRIVATE rnnoise)
endif()
