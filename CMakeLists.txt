cmake_minimum_required(VERSION 3.20)
project(TestEnvironment)

# C++ 17 Standartlarını zorunlu kıl
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- KLASÖR YOLLARI (DİNAMİK) ---
# Proje ana dizinindeki Dependencies klasörünü baz alıyoruz
set(DEPS_DIR ${CMAKE_SOURCE_DIR}/Dependencies)

# --- INCLUDE (BAŞLIK) YOLLARI ---
include_directories(
        ${DEPS_DIR}/SFML/include
        ${DEPS_DIR}/imgui
        ${DEPS_DIR}/imgui-sfml
        ${DEPS_DIR}/tiny_dnn  # Ekran görüntüsünde klasör adı 'tiny_dnn' (alt çizgi ile)
)

# --- LIBRARY (KÜTÜPHANE) YOLLARI ---
link_directories(
        ${DEPS_DIR}/SFML/lib
)

# --- KAYNAK DOSYALAR ---
# ImGui ve ImGui-SFML'in .cpp dosyalarını tek tek ekliyoruz
set(SOURCES
        main.cpp
        ${DEPS_DIR}/imgui/imgui.cpp
        ${DEPS_DIR}/imgui/imgui_draw.cpp
        ${DEPS_DIR}/imgui/imgui_tables.cpp
        ${DEPS_DIR}/imgui/imgui_widgets.cpp
        ${DEPS_DIR}/imgui/imgui_demo.cpp
        ${DEPS_DIR}/imgui-sfml/imgui-SFML.cpp
)

# ImGui-SFML'e ImGui'nin nerede olduğunu söylüyoruz
add_definitions(-DIMGUI_USER_CONFIG="imconfig-SFML.h")
# --- EXECUTABLE OLUŞTURMA ---
add_executable(TestEnvironment ${SOURCES})

# --- LINKING (BAĞLAMA) ---
target_link_libraries(TestEnvironment
        sfml-graphics
        sfml-window
        sfml-audio
        sfml-system
        opengl32
)

# --- DOSYA KOPYALAMA İŞLEMLERİ (DLL ve DATASET) ---

# 1. SFML DLL'lerini kopyala
if(EXISTS "${DEPS_DIR}/SFML/bin")
    file(GLOB SFML_DLLS "${DEPS_DIR}/SFML/bin/*.dll")
    file(COPY ${SFML_DLLS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# 2. Dataset klasörünü kopyala
# Eğer klasör varsa kopyalar, yoksa hata vermez ama boş klasör oluşturur.
if(EXISTS "${CMAKE_SOURCE_DIR}/dataset")
    file(COPY "${CMAKE_SOURCE_DIR}/dataset" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
else()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dataset")
endif()