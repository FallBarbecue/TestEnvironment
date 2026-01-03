cmake_minimum_required(VERSION 3.20)
project(TestEnvironment)

# C++ 17 Standartlarını zorunlu kıl (Filesystem ve SFML için gerekli)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- KLASÖR YOLLARI ---
# Dependencies klasörünü bir değişkene atayalım ki sürekli uzun uzun yazmayalım
# Tırnak içinde ve düz taksim ile:
set(DEPS_DIR "C:/Users/Umut/CLionProjects/TestEnvironment/Dependencies")

# --- INCLUDE (BAŞLIK) YOLLARI ---
# Derleyiciye .h dosyalarını nerede bulacağını söylüyoruz
include_directories(
        ${DEPS_DIR}/SFML/include
        ${DEPS_DIR}/imgui
        ${DEPS_DIR}/imgui-sfml
        ${DEPS_DIR}/tiny-dnn
)

# --- LIBRARY (KÜTÜPHANE) YOLLARI ---
# Derleyiciye .lib (veya .a) dosyalarını nerede bulacağını söylüyoruz
link_directories(
        ${DEPS_DIR}/SFML/lib
)

# --- KAYNAK DOSYALAR ---
# Projemiz sadece main.cpp'den ibaret değil.
# ImGui ve ImGui-SFML'in .cpp dosyalarını da derlememiz gerekiyor.
set(SOURCES
        main.cpp
        ${DEPS_DIR}/imgui/imgui.cpp
        ${DEPS_DIR}/imgui/imgui_draw.cpp
        ${DEPS_DIR}/imgui/imgui_tables.cpp
        ${DEPS_DIR}/imgui/imgui_widgets.cpp
        ${DEPS_DIR}/imgui/imgui_demo.cpp
        ${DEPS_DIR}/imgui-sfml/imgui-SFML.cpp
)

# --- EXECUTABLE OLUŞTURMA ---
add_executable(TestEnvironment ${SOURCES})

# --- LINKING (BAĞLAMA) ---
# SFML kütüphanelerini projemize bağlıyoruz.
# Sıralama önemlidir! (Graphics -> Window -> System -> Audio)
target_link_libraries(TestEnvironment
        sfml-graphics
        sfml-window
        sfml-audio
        sfml-system
        opengl32  # ImGui için Windows'ta gereklidir
)

# --- OTOMATİK DLL KOPYALAMA ---
# CLion'da "Run" dediğinde DLL hatası almamak için harika bir komut.
# SFML/bin içindeki tüm .dll dosyalarını, programın çalıştığı yere kopyalar.
file(GLOB SFML_DLLS "${DEPS_DIR}/SFML/bin/*.dll")
file(COPY ${SFML_DLLS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Ayrıca senin 'dataset' klasörünü de kopyalayalım ki program dosyaları bulabilsin
file(COPY "${CMAKE_SOURCE_DIR}/dataset" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})