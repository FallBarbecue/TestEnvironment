#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem> // C++17 gereklidir
#include <algorithm>

// --- KÜTÜPHANELER ---
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include "imgui.h"
#include "imgui-SFML.h"

// TinyDNN başlıkları (Eğer kütüphane kurulu değilse burayı yorum satırı yapın ve Dummy AI kullanın)
#define USE_TINY_DNN
#ifdef USE_TINY_DNN
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
#endif

namespace fs = std::filesystem;

// --- YAPILANDIRMA VE SABİTLER ---
const int WINDOW_W = 1280;
const int WINDOW_H = 800;
const int GRID_SIZE = 20; // Piksellerin büyüklüğü
const int GRID_DIM = 32;  // 32x32 Grid
const int SAMPLE_RATE = 44100;

// --- ENUMLAR VE STRUCTLAR ---
enum Element { FIRE, WATER, EARTH, AIR };

struct SoundDNA {
    float pitch = 1.0f;         // Ses perdesi (0.5 - 2.0)
    float distortion = 0.0f;    // Bozulma oranı (0.0 - 1.0)
    float attack = 0.01f;       // Sesin başlama hızı
    bool isSampleBased = false; // Dosya mı Sentez mi?
    std::string samplePath = "";// Dosya ise yolu
    int waveform = 0;           // 0:Sine, 1:Square, 2:Saw (Sentez ise)
};

struct SpellSession {
    int id;
    Element element;
    std::vector<float> gridData; // TinyDNN için 32x32 -> 1024 input
    SoundDNA soundParams;
    bool isLiked = false;
    bool isDisliked = false;
    std::string predictedShape;  // AI Tahmini
};

// --- GLOBAL YÖNETİCİLER ---

// 1. DATASET MANAGER (Dosya Sistemi)
class DatasetManager {
public:
    std::vector<std::string> wavFiles;

    void RefreshDataset(const std::string& path) {
        wavFiles.clear();
        if (!fs::exists(path)) {
            fs::create_directory(path); // Klasör yoksa oluştur
        }
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.path().extension() == ".wav") {
                wavFiles.push_back(entry.path().string());
            }
        }
    }
};

// 2. SES MOTORU (Audio Engine)
class SoundEngine {
public:
    sf::SoundBuffer buffer;
    sf::Sound sound;

    // Sentezlenmiş ses verisi üretir
    void PlaySynthesis(SoundDNA dna) {
        std::vector<sf::Int16> samples;
        int duration = SAMPLE_RATE * 1.5; // 1.5 saniye
        samples.resize(duration);

        double frequency = 440.0 * dna.pitch;
        double x = 0;
        double increment = frequency / SAMPLE_RATE;

        for (int i = 0; i < duration; i++) {
            double raw = 0;
            // Dalga Formu Seçimi
            if (dna.waveform == 0) raw = sin(x * 2 * 3.14159); // Sine
            else if (dna.waveform == 1) raw = (sin(x * 2 * 3.14159) > 0 ? 1.0 : -1.0); // Square
            else raw = 2.0 * (x - floor(x + 0.5)); // Sawtooth (Basit)

            // Distortion
            if (raw > 1.0 - dna.distortion) raw = 1.0;

            // Envelope (Basit Attack/Decay)
            double vol = 1.0;
            if (i < SAMPLE_RATE * dna.attack) vol = (double)i / (SAMPLE_RATE * dna.attack);
            else vol = 1.0 - ((double)(i) / duration); // Fade out

            samples[i] = static_cast<sf::Int16>(raw * vol * 30000);
            x += increment;
        }

        buffer.loadFromSamples(&samples[0], samples.size(), 1, SAMPLE_RATE);
        sound.setBuffer(buffer);
        sound.setPitch(1.0f); // Sentezde pitch'i sample içine gömdük
        sound.play();
    }

    // Dosya tabanlı ses çalar
    void PlaySample(SoundDNA dna) {
        if (!fs::exists(dna.samplePath)) return;

        if (buffer.loadFromFile(dna.samplePath)) {
            sound.setBuffer(buffer);
            sound.setPitch(dna.pitch); // Pitch manipülasyonu
            // Distortion simülasyonu (Ses yüksekliği ile overdrive)
            sound.setVolume(dna.distortion > 0.5 ? 100.0f : 70.0f);
            sound.play();
        }
    }

    void Play(SoundDNA dna) {
        if (dna.isSampleBased && !dna.samplePath.empty()) {
            PlaySample(dna);
        } else {
            PlaySynthesis(dna);
        }
    }
};

// 3. YAPAY ZEKA VE GENETİK ALGORİTMA
class AIManager {
public:
    std::vector<SpellSession> history;

#ifdef USE_TINY_DNN
    network<sequential> net;
    bool isModelLoaded = false;
#endif

    AIManager() {
#ifdef USE_TINY_DNN
        // Basit bir model mimarisi kuruyoruz (CNN)
        // Not: Gerçek kullanım için eğitilmiş .json dosyası yüklenmeli.
        // net.load("trained_model.json");
        // Şimdilik sadece mimariyi tanımlıyoruz, rastgele ağırlıklarla çalışacak.
        net << fc(32 * 32, 64) << relu() << fc(64, 4) << softmax();
#endif
    }

    // Görüntüden Özellik Çıkarımı (Feature Extraction)
    // Şekil verisinden ses parametresi türeten ilk kural seti
    SoundDNA FeatureExtractionToSound(const std::vector<float>& grid, Element elem) {
        SoundDNA dna;

        // 1. Piksel Yoğunluğu (Density) -> Pitch'i etkilesin
        float density = 0;
        for (float p : grid) density += p;
        density /= (GRID_DIM * GRID_DIM); // 0.0 ile 1.0 arası

        // Element Bazlı Kurallar
        switch (elem) {
            case FIRE:
                dna.waveform = 2; // Sawtooth (Sert)
                dna.pitch = 0.8f + density;
                dna.distortion = 0.8f;
                break;
            case WATER:
                dna.waveform = 0; // Sine (Yumuşak)
                dna.pitch = 1.0f - (density * 0.5f);
                dna.distortion = 0.1f;
                break;
            case EARTH:
                dna.waveform = 1; // Square
                dna.pitch = 0.5f;
                dna.distortion = 0.4f;
                break;
            case AIR:
                dna.waveform = 0;
                dna.pitch = 1.5f;
                dna.distortion = 0.9f; // Hışırtı (Noise) gibi olsun
                break;
        }
        return dna;
    }

    // TinyDNN Tahmini (Şekil neye benziyor?)
    std::string PredictShape(const std::vector<float>& grid) {
#ifdef USE_TINY_DNN
        vec_t input;
        for(float f : grid) input.push_back(f);

        vec_t result = net.predict(input);
        int maxIndex = std::distance(result.begin(), std::max_element(result.begin(), result.end()));

        // Etiketler (Eğitime göre değişir, şimdilik dummy)
        std::vector<std::string> labels = {"Ok", "Kalkan", "Patlama", "Spiral"};
        return labels[maxIndex];
#else
        return "AI Modeli Yok";
#endif
    }

    // GENETİK ALGORİTMA: Crossover & Mutation
    SoundDNA EvolveSound(SoundDNA baseDNA) {
        // Havuzda beğenilen büyüler var mı?
        std::vector<SpellSession> likedOnes;
        for (const auto& s : history) if (s.isLiked) likedOnes.push_back(s);

        if (likedOnes.empty()) return baseDNA; // Veri yoksa taban DNA'yı kullan

        // Rastgele bir ebeveyn seç
        SpellSession parent = likedOnes[rand() % likedOnes.size()];

        SoundDNA child = baseDNA;

        // Çaprazlama (Crossover)
        // Elementin temel kuralıyla, beğenilen geçmişin özelliğini karıştır
        child.pitch = (baseDNA.pitch + parent.soundParams.pitch) / 2.0f;

        // Mutasyon (Mutation) - %30 Şans
        if (rand() % 100 < 30) {
            float mutation = (rand() % 100 - 50) / 100.0f; // -0.5 ile +0.5 arası
            child.pitch += mutation;
            child.distortion += mutation / 2.0f;
        }

        // Sınırları Koru
        child.pitch = std::clamp(child.pitch, 0.1f, 3.0f);
        child.distortion = std::clamp(child.distortion, 0.0f, 1.0f);

        return child;
    }
};

// --- YARDIMCI FONKSİYONLAR ---
sf::Color GetElementColor(Element e) {
    switch (e) {
        case FIRE: return sf::Color(255, 69, 0);   // Turuncu Kırmızı
        case WATER: return sf::Color(30, 144, 255); // Dodger Blue
        case EARTH: return sf::Color(139, 69, 19);  // Saddle Brown
        case AIR: return sf::Color(220, 220, 220);  // Gainsboro
        default: return sf::Color::White;
    }
}

// --- MAIN LOOP ---
int main() {
    // Pencere Kurulumu
    sf::RenderWindow window(sf::VideoMode(WINDOW_W, WINDOW_H), "Büyü Geliştirme Laboratuvarı v1.0");
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    // Yöneticiler
    DatasetManager datasetMgr;
    SoundEngine soundEngine;
    AIManager aiMgr;

    // Başlangıç Dataset Taraması
    datasetMgr.RefreshDataset("dataset");

    // Durum Değişkenleri
    bool grid[GRID_DIM][GRID_DIM] = { false };
    Element currentElement = FIRE;
    sf::Clock deltaClock;

    // Popup Kontrol
    bool showSavePopup = false;
    SpellSession tempSession; // Kaydedilmeyi bekleyen büyü

    // --- OYUN DÖNGÜSÜ ---
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            if (event.type == sf::Event::Closed) window.close();

            // Çizim İşlemi (Mouse Sol Tık)
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && !ImGui::GetIO().WantCaptureMouse) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                // Grid ofseti (Ekranda ortalamak için)
                int offsetX = 50;
                int offsetY = 50;

                int gx = (mousePos.x - offsetX) / GRID_SIZE;
                int gy = (mousePos.y - offsetY) / GRID_SIZE;

                if (gx >= 0 && gx < GRID_DIM && gy >= 0 && gy < GRID_DIM) {
                    grid[gx][gy] = true;
                }
            }

            // Sağ Tık (Silgi)
            if (sf::Mouse::isButtonPressed(sf::Mouse::Right) && !ImGui::GetIO().WantCaptureMouse) {
                 // ... Silme kodu aynısı (grid[gx][gy] = false) ...
            }
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        // --- SOL PANEL: KONTROLLER ---
        ImGui::Begin("Kontrol Paneli", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::Text("Element Secimi:");
        if (ImGui::RadioButton("ATES", currentElement == FIRE)) currentElement = FIRE; ImGui::SameLine();
        if (ImGui::RadioButton("SU", currentElement == WATER)) currentElement = WATER; ImGui::SameLine();
        if (ImGui::RadioButton("TOPRAK", currentElement == EARTH)) currentElement = EARTH; ImGui::SameLine();
        if (ImGui::RadioButton("HAVA", currentElement == AIR)) currentElement = AIR;

        ImGui::Separator();

        if (ImGui::Button("TEMIZLE (Clear)", ImVec2(100, 30))) {
            for(int i=0; i<GRID_DIM; i++) for(int j=0; j<GRID_DIM; j++) grid[i][j] = false;
        }

        ImGui::Separator();

        // --- KRİTİK BUTON: BÜYÜ OLUŞTUR ---
        if (ImGui::Button("BUYUYU OLUSTUR & ANALIZ ET", ImVec2(250, 50))) {
            // 1. Grid Verisini Vektöre Dönüştür
            std::vector<float> gridVec;
            for(int j=0; j<GRID_DIM; j++) {
                for(int i=0; i<GRID_DIM; i++) {
                    gridVec.push_back(grid[i][j] ? 1.0f : 0.0f);
                }
            }

            // 2. AI Tahmini (Şekil Sınıflandırma)
            std::string shapeName = aiMgr.PredictShape(gridVec);

            // 3. Özellik Çıkarımı (Feature Extraction)
            SoundDNA baseDNA = aiMgr.FeatureExtractionToSound(gridVec, currentElement);

            // 4. Genetik Algoritma Uygula (Mutasyon/Crossover)
            SoundDNA finalDNA = aiMgr.EvolveSound(baseDNA);

            // 5. Veriyi Hazırla
            tempSession.id = rand(); // Geçici ID
            tempSession.element = currentElement;
            tempSession.gridData = gridVec;
            tempSession.soundParams = finalDNA;
            tempSession.predictedShape = shapeName;

            // 6. Sesi Çal
            soundEngine.Play(finalDNA);

            // 7. Kayıt Pop-up'ını tetikle
            showSavePopup = true;
        }
        ImGui::End();

        // --- SAĞ PANEL: SES KÜTÜPHANESİ VE BEĞENİ ---
        ImGui::Begin("Genetik Ses Havuzu");

        // Tab Bar
        if (ImGui::BeginTabBar("LibraryTabs")) {

            // TAB 1: Dataset (Dosyalar)
            if (ImGui::BeginTabItem("Dataset (.wav)")) {
                if (ImGui::Button("Yenile")) datasetMgr.RefreshDataset("dataset");

                static int selectedWav = -1;
                for (int i=0; i < datasetMgr.wavFiles.size(); i++) {
                    std::string fname = fs::path(datasetMgr.wavFiles[i]).filename().string();
                    if (ImGui::Selectable(fname.c_str(), selectedWav == i)) {
                        selectedWav = i;
                        // Seçilen dosyayı geçici DNA'ya yükle ve çal
                        SoundDNA testDNA;
                        testDNA.isSampleBased = true;
                        testDNA.samplePath = datasetMgr.wavFiles[i];
                        soundEngine.Play(testDNA);
                    }
                }
                ImGui::EndTabItem();
            }

            // TAB 2: Üretilen Büyüler (Geçmiş)
            if (ImGui::BeginTabItem("Gecmis / Begeni")) {
                for (auto& spell : aiMgr.history) {
                    ImGui::PushID(spell.id);

                    // Renkli Başlık
                    ImVec4 col = ImVec4(1,1,1,1);
                    if(spell.element == FIRE) col = ImVec4(1,0.3,0,1);
                    else if(spell.element == WATER) col = ImVec4(0,0.5,1,1);

                    ImGui::TextColored(col, "Buyu #%d (%s)", spell.id, spell.predictedShape.c_str());

                    if (ImGui::Button("Dinle")) soundEngine.Play(spell.soundParams);
                    ImGui::SameLine();

                    if (ImGui::Checkbox("Like", &spell.isLiked)) spell.isDisliked = false;
                    ImGui::SameLine();
                    if (ImGui::Checkbox("Dislike", &spell.isDisliked)) spell.isLiked = false;

                    ImGui::Separator();
                    ImGui::PopID();
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();

        // --- KAYIT POP-UP PENCERESİ ---
        if (showSavePopup) {
            ImGui::OpenPopup("Sonuc");
        }
        if (ImGui::BeginPopupModal("Sonuc", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Yapay Zeka Tahmini: %s", tempSession.predictedShape.c_str());
            ImGui::Text("Olusturulan Ses: Pitch %.2f | Dist %.2f",
                        tempSession.soundParams.pitch, tempSession.soundParams.distortion);
            ImGui::Separator();

            if (ImGui::Button("KAYDET ve HAVUZA EKLE", ImVec2(200, 0))) {
                aiMgr.history.push_back(tempSession);
                // Ekranı temizle
                for(int i=0; i<GRID_DIM; i++) for(int j=0; j<GRID_DIM; j++) grid[i][j] = false;
                showSavePopup = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("SIL (Begenmedim)", ImVec2(140, 0))) {
                showSavePopup = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        // --- RENDER (ÇİZİM) ---
        window.clear(sf::Color(20, 20, 20)); // Koyu Gri Arkaplan

        // Grid Çizimi
        sf::Color elemColor = GetElementColor(currentElement);
        int offsetX = 50;
        int offsetY = 50;

        // Çerçeve
        sf::RectangleShape border(sf::Vector2f(GRID_DIM * GRID_SIZE, GRID_DIM * GRID_SIZE));
        border.setPosition(offsetX, offsetY);
        border.setFillColor(sf::Color::Transparent);
        border.setOutlineColor(sf::Color::White);
        border.setOutlineThickness(2);
        window.draw(border);

        // Pikseller
        for (int i = 0; i < GRID_DIM; i++) {
            for (int j = 0; j < GRID_DIM; j++) {
                if (grid[i][j]) {
                    sf::RectangleShape pixel(sf::Vector2f(GRID_SIZE - 1, GRID_SIZE - 1));
                    pixel.setPosition(offsetX + i * GRID_SIZE, offsetY + j * GRID_SIZE);
                    pixel.setFillColor(elemColor);
                    window.draw(pixel);
                }
            }
        }

        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}