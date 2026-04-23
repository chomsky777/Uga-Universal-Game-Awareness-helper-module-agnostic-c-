#pragma once
#include <string>

struct MicroLLMConfig {
    std::string modelPath;      // tiny model or rule file
    std::string lorePath;       // optional: world-specific data
    float phi = 1.6180339887f;  // golden ratio
    float evolutionRate = 0.01f;
};

struct MicroLLMContext {
    std::string role;           // "npc", "director", "boss_ai", etc.
    std::string situation;      // "combat", "dialogue", "chunk_gen"
    std::string worldState;     // JSON or key-value string
    std::string history;        // short memory buffer
};

struct MicroLLMOutput {
    std::string text;           // dialogue or description
    std::string intent;         // "attack", "retreat", "adjust_layout"
    std::string params;         // JSON: { "aggression":0.7, ... }
};

class MicroLLM {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    MicroLLMOutput query(const MicroLLMContext& ctx);

    // Evolution hooks
    void reward(float amount);
    void penalize(float amount);
    void tick(float dt);

private:
    // Internal state (hidden from game)
    void loadModel(const std::string& path);
    void loadLore(const std::string& path);

    float mood = 0.0f;
    float confidence = 0.5f;
    float phi;
    float evolutionRate;

    // Embeddings, templates, tiny neural head, etc.
    // Stored in private structures.
};
