cpp
#pragma once
#include <string>

struct MicroLLMConfig {
    std::string modelPath;      
    std::string lorePath;       
    float phi = 1.6180339887f;  
    float evolutionRate = 0.01f;
};

struct MicroLLMContext {
    std::string role;           
    std::string situation;      
    std::string worldState;     
    std::string history;        
};

struct MicroLLMOutput {
    std::string text;           
    std::string intent;         
    std::string params;         
};

class MicroLLM {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    MicroLLMOutput query(const MicroLLMContext& ctx);

    void reward(float amount);
    void penalize(float amount);
    void tick(float dt);

private:
    void loadModel(const std::string& path);
    void loadLore(const std::string& path);

    float mood = 0.0f;
    float confidence = 0.5f;
    float phi;
    float evolutionRate;

    // Internal subsystems (hidden)
    struct EmbeddingSystem* embeddings;
    struct GrammarSystem* grammar;
    struct NeuralHead* neural;
    struct EvolutionSystem* evolution;
};
