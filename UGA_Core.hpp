// UGA_Core.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// ---------- Shared types ----------

using Vec = std::vector<float>;
using JsonString = std::string;

struct MicroLLMConfig {
    std::string modelPath;
    std::string lorePath;
    float phi = 1.6180339887f;
    float evolutionRate = 0.01f;
};

struct MicroLLMContext {
    std::string role;
    std::string situation;
    JsonString worldState;  // compact JSON or key-value
    JsonString history;     // short memory JSON
};

struct MicroLLMOutput {
    std::string text;
    std::string intent;
    JsonString params;
};

// ---------- Pillar I: Perception Core ----------

class PerceptionCore {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    // Main entry: context -> embedding + priors
    void processContext(const MicroLLMContext& ctx,
                        Vec& outEmbedding,
                        std::unordered_map<std::string, float>& outPatternPriors);

private:
    // Internal helpers
    Vec encodeContext(const MicroLLMContext& ctx);
    void computePatternPriors(const Vec& embedding,
                              std::unordered_map<std::string, float>& priors);

    // Data
    std::unordered_map<std::string, Vec> patternEmbeddings; // name -> vector
};

// ---------- Pillar II: Decision Core ----------

class DecisionCore {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    void decide(const Vec& embedding,
                const std::unordered_map<std::string, float>& patternPriors,
                float phi,
                float mood,
                float confidence,
                std::string& outIntent,
                JsonString& outParams);

private:
    Vec buildDecisionInput(const Vec& embedding,
                           const std::unordered_map<std::string, float>& patternPriors,
                           float phi,
                           float mood,
                           float confidence);

    void runHead(const Vec& decisionInput,
                 std::string& outIntent,
                 JsonString& outParams);

    // Tiny neural head / rule weights
    // (implementation-specific)
};
// ---------- Pillar III: Expression Core ----------

class ExpressionCore {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    std::string generateText(const MicroLLMContext& ctx,
                             const std::string& intent,
                             const std::unordered_map<std::string, float>& patternPriors,
                             float phi);

private:
    std::string selectTemplate(const MicroLLMContext& ctx,
                               const std::string& intent,
                               const std::unordered_map<std::string, float>& patternPriors,
                               float phi);

    std::string fillTemplate(const std::string& tmpl,
                             const MicroLLMContext& ctx);

    // key -> list of templates
    std::unordered_map<std::string, std::vector<std::string>> templates;
};

// ---------- Pillar IV: Adaptation Core ----------

class AdaptationCore {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    void reward(float amount);
    void penalize(float amount);
    void tick(float dt);

    float getMood() const { return mood; }
    float getConfidence() const { return confidence; }

private:
    void applyDrift(float dt);
    void applyPhiMutation(float dt);

    float mood = 0.0f;        // -1..+1
    float confidence = 0.5f;  // 0..1
    float phi;
    float evolutionRate;
    float timeAccumulator = 0.0f;
};

// ---------- UGA Facade ----------

class MicroLLM {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    MicroLLMOutput query(const MicroLLMContext& ctx);

    void reward(float amount);
    void penalize(float amount);
    void tick(float dt);

private:
    PerceptionCore perception;
    DecisionCore decision;
    ExpressionCore expression;
    AdaptationCore adaptation;
};


