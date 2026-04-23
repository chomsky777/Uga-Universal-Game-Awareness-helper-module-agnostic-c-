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

    struct HistoryEvent {
        std::string intent;
        float reward;
        float timestamp;
    };

    std::vector<HistoryEvent> history;
};
