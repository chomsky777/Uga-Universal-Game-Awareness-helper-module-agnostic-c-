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
    std::unordered_map<std::string, int> intentIndex;
    std::unordered_map<std::string, std::vector<std::string>> paramSchema;

    // Neural weights (if used)
    Vec W1, b1, W2, b2;
};
