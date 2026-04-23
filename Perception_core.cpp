class PerceptionCore {
public:
    bool init(const MicroLLMConfig& cfg);
    void shutdown();

    void processContext(const MicroLLMContext& ctx,
                        Vec& outEmbedding,
                        std::unordered_map<std::string, float>& outPatternPriors);

private:
    Vec encodeContext(const MicroLLMContext& ctx);
    Vec mapToEmbedding(const Vec& features);
    void computePatternPriors(const Vec& embedding,
                              std::unordered_map<std::string, float>& priors);

    // Embedding tables
    std::unordered_map<std::string, Vec> roleEmbeddings;
    std::unordered_map<std::string, Vec> situationEmbeddings;
    std::unordered_map<std::string, Vec> tagEmbeddings;
    std::unordered_map<std::string, Vec> patternEmbeddings;
};
