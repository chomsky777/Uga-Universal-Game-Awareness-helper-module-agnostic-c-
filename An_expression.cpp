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
