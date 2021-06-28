#include <ngraph/ngraph.hpp>
#include <inference_engine.hpp>

class OCLLayerOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"OCLLayerOp", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    OCLLayerOp() = default;

    OCLLayerOp(const ngraph::OutputVector& inputs);

    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

class OCLLayerImpl : public InferenceEngine::ILayerExecImpl
{
public:
    explicit OCLLayerImpl(const std::shared_ptr<ngraph::Node>& node);

    ~OCLLayerImpl();

    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config,
                                     InferenceEngine::ResponseDesc *resp) noexcept;

    virtual InferenceEngine::StatusCode
    getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                               InferenceEngine::ResponseDesc* resp) noexcept;

    virtual InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                                std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                                InferenceEngine::ResponseDesc *resp) noexcept;

private:
    std::vector<ngraph::Shape> inpShapes;
    ngraph::Shape outShape;
};

class InfEngineNgraphExtension : public InferenceEngine::IExtension
{
public:
    void Unload() noexcept override {}
    void Release() noexcept override { delete this; }
    void GetVersion(const InferenceEngine::Version*&) const noexcept override {}

    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override;

    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node,
                                                       const std::string& implType) override;
};
