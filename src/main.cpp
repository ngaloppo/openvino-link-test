#include "ocl_ext.hpp"
#include <CL/sycl.hpp>

int main (int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        std::cout << "Usage: "<< argv[0] << " <model_path>" << std::endl;
        return 0;
    }
    std::string modelPath = argv[1];
    std::cout << "model path: " << modelPath << std::endl;

    // Load network
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<InfEngineNgraphExtension>(), "CPU");
    InferenceEngine::CNNNetwork net = ie.ReadNetwork(modelPath);

    InferenceEngine::ExecutableNetwork execNet = ie.LoadNetwork(net, "CPU");
    InferenceEngine::InferRequest infRequest = execNet.CreateInferRequest();

    // Run inference
    // get inpShape / outShape from net
    //
    InferenceEngine::InputInfo::Ptr & inputInfo = net.getInputsInfo().begin()->second;
    auto const & inpShape = inputInfo->getInputData()->getDims();
    auto const & outShape = net.getOutputsInfo().begin()->second->getDims();

    std::cout << "Input shape: " << inpShape[0] << "," << inpShape[1] << "," << inpShape[2] << "," << inpShape[3] << std::endl;
    std::cout << "Output shape: " << outShape[0] << "," << outShape[1] << "," << outShape[2] << "," << outShape[3] << std::endl;

    std::vector<float> inpData(inpShape[0] * inpShape[1] * inpShape[2] * inpShape[3], 0);
    std::vector<float> outData(inpShape[0] * inpShape[1] * inpShape[2] * inpShape[3], 0);

    InferenceEngine::BlobMap inputBlobs, outputBlobs;
    inputBlobs[net.getInputsInfo().begin()->first] = InferenceEngine::make_shared_blob<float>({
        InferenceEngine::Precision::FP32,
        inpShape,
        InferenceEngine::Layout::ANY}, inpData.data());
    outputBlobs[net.getOutputsInfo().begin()->first] = InferenceEngine::make_shared_blob<float>({
        InferenceEngine::Precision::FP32,
        outShape,
        InferenceEngine::Layout::ANY}, outData.data());

    infRequest.SetInput(inputBlobs);
    infRequest.SetOutput(outputBlobs);
    infRequest.Infer();

    return 0;
}
