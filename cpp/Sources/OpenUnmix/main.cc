#include <stdio.h>

#include "OpenUnmix/FFT.h"
#include "OpenUnmix/Spec.h"
#include "torch/script.h"
#include "AudioFile.h"

const int kSampleRate = 44100;
const int kNumChannels = 2;
const int kFFTSize = 4096;
const int kFFTBins = kFFTSize / 2 + 1;
const int kHopSize = 1024;
const int kSegmentSec = 10;
const int kOverlapSamplePerBlock = 16384;

std::string modelPath = "/Users/fylsunghwan/Downloads/model.pt";
std::string audioPath = "/Users/fylsunghwan/Downloads/aroha.wav";
std::string outputPath = "/Users/fylsunghwan/Downloads/aroha_cc.wav";

std::vector<float> processSegment(FFT& fft, torch::jit::Module& model, const std::vector<float>& floats);

int main() {
    // Initialize instances
    auto fft = FFT(kFFTSize);
    auto model = torch::jit::load(modelPath);
    auto audio = AudioFile<float>();
    auto audioInput = std::vector<float>();
    auto audioOutput = std::vector<float>();

    // Prepare Model and Audio
    model.eval();
    audio.load(audioPath);
    audio.printSummary();

    // Preprocess Audio
    audio.setSampleRate(kSampleRate);
    audio.setNumChannels(kNumChannels);

    int numSamples = audio.getNumSamplesPerChannel();
    for(int i = 0; i < numSamples; i++) {
        audioInput.push_back(audio.samples[0][i]);
        audioInput.push_back(audio.samples[1][i]);
    }
    audioOutput = std::vector<float>(audioInput.size());

    // Process
    const unsigned int segment_size = kSegmentSec * kSampleRate * kNumChannels;
    const unsigned int overlap_size = kOverlapSamplePerBlock * kNumChannels;
    unsigned int st_idx = 0;
    unsigned int ed_idx = std::min(static_cast<unsigned long>(segment_size),
                                   static_cast<unsigned long>(audioInput.size()));

    while(ed_idx <= audioInput.size()) {
        auto segmentedBuffer = std::vector<float>(audioInput.begin() + st_idx, audioInput.begin() + ed_idx);
        auto processedBuffer = processSegment(fft, model, segmentedBuffer);
        
        //std::cout <<  << std::endl;
        // Overlapping
        std::copy(processedBuffer.begin() + overlap_size,
                  processedBuffer.begin() + (ed_idx - st_idx),
                  audioOutput.begin() + st_idx + overlap_size);

        for(unsigned int i = 0; i < kOverlapSamplePerBlock; i++) {
            const float coeff = static_cast<float>(i) / static_cast<float>(kOverlapSamplePerBlock);

            audioOutput[st_idx + i * 2] = coeff * processedBuffer[i * 2] + (1 - coeff) * audioOutput[st_idx + i * 2];
            audioOutput[st_idx + i * 2 + 1] = coeff * processedBuffer[i * 2 + 1] + (1 - coeff) * audioOutput[st_idx + i * 2 + 1];
        }

        st_idx += segment_size - overlap_size;
        ed_idx += segment_size - overlap_size;

        if(ed_idx > audioInput.size() && (ed_idx - st_idx) == segment_size) {
            ed_idx = audioInput.size();
        }
    }

    // Interleaved to AudioFile Format
    AudioFile<float>::AudioBuffer buffer;
    buffer.resize(2);

    buffer[0].resize(audioOutput.size() / 2);
    buffer[1].resize(audioOutput.size() / 2);

    for(int i = 0; i < audioOutput.size() / 2; i++) {
        buffer[0][i] = audioOutput[i * 2];
        buffer[1][i] = audioOutput[i * 2 + 1];
    }

    audio.setAudioBuffer(buffer);
    audio.save(outputPath);
    std::cout << "Successfully Separated with openunmix." << std::endl;

    return 0;
}

std::vector<float> processSegment(FFT& fft, torch::jit::Module& model, const std::vector<float>& floats) {
    // STFT
    std::vector<Spec> inputSpec = fft.stft(floats);
    const unsigned int nbFrames = inputSpec.size();

    // Flatten Spectrogram for Model Inference
    std::vector<float> inputSpecFlatten(nbFrames * 1 * kNumChannels * kFFTBins);
    for(int i = 0; i < nbFrames; i++) {
        std::copy(inputSpec[i].mag.begin(), inputSpec[i].mag.end(), inputSpecFlatten.begin() + i * 2 * kFFTBins);
    }

    // Model Inference
    std::vector<torch::jit::IValue> inputTensors;
    inputTensors.emplace_back(torch::from_blob(inputSpecFlatten.data(), {nbFrames, 1, kNumChannels, kFFTBins}));
    auto outputTensor = model.forward(inputTensors).toTensor();

    // Deflatten Inferenced Spectrogram
    std::vector<Spec> outputSpec = std::vector<Spec>(inputSpec.begin(), inputSpec.end());
    for(int i=0;i<nbFrames;i++) {
        std::copy(outputTensor.data_ptr<float>() + i * 2 * kFFTBins,
                  outputTensor.data_ptr<float>() + (i + 1) * 2 * kFFTBins,
                  outputSpec[i].mag.begin());
    }

    // ISTFT
    std::vector<float> outputBuffer = fft.istft(outputSpec, floats.size());
    return outputBuffer;
}