#include <stdlib.h>
#include <iostream>

#include "OpenUnmix/Spec.h"
#include "OpenUnmix/FFTWrapper.h"
#include "OpenUnmix/FFT.h"

FFT::FFT(unsigned int fft_size) :
    fft_size_(fft_size),
    fft_bins_(fft_size / 2 + 1),
    wav_frame_(fft_size * 2),
    fft_(new FFTWrapper(fft_size))
{
  // Hanning Window
  hann_ = (float*) malloc(sizeof(float) * fft_size);
  for(unsigned int i=0;i<fft_size;i++) {
    hann_[i] = 0.5 * (1 - cos(2 * M_PI * i / fft_size));
  }

  // src, tgt
  srcs_fft = (float*) malloc(sizeof(float) * fft_size);
  tgts_fft = (float*) malloc(sizeof(float) * fft_size);
}

FFT::~FFT() {
  delete fft_;
  free(hann_);
  free(srcs_fft);
  free(tgts_fft);
}

void FFT::fft(const std::vector<float>& floats, Spec& spec) {
  // Left Channel
  _fftForward(floats.data(), spec.mag.data(), spec.pha.data(), true);

  // Right Channel
  unsigned int offset = fft_size_ / 2 + 1;
  _fftForward(floats.data(), spec.mag.data() + offset, spec.pha.data() + offset, false);
}

void FFT::ifft(const Spec& spec, std::vector<float>& outputs) {
  // Left Channel
  _fftBackward(spec.mag.data(), spec.pha.data(), outputs.data(), true);

  // Right Channel
  unsigned int offset = fft_size_ / 2 + 1;
  _fftBackward(spec.mag.data() + offset, spec.pha.data() + offset, outputs.data(), false);
}

std::vector<Spec> FFT::stft(const std::vector<float>& floats, unsigned int hop_size) {
  int nbFrames = 1 + (floats.size() / 2 - fft_size_) / hop_size;
  std::vector<Spec> outSpec(nbFrames, Spec(fft_bins_));
  for(int i = 0; i < nbFrames; i++) {
    int copy_size = std::min(int(floats.size() - i * hop_size * 2), int(fft_size_ * 2));
    std::copy(floats.begin() + i * hop_size * 2, floats.begin() + i * hop_size * 2 + copy_size, wav_frame_.begin());
    fft(wav_frame_, outSpec[i]);
  }
  return outSpec;
}

std::vector<float> FFT::istft(const std::vector<Spec>& specs, unsigned int output_size, unsigned int hop_size) {
  std::vector<float> outputs(output_size);
  for(int spec_idx = 0; spec_idx < specs.size(); spec_idx++) {
    int copy_size = std::min(int(outputs.size() - spec_idx * hop_size * 2), int(fft_size_ * 2));

    ifft(specs[spec_idx], wav_frame_);

    for(int buf_idx = 0; buf_idx < copy_size; buf_idx++) {
      outputs[spec_idx*hop_size * 2 + buf_idx] += wav_frame_[buf_idx] * (2.0f/3.0f);
    }
  }
  return outputs;
}

void FFT::_fftBackward(const float* mags, const float* phas, float* tgts, bool left) {
  unsigned int fft_size = fft_size_;

  // Mag,Phase to Real,Image
  srcs_fft[0] = mags[0] * cos(phas[0]);
  srcs_fft[1] = mags[fft_size/2] * cos(phas[fft_size/2]);
  for(unsigned int i=1;i<fft_size/2;i++) {
    srcs_fft[i*2] = mags[i] * cos(phas[i]);
    srcs_fft[i*2+1] = mags[i] * sin(phas[i]);
  }

  // IFFT
  fft_->backwardTransformOrdered(srcs_fft, tgts_fft);

  // Windowing
  if(left) {
    for (unsigned int i = 0; i < fft_size; i++) {
      tgts[i*2] = tgts_fft[i] * hann_[i];
    }
  } else {
    for (unsigned int i = 0; i < fft_size; i++) {
      tgts[i*2+1] = tgts_fft[i] * hann_[i];
    }
  }
}

void FFT::_fftForward(const float* srcs, float* mags, float* phas, bool left) {
  unsigned int fft_size = fft_size_;

  // Windowing
  if(left) {
    for (unsigned int i = 0; i < fft_size; i++) {
      srcs_fft[i] = srcs[i * 2] * hann_[i];
    }
  } else {
    for (unsigned int i = 0; i < fft_size; i++) {
      srcs_fft[i] = srcs[i * 2 + 1] * hann_[i];
    }
  }

  // FFT
  fft_->forwardTransformOrdered(srcs_fft, tgts_fft);

  // Real,Image to Mag,Phase
  mags[0] = abs(tgts_fft[0]);
  phas[0] = atan2(0, tgts_fft[0]);
  mags[fft_size/2] = abs(tgts_fft[1]);
  phas[fft_size/2] = atan2(0, tgts_fft[1]);
  for(unsigned int i=1;i<fft_size/2;i++) {
    float real = tgts_fft[i * 2];
    float imag = tgts_fft[i * 2 + 1];
    mags[i] = sqrt(real * real + imag * imag);
    phas[i] = atan2(imag, real);
  }
}
