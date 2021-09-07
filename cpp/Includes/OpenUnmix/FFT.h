#ifndef FFT_H
#define FFT_H

#include <vector>

class Spec;
class FFTWrapper;

class FFT {
public:
  FFT(unsigned int fft_size);
  ~FFT();
  void fft(const std::vector<float>& floats, Spec& spec);
  void ifft(const Spec& spec, std::vector<float>& outputs);
  std::vector<Spec> stft(const std::vector<float>& floats, unsigned int hop_size=1024);
  std::vector<float> istft(const std::vector<Spec>& specs, unsigned int output_size, unsigned int hop_size=1024);

private:
  void _fftBackward(const float* mags, const float* phas, float* tgts, bool left);
  void _fftForward(const float* srcs, float* mags, float* phas, bool left);
  std::vector<float> wav_frame_;
  unsigned int fft_size_;
  unsigned int fft_bins_;
  FFTWrapper* fft_;
  float* hann_;
  float* srcs_fft;
  float* tgts_fft;
};

#endif //FFT_H
