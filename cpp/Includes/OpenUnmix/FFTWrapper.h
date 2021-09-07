#ifndef __FFT_WRAPPER_H__
#define __FFT_WRAPPER_H__

struct PFFFT_Setup;
class FFTWrapper {
 public:
  FFTWrapper(const unsigned int fftSize);
  ~FFTWrapper();

  /** Don't use these funtions for targeting index of freq bin
      PFFFT does not support ordered fft, please use
     forwardTransformOrdered/backwardTransformOrdered
    */
  void forwardTransform(const float* input, float* output);
  void backwardTransform(const float* input, float* output);

  void forwardTransformOrdered(const float* input, float* output);
  void backwardTransformOrdered(const float* input, float* output);

  void frequencyDomainConvolve(const float* input, const float* convolution, float* output);
  void frequencyDomainConvolveAcc(const float* input_x, const float* input_y, float* output);
  /**
  ForwardReorder
    R1, R2, R3, R4, I1, I2, I3, I4, ... -> R1, I1, R2, I2, R3, I3, R4, I4, ...
  BackwardReorder
    R1, I1, R2, I2, R3, I3, R4, I4, ... -> R1, R2, R3, R4, I1, I2, I3, I4, ...
  */
  void frequecyDomainFowardReorder(const float* input, float* output);
  void frequecyDomainBackwardReorder(const float* input, float* output);

  unsigned int getFFTSize() const;
  unsigned int getSimdSize() const;

 private:
  PFFFT_Setup* pfft_;
  int fftSize_;
  float normalizationFactor_;

#ifdef _MSC_VER
  float* input_;
  float* output_;
  float* tmp_;
#else
  float __attribute__((aligned(32))) * input_;
  float __attribute__((aligned(32))) * output_;
  float __attribute__((aligned(32))) * tmp_;
#endif
};

#endif  // __FFT_WRAPPER_H__
