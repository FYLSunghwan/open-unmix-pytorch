#include "pffft.h"
#include "OpenUnmix/FFTWrapper.h"
#include <assert.h>

#define FFTWRAPPER_NYQUIST_REAL_INDEX_ORDERER 1
#define FFTWRAPPER_NYQUIST_REAL_INDEX_UNORDERER 4

FFTWrapper::FFTWrapper(const unsigned int fftSize)
    : pfft_(NULL), input_(NULL), output_(NULL), tmp_(NULL) {
  // FIXME: Using preprocessor.
  assert(fftSize % 32 == 0);

  fftSize_ = fftSize;
  normalizationFactor_ = 1.0f / (float)fftSize;

  // FIXME: Do not allocate memory in the constructor.
  pfft_ = pffft_new_setup(fftSize, PFFFT_REAL);
  input_ = (float*)pffft_aligned_malloc(fftSize * sizeof(float));
  output_ = (float*)pffft_aligned_malloc(fftSize * sizeof(float));
  tmp_ = (float*)pffft_aligned_malloc(fftSize * sizeof(float));
}

FFTWrapper::~FFTWrapper() {
  if (NULL != pfft_) {
    pffft_destroy_setup(pfft_);
    pfft_ = NULL;
  }

  if (NULL != input_) {
    pffft_aligned_free(input_);
    input_ = NULL;
  }

  if (NULL != output_) {
    pffft_aligned_free(output_);
    output_ = NULL;
  }

  if (NULL != tmp_) {
    pffft_aligned_free(tmp_);
    tmp_ = NULL;
  }
}

void FFTWrapper::forwardTransform(const float* input, float* output) {
  float* in = input_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++)
    in[i] = input[i];

  // #ifdef SIMD_DISABLE
  //   pffft_transform_ordered(pfft_, input_, output_, tmp_, PFFFT_FORWARD);
  //   out[FFTWRAPPER_NYQUIST_REAL_INDEX_ORDERER] = 0.f;
  // #else
  pffft_transform(pfft_, input_, output_, tmp_, PFFFT_FORWARD);
  // out[FFTWRAPPER_NYQUIST_REAL_INDEX_UNORDERER] = 0.f;
  // #endif

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i];
}

void FFTWrapper::backwardTransform(const float* input, float* output) {
  float* in = input_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++)
    in[i] = input[i];

  // #ifdef SIMD_DISABLE
  //   pffft_transform_ordered(pfft_, input_, output_, tmp_, PFFFT_BACKWARD);
  // #else
  pffft_transform(pfft_, input_, output_, tmp_, PFFFT_BACKWARD);
  // #endif

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i] * normalizationFactor_;
}

void FFTWrapper::forwardTransformOrdered(const float* input, float* output) {
  float* in = input_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++)
    in[i] = input[i];

  pffft_transform_ordered(pfft_, input_, output_, tmp_, PFFFT_FORWARD);

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i];
}

void FFTWrapper::backwardTransformOrdered(const float* input, float* output) {
  float* in = input_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++)
    in[i] = input[i];

  pffft_transform_ordered(pfft_, input_, output_, tmp_, PFFFT_BACKWARD);

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i] * normalizationFactor_;
}

void FFTWrapper::frequencyDomainConvolve(const float* input_x,
                                         const float* input_y,
                                         float* output) {
  float* in_x = input_;
  float* in_y = tmp_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++) {
    in_x[i] = input_x[i];
    in_y[i] = input_y[i];
    out[i] = 0.f;
  }

  float scaling = 1.0f;
  pffft_zconvolve_accumulate(pfft_, in_x, in_y, out, scaling);

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i];
}

void FFTWrapper::frequencyDomainConvolveAcc(const float* input_x,
                                            const float* input_y,
                                            float* output) {
  float* in_x = input_;
  float* in_y = tmp_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++) {
    in_x[i] = input_x[i];
    in_y[i] = input_y[i];
    out[i] = output[i];
  }

  float scaling = 1.0f;
  pffft_zconvolve_accumulate(pfft_, in_x, in_y, out, scaling);

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i];
}

void FFTWrapper::frequecyDomainFowardReorder(const float* input, float* output) {
  float* in = input_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++)
    in[i] = input[i];

  pffft_zreorder(pfft_, in, out, PFFFT_FORWARD);

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i];
}

void FFTWrapper::frequecyDomainBackwardReorder(const float* input, float* output) {
  float* in = input_;
  float* out = output_;
  for (unsigned int i = 0; i < fftSize_; i++)
    in[i] = input[i];

  pffft_zreorder(pfft_, in, out, PFFFT_BACKWARD);

  for (unsigned int i = 0; i < fftSize_; i++)
    output[i] = out[i];
}

unsigned int FFTWrapper::getFFTSize() const {
  return fftSize_;
}

unsigned int FFTWrapper::getSimdSize() const {
  return (unsigned int)pffft_simd_size();
}
