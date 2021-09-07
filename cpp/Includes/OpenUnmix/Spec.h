#ifndef GAUDIOCGSEP_SPEC_H
#define GAUDIOCGSEP_SPEC_H

#include <vector>

class Spec {
public:
  Spec();
  Spec(int fft_bins);
  Spec(std::vector<float> mag, std::vector<float> pha);
  std::vector<float> mag;
  std::vector<float> pha;
  unsigned int length;
};

#endif //GAUDIOCGSEP_SPEC_H
