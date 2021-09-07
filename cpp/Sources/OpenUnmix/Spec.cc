//
// Created by Daniel Kim on 2020/12/11.
//
#include <vector>
#include "OpenUnmix/Spec.h"

Spec::Spec() :
    mag(std::vector<float>(2 * 2049)),
    pha(std::vector<float>(2 * 2049)),
    length(2049 * 2)
{}

Spec::Spec(int fft_bins) :
    mag(std::vector<float>(2 * fft_bins)),
    pha(std::vector<float>(2 * fft_bins)),
    length(fft_bins * 2)
{}

Spec::Spec(std::vector<float> mag, std::vector<float> pha) :
    mag(mag),
    pha(pha),
    length(mag.size())
{}
