#pragma once

#include <MetaNN/data/facilities/traits.h>
namespace MetaNN
{
// matrices
template<typename TElem, typename TDevice>
class Matrix;

template <typename TElem, typename TDevice>
constexpr bool IsMatrix<Matrix<TElem, TDevice>> = true;
}
