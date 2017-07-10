#pragma once

#include <type_traits>

namespace MetaNN
{
/// lower access
template<typename TData>
struct LowerAccessImpl;

template <typename TMatrix>
auto LowerAccess(const TMatrix& p)
{
    TMatrix& castRes = const_cast<TMatrix&>(p);
    return LowerAccessImpl<TMatrix>(castRes);
}
}
