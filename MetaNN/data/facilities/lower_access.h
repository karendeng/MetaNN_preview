#pragma once

namespace MetaNN
{
/// lower access
template<typename TData>
struct LowerAccessImpl;

template <typename TData>
auto LowerAccess(const TData& p)
{
    TData& castRes = const_cast<TData&>(p);
    return LowerAccessImpl<TData>(castRes);
}
}
