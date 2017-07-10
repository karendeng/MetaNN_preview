#pragma once

#include <unordered_map>
#include <MetaNN/data/facilities/tags.h>

namespace MetaNN
{
template <typename TDevice>
class BaseEvalUnit;

template <>
class BaseEvalUnit<DeviceTags::CPU>
{
public:
    using DeviceType = DeviceTags::CPU;

    virtual ~BaseEvalUnit() = default;
    virtual void Eval() = 0;
};
}