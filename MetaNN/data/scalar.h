#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>

namespace MetaNN
{
template <typename TElem, typename TDevice = DeviceTags::CPU>
struct Scalar
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not an available type");
                  
    using ElementType = TElem;
    using DeviceType = TDevice;
    using StorageType = ElementType;
    
    Scalar(TElem elem = TElem())
        : m_elem(elem) {}
     
    StorageType& Value()
    {
        return m_elem;
    }
   
    StorageType Value() const
    {
        return m_elem;
    }
    
    bool operator== (const Scalar& val) const
    {
        return m_elem == val.m_elem;
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    StorageType m_elem;
};

template <typename TElem, typename TDevice>
constexpr bool IsScalar<Scalar<TElem, TDevice>> = true;
}