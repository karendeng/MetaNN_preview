#pragma once

#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <memory>

namespace MetaNN
{
template <typename TData>
class EvalBuffer
{
public:
    using DataType = TData;
    
    EvalBuffer()
        : m_buf(new std::shared_ptr<TData>())
    {}

    auto Handle() const
    {
        return EvalHandle<TData>(m_buf);
    }
    
    auto ConstHandle() const
    {
        return ConstEvalHandle<EvalHandle<TData>>(Handle());
    }
    
    bool IsEmpty() const
    {
        return (*m_buf == nullptr);
    }
    
private:
    std::shared_ptr<std::shared_ptr<TData>> m_buf;
};
}