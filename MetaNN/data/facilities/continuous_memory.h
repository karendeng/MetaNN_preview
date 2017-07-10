#pragma once

#include <MetaNN/data/facilities/allocators.h>
namespace MetaNN
{
template <typename TElem, typename TDevice>
class ContinuousMemory
{
public:
    explicit ContinuousMemory(size_t p_size)
        : m_mem(Allocator<TDevice>::template Allocate<TElem>(p_size))
        , m_memStart(m_mem.get())
    {}

    ContinuousMemory(std::shared_ptr<TElem> p_mem, TElem* p_memStart)
        : m_mem(std::move(p_mem))
        , m_memStart(p_memStart)
    {}

    TElem* RawMemory() const { return m_memStart; }

    const std::shared_ptr<TElem> SharedPtr() const
    {
        return m_mem;
    }

    size_t UseCount() const
    {
        return m_mem.use_count();
    }

    bool operator== (const ContinuousMemory& val) const
    {
        return (m_mem == val.m_mem) && (m_memStart == val.m_memStart);
    }

    bool operator!= (const ContinuousMemory& val) const
    {
        return !(operator==(val));
    }

private:
    std::shared_ptr<TElem> m_mem;
    TElem*                 m_memStart;
};
}