#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <deque>

namespace MetaNN
{
template <typename TDevice, typename TDummy = void>
struct Allocator;

template <typename TDummy>
struct Allocator<DeviceTags::CPU, TDummy>
{
private:
    struct _AllocHelper
    {
        std::unordered_map<size_t, std::deque<void*> > avail_mem;
        ~_AllocHelper()
        {
            ReleaseGarbage();
        }

        void ReleaseGarbage()
        {
            for (auto& p : avail_mem)
            {
                auto& ref_vec = p.second;
                for (auto& p1 : ref_vec)
                {
                    char* buf = (char*)(p1);
                    delete []buf;
                }
                ref_vec.clear();
            }
        }
    };

    struct _desImpl
    {
        _desImpl(std::deque<void*>& p_refPool)
            : ref_pool(p_refPool) {}

        template<typename T>
        void operator () (T* p_val) const
        {
            std::lock_guard<std::mutex> guard(s_mutex);
            ref_pool.push_back((void*)p_val);
        }
    private:
        std::deque<void*>& ref_pool;
    };

public:
    template<typename T>
    static std::shared_ptr<T> Allocate(size_t p_elemSize)
    {
        if (p_elemSize == 0)
        {
            return std::shared_ptr<T>();
        }
        p_elemSize *= sizeof(T);
        if ((p_elemSize % 1024) != 0)
        {
            p_elemSize = (p_elemSize / 1024 + 1) * 1024;
        }

        std::lock_guard<std::mutex> guard(s_mutex);

        auto& slot = s_alloc_helper.avail_mem[p_elemSize];
        if (slot.empty())
        {
            void* raw_buf = (void*)new char[p_elemSize];
            return std::shared_ptr<T>((T*)raw_buf, _desImpl(slot));
        }
        else
        {
            void* mem = slot.back();
            slot.pop_back();
            return std::shared_ptr<T>((T*)mem, _desImpl(slot));
        }
    }
    
private:
    static std::mutex s_mutex;
    static _AllocHelper s_alloc_helper;
};

template <typename TDummy>
std::mutex Allocator<DeviceTags::CPU, TDummy>::s_mutex;

template <typename TDummy>
typename Allocator<DeviceTags::CPU, TDummy>::_AllocHelper
                Allocator<DeviceTags::CPU, TDummy>::s_alloc_helper;
}
