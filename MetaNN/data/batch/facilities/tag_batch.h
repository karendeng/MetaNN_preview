#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/batch/batch.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <cassert>
#include <stdexcept>
#include <type_traits>

namespace MetaNN
{
namespace NSBatchMatrix
{
template <typename TOpHandle, typename TDevice>
class EvalUnit;

template <typename TOpHandle>
class EvalUnit<TOpHandle, DeviceTags::CPU> : public BaseEvalUnit<DeviceTags::CPU>
{
    using TOperandData = std::decay_t<decltype(std::declval<TOpHandle>().Data())>;
public:
    using ElementType = typename TOperandData::ElementType;
    using DeviceType = typename TOperandData::DeviceType;
    static_assert(std::is_same<DeviceType, DeviceTags::CPU>::value,
                  "Device type mismatch");

    EvalUnit(std::vector<TOpHandle> opers,
             EvalHandle<Batch<Matrix<ElementType, DeviceType>>> evalRes)
        : m_opers(std::move(opers))
        , m_evalRes(std::move(evalRes)) {}

    void Eval() override
    {
        auto& res = m_evalRes.Data();
        assert(res.IsEmpty());
        res.Reserve(m_opers.size());
        for (size_t i = 0; i < m_opers.size(); ++i)
        {
            res.PushBack(m_opers[i].Data());
        }
    }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& opMap) const
    {
        int res = -1;
        for (const auto& h : m_opers)
        {
            auto it = opMap.find(h.DataPtr());
            if (it != opMap.end())
            {
                res = std::max(res, (int)(it->second));
            }
        }
        return (size_t)res;
    }

private:
    std::vector<TOpHandle> m_opers;
    EvalHandle<Batch<Matrix<ElementType, DeviceType>>> m_evalRes;
};
}

template <typename TData, typename TDataCate>
class TagBatch;

template <typename TData>
class TagBatch<TData, CategoryTags::Matrix>
{
    static_assert(std::is_same<std::decay_t<TData>, TData>::value,
                  "TData is not an available type");
public:
    using DeviceType = typename TData::DeviceType;
    using ElementType = typename TData::ElementType;

    TagBatch(size_t rowNum, size_t colNum)
        : m_rowNum(rowNum)
        , m_colNum(colNum) {}
        
    template <typename TIterator>
    TagBatch(size_t rowNum, size_t colNum,
              TIterator b, TIterator e)
        : m_rowNum(rowNum)
        , m_colNum(colNum)
    {
        for (auto& p : m_buffer)
        {
            if ((p.RowNum() != rowNum) || (p.ColNum() != colNum))
            {
                throw std::runtime_error("Dimension mismatch");
            }
        }
    }
    
public:
    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

    void PushBack(TData mat)
    {
        assert(m_evalBuf.IsEmpty());
        if ((mat.RowNum() != m_rowNum) || (mat.ColNum() != m_colNum))
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer.emplace_back(std::move(mat));
    }
    
    template <typename...TArgs>
    void EmplaceBack(TArgs&&... args)
    {
        assert(m_evalBuf.IsEmpty());
        TData tmp(std::forward<TArgs>(args)...);
        if ((tmp.RowNum() != m_rowNum) || (tmp.ColNum() != m_colNum))
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer.emplace_back(std::move(tmp));
    }
    
    size_t BatchNum() const
    {
        return m_buffer.size();
    }
    
    void Reserve(size_t num)
    {
        assert(m_evalBuf.IsEmpty());
        m_buffer.reserve(num);
    }
    
    void Clear()
    {
        m_buffer.clear();
        m_evalBuf = EvalBuffer<Batch<Matrix<ElementType, DeviceType>>>{};
    }
    
    bool IsEmpty() const
    {
        return m_buffer.empty();
    }
    
    const auto& operator[] (size_t id) const
    {
        return m_buffer[id];
    }
    
    auto& operator[] (size_t id)
    {
        return m_buffer[id];
    }
    
    auto begin() { return m_buffer.begin(); }
    auto begin() const { return m_buffer.begin(); }
    auto end() { return m_buffer.end(); }
    auto end() const { return m_buffer.end(); }
    
    template <typename TBatch,
              std::enable_if_t<IsBatchMatrix<TBatch>>* = nullptr>
    bool operator== (const TBatch& val) const
    {
        auto tmp = static_cast<const TagBatch&>(val);
        return m_buffer == tmp.m_buffer;
    }

    template <typename TOtherData,
              std::enable_if_t<!IsBatchMatrix<TOtherData>>* = nullptr>
    bool operator== (const TOtherData&) const
    {
        return false;
    }

    template <typename TOtherData>
    bool operator!= (const TOtherData& val) const
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        if (m_evalBuf.IsEmpty())
        {
            using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;
            std::vector<TOpEvalHandle> handleBuf;
            handleBuf.reserve(BatchNum());
            for (size_t i = 0; i < BatchNum(); ++i)
            {
                handleBuf.push_back(m_buffer[i].EvalRegister());
            }
            
            using EvalUnit = NSBatchMatrix::EvalUnit<TOpEvalHandle, DeviceType>;
            using GroupType = TrivalEvalGroup<EvalUnit>;
            
            auto outHandle = m_evalBuf.Handle();
            outHandle.Allocate(m_rowNum, m_colNum);
            const void* dataPtr = outHandle.DataPtr();
            EvalUnit unit(std::move(handleBuf), std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
        }
        return m_evalBuf.ConstHandle();
    }
    
private:
    size_t m_rowNum;
    size_t m_colNum;
    std::vector<TData> m_buffer;
    EvalBuffer<Batch<Matrix<ElementType, DeviceType>>> m_evalBuf;
};
}